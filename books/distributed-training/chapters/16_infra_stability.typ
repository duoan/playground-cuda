#import "../template.typ": *

= 工程稳定性：Checkpoint, 容错, 慢卡诊断

MegaScale (Jiang et al. 2024, ByteDance) 报告显示，训练 12,288 GPU 一个月，硬件故障发生 100+ 次。跑 6 万卡 (Llama-3) 或 10 万卡 (xAI Colossus) 的时代，MTBF 只有几十小时——不容错的系统根本训不完。这一章讲面试里问"你怎么保证 6 万卡训 3 个月不断"的答案。

== 稳定性问题的规模

*硬件故障率*（业界共识）：
- GPU 硬件失效 (ECC error, HBM error)：单卡 MTBF ~10 年
- 10000 卡集群：期望 MTBF ≈ 10 年 / 10000 = 9 小时
- 加上 NIC / 网络 / 电源 / 交换机故障：总 MTBF ~2-5 小时

对 10K+ 卡训练，*每小时都可能死一台机器*。系统必须能：

+ 快速检测故障 (< 1 min)
+ 快速隔离故障节点 (< 5 min)
+ 从最近 checkpoint 恢复 (< 30 min)
+ 用余下健康节点继续 (elastic training)

== Checkpoint: 从秒级到分钟级

Checkpoint 是恢复训练的唯一手段。基本组成：

- Model weight（每 rank 的 shard，可能几 GB - 几十 GB）
- Optimizer state（比 weight 大 3-6 倍）
- LR scheduler state, RNG state, step number
- DataLoader state (streaming 位置)

7B model checkpoint ≈ 100 GB (weight + optim FP32)。70B ≈ 1 TB。670B ≈ 10 TB。

*保存频率*：MegaScale 报告 15-30 min 一次。太频繁 IO 拖 training，太稀 fail 后 lose progress。

=== Async Checkpoint

Naive: `torch.save(state_dict())` — 阻塞 training，1 TB 存到 NAS 要几分钟，training 停 → MFU 掉。

*Async pattern*:

+ Training step 结束时快速把 GPU tensor D2H copy 到 CPU pinned memory (10 秒级)
+ Background thread 慢慢写 CPU tensor → NAS/S3 (分钟级)
+ Training 继续下一 step，不阻塞

torch.distributed.checkpoint (DCP) + `save_state_dict(no_dist=False)` 支持。或用 `nvidia_dlfw_inspect` / MegaScale 内部工具。

代价：CPU memory 峰值需要 checkpoint size 的空间 (7B: 100 GB, 需要节点有 200+ GB DDR)。

=== Sharded Checkpoint

每 rank 独立存自己的 shard，无需 gather 到 rank 0。torch DCP 的 `FileSystemWriter` 是这样。

优：无 gather bottleneck；rank 之间并行 IO。
缺：restore 时需要 same world size (或用 DCP 的 resharding API)。

*Rank-agnostic checkpoint*：DCP 用 metadata 记录每 tensor 的 sharding，加载时能自动 reshard 到新 world size。允许换 world size (e.g. GPU 挂了减规模)。

=== In-memory checkpoint

MegaScale 提出的 trick：把 checkpoint 存在集群内*另一台机器的 CPU memory*——完全绕过 disk。fail 时从 peer 拉回。

优：秒级恢复。
缺：peer 也挂就丢了。所以 in-memory + periodic disk backup 组合。

=== 一个 Checkpoint 保存代码骨架

```python
import torch.distributed.checkpoint as DCP
from torch.distributed.checkpoint import FileSystemWriter, FileSystemReader

def save_ckpt(model, optim, step, path):
    state = {
        "model": model.state_dict(),      # FSDP sharded state
        "optim": optim.state_dict(),
        "step":  step,
        "rng":   torch.get_rng_state(),
    }
    DCP.save(state_dict=state,
             storage_writer=FileSystemWriter(path),
             planner=DCP.DefaultSavePlanner(),
             process_group=None)  # world

def load_ckpt(model, optim, path):
    state = {"model": model.state_dict(), "optim": optim.state_dict()}
    DCP.load(state_dict=state,
             storage_reader=FileSystemReader(path),
             planner=DCP.DefaultLoadPlanner())
    model.load_state_dict(state["model"])
    optim.load_state_dict(state["optim"])
    return state["step"]
```

生产要加 async wrapper + retention policy (只保最近 3 ckpt) + validation checksum。

== 容错：从"节点挂了"到"训练继续"

*步骤*：

+ *检测*：某 rank 报错 (NCCL timeout / CUDA OOM / segfault)
+ *broadcast fault*：其他 rank 需要知道有人挂了
+ *隔离*：调度层把坏节点标记 down
+ *替换*：若集群有 spare node，调 spare 上 join；无 spare 则减小 world size
+ *reshard*：checkpoint 从旧 world size resharding 到新
+ *resume*：从最近 checkpoint 恢复 training

torch.distributed 有 `elastic` 支持部分：`torchrun` 的 `--rdzv-backend=etcd`，故障后 restart。但生产用 Megatron / vLLM 用 K8s Operator (Volcano, Kubeflow) 或自研 orchestrator (MegaScale, Alibaba PAI, ByteDance MegaScale-Ops)。

=== NCCL Watchdog

NCCL 提供 `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` + `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600`：某个 rank 长时间不响应就 abort。避免"僵尸训练"（NCCL kernel 卡死 但 Python 继续等）。

生产必开：

```bash
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
```

=== 慢卡 (Straggler) 诊断

*现象*：某几个 rank 通信总是慢，拖整个 DP AllReduce。原因：

+ HBM ECC error 频发（GPU 硬件老化）
+ NIC firmware 有 bug
+ 温度过高（thermal throttling），SM 降频
+ Host OS 挂了 daemon 抢 CPU
+ IB link degraded（rate 从 400G 掉到 100G）

*诊断工具*：

+ *nsys per-rank profile*：看哪 rank 的 GEMM kernel 慢
+ *NCCL timing*：`NCCL_DEBUG=INFO` 输出每 op 时间，找慢的 rank
+ *dcgm* (Data Center GPU Manager)：监控 GPU 温度、power、SM 频率
+ *`ibstat` / `ibportstate`*：IB link status
+ *heartbeat 时间戳*：每 rank 每 step 记录到 syslog，事后分析

*MegaScale 自研工具*：
+ *"1-step time" histogram*：每 iter 时间分布，右尾 outlier 就是慢卡
+ *"comm bottleneck detector"*：把 NCCL kernel 时间归因到具体 rank

发现慢卡后，通常做法：*调度器把该节点 drain 出 pool 修复*，替换为 spare node。

== 训练发散 / loss spike 的检测和处理

发散 = loss 突然爆到 nan 或几倍。原因：

+ *梯度爆炸*：某 batch 有 outlier data → grad blow up
+ *FP16/FP8 数值不稳*：某个 tensor scale 猜错
+ *硬件故障*：ECC error 篡改 tensor
+ *Overfitting / degenerate solution*：训久了 loss 突然被特定 pattern 主导

*检测*：

```python
# 每 step 检查
if not torch.isfinite(loss):
    print(f"[rank {rank}] loss={loss.item()} NAN at step {step}")
    save_debug_ckpt()
    # dist.barrier + notify orchestrator
    raise RuntimeError("loss NaN")

grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
if grad_norm > 100.0:
    print(f"[rank {rank}] grad_norm={grad_norm:.2f} SPIKE at step {step}")
    # skip step? or rollback to earlier ckpt?
```

*应对策略*（PaLM 2, Llama, MegaScale 等文献里描述）：

+ *Skip bad batch*：grad_norm > threshold 时 skip optimizer.step()，继续
+ *Rollback to earlier ckpt*：连续 N 步 spike，回退 100 步之前的 ckpt
+ *Automatic hyperparameter adjustment*：spike 后暂时降 LR，慢慢升回
+ *Data audit*：把导致 spike 的 batch dump，人工审 (常见：encoding 错的乱码数据)

Llama-3 报告：训练 405B 有 466 次 spike，用 (a) 90% 直接 skip 修复；剩下 rollback。

== 监控指标 (Metrics)

生产必看的 metrics：

#figure(
  table(
    columns: (auto, 1fr, auto),
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (left, left, left),
    [*类别*], [*指标*], [*正常范围*],
    [Training],  [step time (rank max/mean/p99)], [波动 < 10%],
    [],          [loss curve],                    [平滑下降],
    [],          [grad_norm],                     [< 1.0 后期],
    [],          [LR],                            [符合 schedule],
    [Hardware],  [GPU util (SM)],                 [> 80%],
    [],          [HBM util],                      [> 60%],
    [],          [GPU temp],                      [< 80°C],
    [],          [GPU power],                     [< 700W (H100)],
    [Network],   [IB link rate],                  [400G nominal],
    [],          [NCCL AllReduce BW],             [每步稳定],
    [Data],      [dataloader wait time],          [< 2% step],
    [Ckpt],      [ckpt save duration],            [< 2% step (async)],
    [Errors],    [nan/inf count / hour],          [0],
  ),
  kind: table,
  caption: [生产训练监控指标。所有 rank 的 metric 都要收集，avg + p99 + max 都看。异常 rank 用于慢卡诊断。],
)

Prometheus + Grafana 是标配。dcgm-exporter 提供 GPU metrics。自定义 python metric 用 statsd 或 pushgateway。

== 一个 healthy training run 的 "signature"

看到这些说明训练在轨道上：

+ loss 平滑下降，跟 scaling law 预测吻合
+ grad_norm 前 1000 步降到 < 1.0，之后稳定
+ step time p99 / p50 < 1.15 (无严重 straggler)
+ 无 nan / inf spike
+ GPU util 均在 80%+ (per rank)
+ IB throughput 稳定
+ Checkpoint save 无异常
+ 每 N step 的 val loss / eval 有稳定改善

生产 dashboard 直接把这些 chart 摆一起，异常一眼看出。

== MegaScale 的关键工程 tricks

从 Jiang et al. 2024 挑几个：

+ *Datacenter-level determinism*：所有 rank RNG state 都记录，同 seed 可 bit-exact 复现，便于 debug
+ *Fine-grained profiling with < 3% overhead*：selective sampling (不是 every step trace)
+ *"Fault injection" testing*：训练前主动 kill 几个 rank 测容错逻辑
+ *Elastic scaling*：故障时自动 shrink，故障修完自动 expand
+ *Cost-aware scheduling*：优先 spare node 是 cheapest 的（老 GPU 便宜但 rare）
+ *Log aggregation across 12K rank*：中央 log store，可 query "rank X 的 step Y 到 Z 之间发生了什么"

xAI Colossus (100K GPU) 用类似原理，规模更大。

== 面试考点

#interview[
  *Q1*: 10000 GPU 训 3 个月，你怎么设计容错？

  A: 三层：(1) 硬件监控：dcgm + IB 状态 + 温度 → 提前预警慢卡；(2) 容错：NCCL timeout + async error handling，某 rank 挂了 broadcast 全 rank；(3) 恢复：async checkpoint (15 min 一次) + rank-agnostic 格式 + spare node auto-replace + resharding。加上 loss/grad_norm monitor 处理数值发散。目标：MTBF 2h → downtime \< 5%。
]

#interview[
  *Q2*: async checkpoint 怎么实现？为什么比 sync 快？

  A: sync checkpoint: training step 完成后 gather 所有 shard 到 rank 0，rank 0 write 到 NAS。1 TB 数据几分钟，training 全停。async: (a) D2H copy GPU tensor 到 CPU pinned memory (~10s)；(b) background thread 慢慢写 disk；(c) training 立刻继续。写 disk 不阻塞 training，一次 ckpt 从"几分钟停 training"变成"< 10s 停 training + 后台写"。
]

#interview[
  *Q3*: 一个 rank 突然通信慢，你怎么定位是硬件问题还是软件问题？

  A: (1) 先用 nccl-tests 单独测那 rank 的 raw 带宽——低于同型号其他 rank 是硬件；(2) `nvidia-smi -q` 看 SM clock 是否降频（thermal）、ECC error count；(3) `ibstat` 看 IB link 是否 degraded；(4) `dcgmi diag` 跑硬件自检；(5) 若都正常，重启进程排除 CUDA context / NCCL state corruption。硬件问题必换机；软件问题重启多半解决。
]

#interview[
  *Q4*: loss 突然 nan，你怎么处理？

  A: 立刻 (a) save debug ckpt (包含最近 batch, activation stats); (b) 检查 grad_norm 是不是 spike; (c) skip 这 batch 继续训 (试试)——多数情况能自恢复; (d) 若连续 nan，rollback 到 100 步之前 ckpt，同时降 LR 20%; (e) audit batch 找 outlier data。Llama-3 训练 466 次 spike 90% 用 (c) 处理。
]

#interview[
  *Q5*: 为什么"rank-agnostic checkpoint"重要？

  A: 允许换 world size。原本 512 卡训练，挂 8 卡剩 504——如果 checkpoint 是 fixed shard 就无法 load。rank-agnostic ckpt (torch DCP) 存 metadata 记录每 tensor 的 sharding info，load 时能自动 reshard 到新 world size。允许 elastic training（缩到 504 卡继续，修好后扩回 512）。
]

#interview[
  *Q6*: MegaScale 里 "in-memory checkpoint" 是什么？为什么快？

  A: 把 ckpt 存在集群里*另一台机器*的 CPU DDR，而不是 NAS/S3。NVLink/NIC 直连 CPU memory，几十 GB/s；vs NAS 通常 1-5 GB/s。恢复时从 peer memory 拉，秒级 vs 分钟级。缺点：peer 也挂就丢，所以配 periodic disk backup（每 6 h 一次）。MegaScale 报告 checkpoint overhead 从 5% 降到 0.5%。
]

#interview[
  *Q7*: 训练 metric 里哪几个最重要？

  A: 我按重要性排：(1) loss curve — 训崩了就没意义；(2) grad_norm — 发散预警；(3) step time p99 / mean — 慢卡；(4) nan/inf count — 数值稳定；(5) GPU util & HBM util — 效率；(6) IB throughput — 通信 bottleneck；(7) dataloader wait — data 供不上；(8) ckpt save time — IO 是否健康。dashboard 里 (1)-(4) 一定要有 alert。
]

#interview[
  *Q8*: 集群里 spare node 保留多少比例合适？

  A: 依赖 MTBF 和替换 SLA。10K 卡集群 MTBF ~2h, 替换 SLA 30 min → 每 2h 有 15 min 用 spare → 12.5% overhead。生产通常 spare 5-10%（10K 卡准备 500-1000 spare），配合 hardware repair 快速回归。太多 spare 浪费，太少 fail 后训练缩 world size 影响效率。xAI Colossus 报告 100K GPU 有 ~5K spare。
]
