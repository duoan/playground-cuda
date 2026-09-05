#import "../template.typ": *

= 分布式 debug、checkpoint 与容错

前面几章讲的是"怎么把训练跑起来"，这一章讲"跑起来之后它挂了怎么办"。这是面试里区分"看过教程"和"真跑过大规模训练"的地方：hang 的成因分类、flight recorder 怎么用、checkpoint 为什么不能用 `torch.save`、timeout 该设多长——这些问题背不出来就骗不过做过训练平台的面试官。核心事实只有一条：*collective 是同步的，所有 rank 必须以相同顺序、相同 shape、相同次数调用相同的 collective*。这条被破坏时 PyTorch 不会报错，它会等。

== hang 的排查：最高频的分布式面试题

hang 的本质：某个 rank 进了 collective 在等其他人，而其他人永远不会来。NCCL 的 kernel 已经在 GPU 上排好队等数据，Python 侧看起来完全正常——没有 traceback、没有 CPU 占用异常，`nvidia-smi` 里 GPU 利用率可能还是 100%（因为 NCCL 的 spin-wait 也算利用率）。

=== 成因分类

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left),
  [*成因*], [*典型来源*], [*现象特征*],
  [collective 调用不匹配],
  [`if rank == 0:` 里做了 collective / log / 打印 DTensor],
  [每次都挂在同一个位置，第一个 step 就挂],

  [各 rank step 数不同],
  [数据量不同（最后一个 batch 被 drop 了一部分 rank）、`IterableDataset` 各 shard 长度不等],
  [跑很多 step 之后才挂，挂在 epoch 末尾],

  [走了不同分支],
  [按 loss / 数据内容决定是否 `continue`；OOM 后 `try/except` 跳过这一步；早停],
  [偶发，与数据相关，重跑换种子就变],

  [shape 不一致],
  [变长序列没 pad 齐、动态 batch size],
  [`TORCH_DISTRIBUTED_DEBUG=DETAIL` 直接报 shape mismatch],

  [DDP 的 unused parameter],
  [只有部分 rank 走到某个 module（MoE、条件分支），`find_unused_parameters=False`],
  [挂在 backward，且与数据分布有关],

  [部分 rank 已经死了],
  [某 rank OOM / segfault 退出，其余 rank 继续等它],
  [日志里能找到一个 rank 的 traceback，其他 rank 静默],
)

#warn[
  最经典的写法错误：
  ```python
  if dist.get_rank() == 0:
      total = torch.tensor(len(dataset), device="cuda")
      dist.all_reduce(total)          # 只有 rank 0 参与 → 全体 hang
  ```
  正确写法是所有 rank 都调 collective，只让 rank 0 *使用*结果：
  ```python
  total = torch.tensor(len(dataset), device="cuda")
  dist.all_reduce(total)              # 所有 rank 都调
  if dist.get_rank() == 0:
      print(total.item())
  ```
  同一类错误的变体：`if rank == 0: dcp.save(...)`（DCP 内部有 collective）、`print(dtensor)`（触发 AllGather）、只在 rank 0 上做 `model.eval()` 后跑一遍带 BN 的验证（BN 的 buffer 同步是 collective）。
]

=== 排查工具

按"先让它报错、再定位到哪一行"的顺序：

+ *让它别死等*。`init_process_group(timeout=timedelta(minutes=30))` 加上 `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`（torch 2.x 默认已开），超时后 watchdog 会 abort 进程并打出栈，而不是挂到你手动 kill。没有这一步，后面所有工具都用不上。
+ *`TORCH_DISTRIBUTED_DEBUG=DETAIL`*。给每个 collective 包一层 wrapper，校验各 rank 的 op 类型与 shape 是否一致，不一致直接报错并指出是哪个 collective。代价是每个 collective 多一次同步，*只在排查时开*。
+ *flight recorder*（`TORCH_NCCL_TRACE_BUFFER_SIZE=2000` + `TORCH_NCCL_DUMP_ON_TIMEOUT=1`）。这是排查 hang 最有效的工具：每个 rank 维护一个环形 buffer 记录最近 $N$ 个 collective 的类型、shape、seq id、开始/完成状态，超时时各 rank 把 buffer dump 到 `TORCH_NCCL_DEBUG_INFO_TEMP_FILE` 指定的路径。用 `torch.distributed.flight_recorder.fr_trace` 合并所有 rank 的 dump，它会直接告出"seq 12345 这个 AllReduce 有 63 个 rank 参与、rank 17 没到"——*谁少调了一个 collective 一眼可见*。
+ *`NCCL_DEBUG=INFO`*（配 `NCCL_DEBUG_SUBSYS=INIT,COLL`）。看的是 NCCL 自己的视角：ring/tree 拓扑怎么建的、走了哪块网卡、用了什么算法。初始化阶段的问题（选错网卡、P2P 不可用）主要靠它。日志量很大，别在生产长期开。
+ *`py-spy dump --pid <PID>`*。不需要重启进程、不需要改代码，直接打出目标进程当前的 Python 栈。对每个 rank 都 dump 一次然后 diff：绝大多数 hang 都表现为"63 个 rank 停在 `all_reduce` 那一行，1 个 rank 停在 `dataloader.__next__`"。PID 用 `nvidia-smi` 或 `ps` 找。
+ *`TORCH_NCCL_DESYNC_DEBUG=1`*。超时时分析各 rank 的最后一个 collective，直接报告谁不同步。

#warn[
  `TORCH_NCCL_BLOCKING_WAIT=1` 和 watchdog 是互斥的：开了 blocking wait 就*不会创建 watchdog 线程*，超时行为完全不同。两个都设会让排查行为变得难以预测。生产上选 `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`（默认），不要开 blocking wait。
]

=== 排查流程

+ *确认真的是 hang 而不是慢*。看日志时间戳：step time 是不是突然从 2 s 变成无穷，还是逐渐变慢（后者是 straggler 或数据供不上，不是 hang）。
+ *确认所有 rank 都活着*。有没有某个 rank 已经退出（OOM / segfault）。`torchrun` 通常会报 "one of the processes exited"，但如果它自己也卡住了，就直接 `ps` 数进程数。
+ *`py-spy dump` 每个 rank，按栈分组*。落单的那个 rank 就是罪魁。
+ *如果所有 rank 的栈都在同一个 collective*，说明是 shape 或 op 不匹配（大家都在等，但等的不是同一件事），或者有 rank 已经死了。上 flight recorder dump 对比 seq id。
+ *定位到具体代码后问三个问题*：这个 collective 是不是在 rank 条件分支里？各 rank 的 step 数是不是一样？shape 是不是数据相关的？
+ *复现*。把 world size 缩到 2、数据固定，用 `TORCH_DISTRIBUTED_DEBUG=DETAIL` 单机跑。80% 的 collective 不匹配在 2 卡上就能复现。

== 其他典型故障

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left),
  [*现象*], [*成因*], [*处理*],
  [只有某个 rank OOM],
  [rank 间显存本来就不均：PP 的 stage 负载不均、rank 0 额外持有 dataloader 的 prefetch buffer / logger / 完整 `state_dict` 的临时副本],
  [先确认"该不该不均"。PP 就应该给首尾 stage 少放层；rank 0 的额外占用要挪到 CPU],

  [step time 忽快忽慢，某 rank 总是最慢],
  [straggler：单卡降频（温度/功耗）、ECC 错误重试、host 侧被别的进程抢 CPU、IB link 掉速],
  [collective 是同步的，一张慢卡拖垮全局。按 rank 打点 step time 找出 outlier，`nvidia-smi -q` 看 clock 与 ECC，确认后把节点换掉],

  [`init_process_group` 就卡住或超时],
  [多网卡机器 NCCL 选错网卡（挑了个不通的 docker0 / 管理网口）],
  [`NCCL_SOCKET_IFNAME=eth0` 显式指定；`NCCL_DEBUG=INFO` 看它实际选了哪块],

  [单机正常、跨机就挂],
  [`MASTER_ADDR` 不可达、端口被防火墙拦、IB 没起来],
  [先用 `nc -zv $MASTER_ADDR 29500` 和 `ibstat` 验证链路，再跑 `nccl-tests` 的 `all_reduce_perf` 排除框架],

  [`Address already in use`],
  [上一次训练的进程没清干净，还占着 `MASTER_PORT`],
  [`ps aux | grep torchrun` 清残留；或换端口 / 用 `--rdzv-backend=c10d` 让它自己选],
)

#note[
  遇到疑似通信问题，先跑 `nccl-tests`（`all_reduce_perf -b 8 -e 1G -f 2 -g 8`）。它能在 5 分钟内把问题分成两半：带宽达不到标称值是集群/驱动问题，找 SRE；带宽正常那就是你的代码问题。跳过这一步会浪费很多天。
]

== 环境变量生产模板

```bash
#!/usr/bin/env bash
# ---- 让故障变成报错，而不是永久 hang（必开）----
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1   # watchdog 检测到超时/异常就 abort 进程
                                            # torch 2.x 默认为 1，显式写防止被环境覆盖
# 注意：不要同时开 TORCH_NCCL_BLOCKING_WAIT，它会禁掉 watchdog

# ---- flight recorder：hang 之后能知道谁少调了 collective（强烈建议常开）----
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000    # 每 rank 记录最近 2000 个 collective
export TORCH_NCCL_DUMP_ON_TIMEOUT=1         # 超时时自动 dump
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/shared/fr/rank   # 必须是所有节点都能写的共享路径

# ---- 网络：多网卡/多 HCA 的机器必须显式指定，否则 NCCL 可能挑错 ----
export NCCL_SOCKET_IFNAME=eth0              # 控制面（bootstrap）走哪块网卡
export NCCL_IB_DISABLE=0                    # 有 IB 就别关
# export NCCL_IB_HCA=mlx5                   # 多 HCA 时限定用哪些

# ---- 只在排查时打开，日志量大 / 有额外同步开销 ----
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,COLL
# export TORCH_DISTRIBUTED_DEBUG=DETAIL     # 校验各 rank 的 collective shape
# export TORCH_NCCL_DESYNC_DEBUG=1          # 超时时报告谁不同步
# export CUDA_LAUNCH_BLOCKING=1             # 只用于定位 CUDA 报错的真实位置，极慢

torchrun --nnodes 2 --nproc-per-node 8 \
         --rdzv-backend c10d --rdzv-endpoint "$MASTER_ADDR:29500" \
         --max-restarts 3 \
         train.py
```

`NCCL_SOCKET_IFNAME` 是最常见的坑：它管的是*建连阶段*用哪块网卡，多网卡机器上 NCCL 的自动选择经常挑到 docker 网桥或管理网口，表现是 `init_process_group` 卡住若干分钟然后超时。数据面走 IB 是另一套变量（`NCCL_IB_*`）。

== 数值不一致的排查

先分清哪些不一致是正常的：

- *各 rank 的 loss 不同：正常*。每个 rank 吃不同的数据。要看全局 loss 就自己 AllReduce 求平均后再打印，不要拿 rank 0 的 loss 当全局指标（噪声大，且会掩盖某个 rank 的数据问题）。
- *各 rank 的梯度在 AllReduce 之前不同：正常*。之后应该相同。
- *各 rank 的参数不同：不正常*。DDP 语义要求所有 rank 的参数逐位相同（FSDP/TP 下则是"同一个 replica 组内的对应分片相同"）。

参数 drift 是最难查的一类 bug，因为它不报错，只表现为"loss 降得比预期慢"或者"eval 结果诡异"。定期校验的成本很低：

```python
@torch.no_grad()
def check_replica_consistency(model, group=None):
    """校验 group 内所有 rank 的参数逐位相同。DDP 传 None（整个 world）；
    TP 场景传 TP 组，只校验 replicate 的参数（LayerNorm、bias）。"""
    flat = torch.cat([p.detach().reshape(-1).float() for p in model.parameters()])
    sig = torch.stack([flat.sum(), (flat * flat).sum(), flat.abs().max()])
    ref = sig.clone()
    src = 0 if group is None else dist.get_global_rank(group, 0)
    dist.broadcast(ref, src=src, group=group)     # src 要的是全局 rank
    if not torch.equal(sig, ref):
        raise RuntimeError(f"[rank {dist.get_rank()}] param drift: "
                           f"{sig.tolist()} != {ref.tolist()}")
```

每 500 step 调一次，开销可以忽略。三种最常见的 drift 成因：

+ *自定义 `autograd.Function` 的 forward 做了通信，backward 没做对偶通信*（或反过来）。第 20 章的 `f`/`g` 一旦写错一个方向，梯度就少加了一部分，各 rank 的更新量不同。
+ *buffer 没同步*。`nn.Parameter` 由 DDP 的梯度 AllReduce 保证一致，但 `register_buffer` 的东西（BN 的 `running_mean`、KV cache、自己维护的计数器）不在梯度路径上。DDP 有 `broadcast_buffers=True` 每次 forward 广播一次，FSDP / 手写并行要自己管。
+ *replicate 参数的梯度漏了组内 AllReduce*。TP 组内 LayerNorm 的 weight 是复制的，梯度必须在 TP 组内额外 AllReduce（第 20 章）。漏掉就是每张卡各自朝不同方向更新同一个参数。

另外两个必须一致的东西：随机数（dropout mask、数据增强的 seed 策略）和 optimizer 的超参（LR schedule 必须由 step 数唯一决定，不要依赖 wall clock）。

== Checkpoint

=== 朴素做法为什么不行

`rank0 gather 全量 state_dict 再 torch.save` 有三个问题，规模越大越致命：

+ *显存/内存峰值*：rank 0 要装下完整的模型 + 优化器状态。70B 模型混合精度下约 1.1 TB，rank 0 装不下。
+ *慢且不并行*：所有分片先汇到一张卡，再由一个进程串行写盘。别的 rank 全程干等。
+ *与并行度绑死*：存下来的是裸张量，没有"这一片是全局张量的哪一段"的元信息，换并行度就读不回来。

=== DCP：每 rank 存自己的分片 + 元数据

`torch.distributed.checkpoint`（DCP）的模型是：每个 rank 只写自己持有的分片，另外写一份共享的 metadata 记录每个张量的全局 shape 与各分片的覆盖范围。加载时按 metadata 做区间匹配，*把需要的字节读到当前并行度要求的位置上*——所以改变并行度也能恢复。

```python
# torchrun --nproc-per-node 2 ckpt_demo.py
import torch, torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful

class AppState(Stateful):
    """DCP 会调用 state_dict()/load_state_dict()，把 model 与 optim 一起处理。
    用 get_state_dict / set_state_dict 是为了拿到 DTensor 形态的 sharded
    state_dict，并处理 FSDP/TP 的参数名与优化器状态映射。"""
    def __init__(self, model, optim, step=0):
        self.model, self.optim, self.step = model, optim, step

    def state_dict(self):
        msd, osd = get_state_dict(self.model, self.optim)
        return {"model": msd, "optim": osd, "step": self.step}

    def load_state_dict(self, sd):
        set_state_dict(self.model, self.optim,
                       model_state_dict=sd["model"], optim_state_dict=sd["optim"])
        self.step = sd["step"]

# ---- save：所有 rank 都要调，内部有 collective ----
state = {"app": AppState(model, optim, step)}
dcp.save(state, checkpoint_id=f"/shared/ckpt/step-{step}")

# ---- load：先按当前并行度建好 model/optim，再原地填回 ----
state = {"app": AppState(model, optim)}
dcp.load(state, checkpoint_id="/shared/ckpt/step-1000")
start_step = state["app"].step
```

#insight[
  DCP 的核心卖点是 *resharding*：`tp=8` 存的 checkpoint 可以用 `tp=4, dp=2` 加载，512 卡训练挂了 8 张卡之后能用 504 卡（或任意新配置）继续。前提是张量的分布信息被记了下来——也就是第 21 章的 DTensor。这两章是一件事的两半。
]

#note[
  DCP 的 API 在 2.x 期间演进过（`save_state_dict` 已废弃、`checkpoint_id` 与 `storage_writer` 两种指定方式、`async_save` 的返回类型），确切签名以对应版本文档为准。稳定的是数据模型：分片 + metadata + 按区间匹配加载。
]

=== 必须存什么

漏一样就是"恢复之后曲线不接续"。按重要性：

#table(
  columns: (auto, 1fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left),
  [*内容*], [*漏了会怎样*],
  [model 参数], [显然],
  [optimizer 状态（Adam 的两个 moment、fp32 master weight）], [loss 会明显跳一下再慢慢恢复；这部分比参数还大],
  [LR scheduler 状态（或 `step` 本身）], [学习率跳回起点，最容易被忽略],
  [`GradScaler` 状态（用 fp16 时）], [前若干 step 白跑在 scale 探测上],
  [dataloader 进度（shard 位置 / 已消费样本数）], [重复训练同一批数据，大规模下会造成可观的过拟合],
  [RNG 状态（CPU + CUDA，每 rank 一份）], [dropout / 数据增强序列变了，不可复现],
  [`step` 与 epoch], [日志、eval 节奏、以上全部的对齐基准],
)

=== 异步 checkpoint

70B 模型一次 checkpoint 是 TB 量级，写共享存储要好几分钟。同步存的话这几分钟所有 GPU 全闲。

#figure(
  align(center, timeline(streams: (
    ("sync train", (("compute", 10), ("wait", 8), ("compute", 10))),
    ("sync IO",    (("wait", 10), ("comm", 8), ("wait", 10))),
    ("async train", (("compute", 10), ("wait", 1), ("compute", 17))),
    ("async IO",   (("wait", 11), ("comm", 8), ("wait", 9))),
  ), unit: 0.4, title: "sync vs async checkpoint")),
  caption: [同步 checkpoint：训练停在原地等落盘完成。异步：先把张量 D2H 拷到 CPU pinned memory（图中那一小格，只有这段挡住训练），再由后台线程慢慢写盘，训练立刻继续。],
) <fig-async-ckpt>

```python
fut = dcp.async_save(state, checkpoint_id=path)   # D2H 之后立刻返回
...                                               # 训练继续
fut.result()                                      # 下次 save 之前确认落盘完成
```

代价是 CPU 内存要放得下一份 checkpoint 的暂存副本，以及*不能在上一次没写完时就发起下一次*（所以要留住 future）。

#warn[
  *checkpoint 与 `init_process_group(timeout=...)` 的关系是高频题。* 存 checkpoint 期间 DCP 内部有 collective，而不同 rank 写盘速度可能差很多（共享存储争抢）。写得慢的 rank 让快的 rank 停在 collective 里等——一旦超过 timeout，watchdog 就把这次等待判成 hang 并 abort 整个 job。NCCL 后端的默认 timeout 是 10 分钟，对大模型 checkpoint 明显不够。生产上把它拉到 30–60 分钟：

  ```python
  from datetime import timedelta
  dist.init_process_group("nccl", timeout=timedelta(minutes=45))
  ```

  这也是异步 checkpoint 的另一个好处：GPU 侧的等待窗口从"几分钟写盘"缩短到"几秒 D2H"，撞 timeout 的概率大幅下降。
]

== 容错与弹性

几百卡以上，硬件故障是日常而不是例外：单卡 MTBF 按年算，几千张卡一乘就是每天若干次。所以系统必须假设"随时有卡会挂"。

`torchrun` 提供的部分：

- `--max-restarts N`：某个 worker 挂了，agent 把整组 worker 全部重启（不是只重启挂的那个——collective 的通信组已经废了，只能整组重建），最多 $N$ 次。
- `--rdzv-backend c10d --rdzv-endpoint host:port`：rendezvous，重启后各节点重新会合并重新分配 rank。
- `--nnodes 4:8` 这种弹性区间：允许在 4 到 8 个节点之间伸缩。

弹性能生效有两个硬前提，缺一个就白搭：

+ *进程重启后能从 checkpoint 恢复*。`torchrun` 只负责把进程拉起来，恢复训练状态是你的代码的事。所以启动逻辑必须是"扫描 checkpoint 目录 → 找最新的 → `dcp.load` → 从那个 step 继续"，而不是"从头开始"。
+ *checkpoint 能 resharding*。节点数变了并行度就变了，固定分片的 checkpoint 读不回来。这就是 DCP 的价值所在。

工程上还要加三件 `torchrun` 不管的事：

+ *启动前健康检查*：跑一个小的 AllReduce + 每卡一个小 GEMM，验证所有卡可用、带宽正常。有问题的节点在开训前就踢掉，比训到一半挂掉便宜得多。
+ *按 rank 上报指标*：step time、loss、grad norm、显存峰值都要带 rank 打点。straggler 和"某个 rank 数据坏了"只能靠 per-rank 指标发现。
+ *自动重启的止损*：反复重启同一个坏节点是最常见的运维事故。重启要记录原因，同一节点连续失败就把它拉出资源池，而不是无限 `--max-restarts`。

#insight[
  面试被问"怎么保证几千卡训几个月不断"，答三层：*检测*（timeout + watchdog + flight recorder + per-rank 指标）、*恢复*（异步 DCP checkpoint，15–30 分钟一次，能 resharding）、*调度*（健康检查 + 坏节点隔离 + spare 节点替换 + 从最新 checkpoint 自动续跑）。checkpoint 频率的取舍就是一句话：期望损失 $approx$ 故障率 $times$ 半个 checkpoint 间隔。
]

== 面试考点

#interview[
  *Q1*：训练 hang 了，你的排查顺序是什么？

  A：先分清 hang 和慢（看 step time 是突然变无穷还是逐渐变慢）。然后确认所有进程都活着——有没有 rank 已经 OOM 退出而其余在等它。接着 `py-spy dump --pid` 每个 rank 打栈并按栈分组，落单的那个就是罪魁。如果所有栈都停在同一个 collective，说明是 shape/op 不匹配或有 rank 已死，用 flight recorder 的 dump 对比各 rank 的 seq id。定位到代码后问三件事：collective 是不是写在 rank 条件分支里、各 rank 的 step 数是否相同、shape 是否数据相关。最后缩到 2 卡加 `TORCH_DISTRIBUTED_DEBUG=DETAIL` 复现。
]

#interview[
  *Q2*：hang 最常见的根因是什么？举个具体例子。

  A：collective 调用不匹配。最经典的是把 collective 写进了 `if rank == 0:`——比如只在 rank 0 上 `all_reduce` 一个统计量、只在 rank 0 上 `dcp.save`、或者 `print(dtensor)` 触发了 AllGather。正确写法是所有 rank 都调 collective，只让 rank 0 使用结果。第二常见的是各 rank step 数不同（数据量不等、`IterableDataset` 的 shard 长度不齐），特征是跑很久之后挂在 epoch 末尾。
]

#interview[
  *Q3*：flight recorder 是什么？为什么它比看日志有效？

  A：`TORCH_NCCL_TRACE_BUFFER_SIZE=N` 开启后，每个 rank 用一个环形 buffer 记录最近 $N$ 个 collective 的类型、shape、seq id 和完成状态；配 `TORCH_NCCL_DUMP_ON_TIMEOUT=1`，超时时各 rank 自动把 buffer dump 到共享路径。用 `fr_trace` 合并所有 rank 的 dump，它按 seq id 对齐，直接告诉你"这个 AllReduce 有 63 个 rank 到了、rank 17 没到"。日志做不到这件事，因为 hang 的时候没有人报错，你只有一堆看起来正常的日志。
]

#interview[
  *Q4*：为什么某个 rank OOM 而其他 rank 好好的？

  A：rank 间显存本来就可能不均。三类来源：PP 的 stage 负载不均（lm_head 那段最重）；rank 0 额外持有东西（dataloader prefetch、logger、gather 全量 `state_dict` 的临时副本）；数据长度不均（变长序列没按 token 数均衡分桶）。排查先问"该不该不均"——PP 就应该给首尾 stage 少放层。另外 `torch.cuda.max_memory_allocated()` 要按 rank 打点，不然根本看不到不均。
]

#interview[
  *Q5*：什么是 straggler？一张慢卡为什么能拖垮整个集群？

  A：collective 是同步的，AllReduce 要等最后一个 rank 到达，所以全局 step time 等于最慢那个 rank 的 step time——一张卡慢 20%，整个 job 慢 20%，加卡完全无用。常见成因：温度或功耗墙导致降频、ECC 错误重试、host 侧 CPU 被抢、IB link 掉速。定位靠 per-rank step time 打点找 outlier，再用 `nvidia-smi -q` 看 clock 和 ECC 计数、`ibstat` 看链路速率。确认是硬件就换节点，这类问题修不好。
]

#interview[
  *Q6*：`NCCL_SOCKET_IFNAME` 是干什么的？什么时候必须设？

  A：指定 NCCL bootstrap（建连阶段）走哪块网卡。多网卡机器上 NCCL 的自动选择经常挑到 docker 网桥或不通的管理网口，表现是 `init_process_group` 卡住然后超时，且日志里没有任何有用信息。所以容器环境、多网卡机器一律显式设置，用 `NCCL_DEBUG=INFO` 确认它实际选了哪块。数据面走 IB 是另一套 `NCCL_IB_*` 变量。
]

#interview[
  *Q7*：`rank0 gather 之后 torch.save` 有什么问题？DCP 怎么解决？

  A：三个问题：rank 0 要装下完整模型加优化器状态（70B 是 TB 量级，装不下）；汇聚加串行写盘很慢且其他 rank 全程干等；存下来的是裸张量，没有分布元信息，换并行度就读不回来。DCP 让每个 rank 只写自己的分片，另外写一份 metadata 记录每个张量的全局 shape 和各分片的覆盖范围；加载时按区间匹配读到当前并行度需要的位置，所以支持 resharding。写盘是各 rank 并行的，也没有显存峰值问题。
]

#interview[
  *Q8*：checkpoint 必须存哪些东西？

  A：model 参数、optimizer 状态（Adam 的两个 moment 和 fp32 master weight，比参数还大）、LR scheduler 状态或等价的 `step`、`GradScaler` 状态（fp16 时）、dataloader 进度、每 rank 的 RNG 状态、以及 step/epoch 本身。最常被漏的是 scheduler 和 dataloader 进度：前者让 LR 跳回起点，后者让你重复训同一批数据。
]

#interview[
  *Q9*：为什么大模型训练一定要异步 checkpoint？它和 `timeout` 有什么关系？

  A：TB 级 checkpoint 写共享存储要几分钟，同步存意味着这几分钟所有 GPU 全闲；15 分钟存一次的话开销可观。异步的做法是先把张量 D2H 拷到 CPU pinned memory（秒级，这段确实挡住训练），再由后台线程写盘，训练立刻继续。和 timeout 的关系：存 checkpoint 期间 DCP 内部有 collective，各 rank 写盘速度差异会让快的 rank 停在 collective 里等，超过 `init_process_group(timeout=...)` 就被 watchdog 判成 hang 并 abort 整个 job。NCCL 后端默认 10 分钟，对大模型不够，生产设 30–60 分钟；异步 checkpoint 把 GPU 侧的等待窗口从几分钟缩到几秒，也顺带降低了撞 timeout 的概率。
]

#interview[
  *Q10*：`torchrun --max-restarts` 能实现弹性训练吗？还缺什么？

  A：它只解决"进程被重新拉起来"，而且是整组 worker 一起重启（通信组已经废了，没法只补一个 rank）。要真的弹性还差两件事：代码启动时必须自动扫描并从最新 checkpoint 恢复，否则重启等于从头训；checkpoint 必须能 resharding，因为节点数变了并行度就变了，固定分片读不回来——这是 DCP 加 DTensor 的价值。工程上还要加启动前健康检查、per-rank 指标上报、以及坏节点隔离，否则会反复重启同一个坏节点。
]
