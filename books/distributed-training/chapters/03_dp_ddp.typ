#import "../template.typ": *

= Data Parallel：从 DP 到 DDP 到 gradient accumulation

Data parallel 是最简单也最基础的并行——每卡持有完整模型，各处理不同 batch 分片，梯度 AllReduce。看起来平淡无奇，但里面 DDP overlap、gradient accumulation 语义、last-batch handling 这些细节是面试高频问点。

== `nn.DataParallel` 为什么被淘汰

`torch.nn.DataParallel`（简称 DP，注意与广义"data parallel"区分）：

+ Master GPU (rank 0) 把 batch scatter 到其他 GPU
+ 每 GPU forward，output gather 回 rank 0
+ Loss 在 rank 0 计算，backward
+ 梯度 gather 回 rank 0，rank 0 更新
+ 权重 broadcast 回其他 GPU

问题：
- *rank 0 是 bottleneck*：显存、算力、通信都不均衡
- *Python GIL*：单进程多线程，被 GIL 卡
- *无法跨机*：只支持单进程 (single node)
- *无 overlap*：所有通信都在 backward 结束后同步做

`DataParallel` 从 PyTorch 1.x 开始就*不推荐*了。所有生产用 *DDP (DistributedDataParallel)*。

== DDP 的三大改进

+ *每 GPU 一进程*：绕开 GIL，跨节点原生支持
+ *梯度 AllReduce 而非 gather*：每卡拿到完整梯度，各自更新，权重天然一致
+ *Bucket + backward hook overlap*：这是核心

=== Bucket 机制

DDP 在 `__init__` 时把所有 parameter 按顺序切成"bucket"（默认 25 MB 一个），每个 bucket 是一次 AllReduce 的单位。

Backward 走的时候按*相反顺序*产生梯度（output layer 最先），DDP 注册 backward hook：*一个 bucket 内所有 param 的 grad 都算完 → 立刻起一次 AllReduce（async），backward 继续往前走*。

#figure(
  align(center, timeline(
    streams: (
      ("naive DDP compute ", (("compute", 10), ("wait", 4))),
      ("naive DDP comm    ", (("wait", 10), ("comm", 4))),
      ("bucket DDP compute", (("compute", 10),)),
      ("bucket DDP comm   ", (("wait", 2), ("comm", 3), ("wait", 1), ("comm", 3), ("wait", 1))),
    ),
    unit: 0.4, bar-h: 0.5,
    title: "DDP bucket + backward hook 让 AR overlap 反向计算",
  )),
  caption: [naive DDP 等 backward 全部完成后一次大 AR；bucket DDP 让每个 bucket 就绪就发起 async AR，与后续 backward 计算 overlap。手写实现：`src/distributed_training/02_ddp_from_scratch.py::MyDDP`，与 `torch.nn.parallel.DDP` 输出对齐。],
) <fig-ddp-overlap>

伪代码：

```python
class DDP:
    def __init__(self, module, bucket_cap_mb=25):
        self.buckets = split_params_into_buckets(module.parameters(),
                                                 bucket_cap_mb)
        for p in module.parameters():
            p.register_hook(self._grad_ready)

    def _grad_ready(self, grad):
        bucket = self.bucket_of(grad)
        bucket.mark_ready()
        if bucket.all_ready():
            # 起 async AllReduce, 不 block backward
            handle = dist.all_reduce(bucket.buffer, async_op=True)
            self.pending.append(handle)

    def finish_backward(self):
        for h in self.pending:
            h.wait()      # 等所有 bucket 的 AR 完成
        for p in self.parameters():
            p.grad = p.grad / self.world_size    # 除以 W (平均而非求和)
```

*关键*：AllReduce 与前面层的 backward compute *时间重叠*。理想情况下最后一个 bucket AllReduce 完成的时刻恰好是 backward 结束。

=== Bucket size 的 tradeoff

- 太小：多次 AllReduce，每次都有启动延迟 $alpha$，总时间 $= n times alpha + V/B$，$alpha$ 主导
- 太大：backward 到最后 bucket 还没起 AR，overlap 少

PyTorch 默认 25 MB 是通用值。大模型 (7B+) 可以调到 200-500 MB 减少 overhead。

调节：

```python
model = DDP(model, bucket_cap_mb=200, gradient_as_bucket_view=True)
```

`gradient_as_bucket_view=True` 让 param.grad 直接 view 到 bucket buffer，省一次 copy——所有 7B+ 训练都建议开。

=== `find_unused_parameters` 的坑

如果模型里有 parameter 在某些 batch 不参与 forward（例如 MoE 的部分 expert、条件分支），DDP 默认假设所有 param 都会有 grad，等不到会 hang。

*解决*：`DDP(model, find_unused_parameters=True)`。

*代价*：每步做一次 dtype+shape 的额外遍历判定，*step time 增加 10-30%*。所以能不开就不开——重写模型让所有 param 都参与更好。

MoE 常见做法：aux loss 里加一个 tiny term 让所有 expert weight 有梯度（即使为 0），避免 `find_unused_parameters`。

== Gradient Accumulation：显存换 batch

现在你想训 GBS=1024，但每卡显存只够 micro batch (MBS) = 4。累积梯度：

```python
model.zero_grad()
for i in range(accum_steps):
    x = next(data_iter)               # MBS = 4
    with model.no_sync() if i < accum_steps - 1 else nullcontext():
        loss = model(x).loss / accum_steps
        loss.backward()
optimizer.step()
```

*关键点*：

+ `loss / accum_steps` — 让梯度是平均而非求和。或等价：`loss.backward()` 累积后 `p.grad /= accum_steps`。
+ `model.no_sync()` — 前 $n-1$ 步*不做 AllReduce*，只在最后一步同步。DDP 支持这个 context manager，省下 $"accum_steps" - 1$ 次 AllReduce。

*公式*：GBS = MBS × accum_steps × DP_world_size。

一个 512 卡 setup：
- MBS = 1, accum = 2, DP = 512 → GBS = 1024（简单）
- MBS = 4, accum = 4, DP = 64 → GBS = 1024（更多 DP，更少 accum，通信更多但 kernel 更满）

=== Gradient Accumulation 的隐藏坑

+ *LayerNorm/BatchNorm 的统计*：如果模型用 BatchNorm，micro-batch stats 只在当前 micro-batch 上算，不等价于 GBS。所以大模型都用 LayerNorm/RMSNorm 而不 BN。
+ *Dropout*：每 micro-batch 独立采样，等价于 GBS 里每个样本各自 dropout — 这是 desired behavior。
+ *学习率 warmup*：warmup 按*步*算而非样本。GBS 加倍不代表 warmup 加倍。经验规则 LR $prop sqrt("GBS")$（Chinchilla 建议）或 linear scaling（Goyal 2017）。
+ *梯度 clip 顺序*：应该在 AllReduce *之后*（每步一次），不是 micro-batch backward 之后。累积期间 grad 是 partial 的，clip 无意义。

== DP 的通信量分析

每步 AllReduce 一次全模型梯度：

$ "vol"_"DP" = 2 P (W-1)/W approx 2 P $

$P$ = 模型参数字节数。LLaMA-2 7B (BF16 grad): $P = 14$ GB, per-step comm = *28 GB*。

带宽 400 GB/s (NVLink) 下要 70 ms；50 GB/s (IB) 下 560 ms。如果 step compute 只要 500 ms，DP 通信在跨节点不 overlap 就把 step 拖成 1 秒——2× 慢。

*所以 DDP overlap 至关重要*：把这 28 GB 的 AR 完全隐藏在 backward compute 里。

*何时 DDP overlap 会失败*？

+ Backward 太快，AR 起不完：小模型 + 大集群
+ 通信带宽极差：老 Ethernet
+ Bucket size 不匹配：太小或太大

诊断：`torch.profiler` 看 AR kernel 是不是 overlap 在 compute 里。

== DDP 与 gradient scaling / AMP 的互动

用 `torch.cuda.amp.GradScaler` (FP16 训练) 时：

```python
scaler = GradScaler()
for x in data:
    with autocast(dtype=torch.float16):
        loss = model(x).loss
    scaler.scale(loss).backward()            # scale 后的 loss 反传
    scaler.unscale_(optimizer)               # 取 scale
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

*要点*：
+ `scaler.scale(loss).backward()` 里的 scale 值广播 → 每 rank 用相同 scale
+ `unscale_` 是本地操作，各 rank 独立除 → 结果一致
+ `scaler.update()` 里的 `found_inf` 需要*跨 rank AllReduce*（DDP 自动做，FSDP 要手动）

BF16 不需要 GradScaler（BF16 exponent 范围与 FP32 相同）。所以 H100+ 时代基本弃用 FP16 + GradScaler，直接 BF16。

== 一个"看似 DP，实际不是 DP"的坑

*Batch Norm* 在 DP 下会因为各卡独立算 stats 而不等价于单卡大 batch。修复：`nn.SyncBatchNorm.convert_sync_batchnorm(model)`——跨 rank 同步 mean/var。但每层 BN 会加两次 AllReduce（一次 mean，一次 var），overhead 显著。大模型不用 BN。

*Sequence-level RNG state*：dropout 随机 seed。如果每 rank 用同一 seed，DP 里同一个 batch 的 dropout mask 相同 → 变相减少数据多样性。修复：seed = rank 相关。torch 默认 DataLoader 已经处理，但自己写 dataset 时要注意。

== DDP 与 batch size：`drop_last` 陷阱

`DataLoader(dataset, batch_size=B, shuffle=True)` 在 `DistributedSampler` 下：

```python
sampler = DistributedSampler(dataset, num_replicas=W, rank=r,
                             shuffle=True, drop_last=False)
loader  = DataLoader(dataset, batch_size=B, sampler=sampler)
```

- `drop_last=False`：最后一个 batch 可能 $< W$ tokens。DDP AllReduce 期待每 rank 都有梯度，*少梯度 rank 会 hang*。修复：`drop_last=True` 或 pad。
- `set_epoch(epoch)` 必须每 epoch 调用，否则 shuffle seed 每 epoch 相同 → data 完全一样。这是新手常见 bug。
- 多个 dataset 混合时用 `WeightedRandomSampler` 或 `ConcatDataset`——但 `DistributedSampler` 只能包一层，需要自己写。

== DDP vs FSDP：什么时候还用 DDP？

DDP 是"每卡完整模型"，FSDP 是"参数切片"。DDP 显存更差，通信更少 (2P vs 3P)。适用：

+ *模型小*：< 1B 参数，一张卡装得下 → DDP 更快
+ *调试期*：DDP 简单，debug 快
+ *推理时的 DP*：如果只是 forward，无需 grad AR，DDP 相当于 replica broadcast

7B+ 训练基本弃 DDP 用 FSDP/ZeRO-3。1-7B 之间看具体情况。< 1B dense 模型 DDP 最快。

== 面试考点

#interview[
  *Q1*: DDP 里 backward 和 AllReduce 怎么 overlap？

  A: DDP 把 parameter 分 bucket (~25 MB)，每 bucket 所有 param 的 grad 完成后立刻起 async AllReduce。因为 backward 是从 output 层往 input 层反向计算，"最靠近 output 的 bucket 最先 ready"，早期 bucket 的 AR 与后期 bucket 的 backward compute 时间重叠。理想情况 overlap 率 > 90%。
]

#interview[
  *Q2*: `no_sync()` 什么时候用？为什么能省时间？

  A: gradient accumulation 时用。累积期间 grad 是"partial"，不需要同步。DDP 的 `no_sync()` context 暂停 AR，只在最后一个 micro-batch 后同步一次。accum_steps=8 时省 7 次 AR，直接省 87.5% comm time。
]

#interview[
  *Q3*: bucket_cap_mb 调大调小的 tradeoff？

  A: 大：单次 AR 效率高（带宽项），但 overlap 空间小（要等更多 grad ready）。小：多次 AR，延迟项累积。7B+ 模型建议 100-500 MB。跨节点 (IB 慢) 时更大更好；单机 NVLink 时 25 MB 默认够。
]

#interview[
  *Q4*: DDP 的 loss 是每卡各自算的，最终优化器用的 grad 是"平均"还是"求和"？

  A: 平均。`dist.all_reduce(..., op=SUM)` 之后 DDP 内部自动 `grad /= world_size`。所以 loss 直接算不用手动除。用 `all_reduce` op=AVG (NCCL 支持) 更快但依赖 NCCL 版本。
]

#interview[
  *Q5*: `find_unused_parameters=True` 为什么慢？

  A: DDP 要跟踪哪些 param 参与了 forward autograd graph，需要遍历所有 param + hook counting，*每 step* 都要跑一遍。CPU overhead 显著（10-30% step time）。而且 finalize_backward 会做额外 AR 判定谁 ready。改进：重构模型让所有 param 参与 forward（哪怕加 0）。
]

#interview[
  *Q6*: 如果 GBS=1024，你会选 (DP=64, accum=4, MBS=4) 还是 (DP=256, accum=1, MBS=4)？

  A: 主要看通信 overhead 与 kernel 效率的平衡。DP=256 一步一次 AR（跨节点很贵）；DP=64+accum=4 只每 4 步 AR 一次（省 4×），但每步 kernel 里 MBS=4 是够大的。跨节点带宽有限时选前者，同 NVLink 域选后者。实测：H100 8卡节点上 DP=8 nested DP=8 (across nodes) with accum=4 通常最快。
]

#interview[
  *Q7*: DDP 用 BF16 训练，为什么不需要 GradScaler？

  A: FP16 exponent 只有 5 bit，动态范围 $tilde.op 6 times 10^-5$ 到 $6 times 10^4$；梯度 underflow 到 0 是常态。GradScaler 把 loss scale $times 65536$ 让梯度进入 FP16 表示范围。BF16 exponent 8 bit，动态范围与 FP32 相同 ($10^-38$ 到 $10^38$)，无 underflow 问题，直接省掉 GradScaler。
]
