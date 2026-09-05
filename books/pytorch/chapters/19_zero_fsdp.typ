#import "../template.typ": *

= ZeRO 与 FSDP：把训练状态切开

DDP 的假设是"模型能装进一张卡"。7B 模型混合精度训练要 112 GB，A100 只有 80 GB——这个假设一开始就不成立。ZeRO（Rajbhandari 2020）和它的 PyTorch 原生实现 FSDP 干的事只有一句话：*DDP 在每张卡上存了 $N$ 份完全相同的参数、梯度和优化器状态，把它们沿 DP 组切成 $N$ 片，每卡只存一片*。这一章要说清三件事：三级切分各省多少显存、各多付多少通信、以及 FSDP 那几个必须调对的旋钮。上一章的 $2(N-1)\/N dot D$ 和 "AllReduce = ReduceScatter + AllGather" 是这里全部推导的工具。

== 动机：DDP 冗余了什么

混合精度 AdamW 训练一个 $Phi$ 参数的模型，每卡的常驻显存（不含 activation）：

#formula[$ M_"DDP" = underbrace(2 Phi, "bf16 参数") + underbrace(2 Phi, "bf16 梯度") + underbrace(4 Phi, "fp32 master") + underbrace(4 Phi, "Adam " m) + underbrace(4 Phi, "Adam " v) = 16 Phi $]

为什么要 fp32 master weight：bf16 只有 8 位 mantissa，`w += lr * update` 里当 update 比 w 小 3 个数量级以上时加法直接被舍掉，参数就再也动不了。所以 optimizer 在一份 fp32 副本上更新，再 cast 成 bf16 给 forward 用。$m$、$v$ 同理保 fp32。

7B 模型：$16 times 7 = 112$ GB。这 112 GB 里，*每张卡的内容完全相同*——8 卡就是同一份数据存了 8 遍。ZeRO 就是把这份冗余吃掉。

#figure(
  align(center, mem-stack(
    configs: (
      ("DDP",    (("params", 14.0), ("grads", 14.0), ("optim", 84.0))),
      ("ZeRO-1", (("params", 14.0), ("grads", 14.0), ("optim", 10.5))),
      ("ZeRO-2", (("params", 14.0), ("grads", 1.75), ("optim", 10.5))),
      ("ZeRO-3", (("params", 1.75), ("grads", 1.75), ("optim", 10.5))),
    ),
    width: 9.5, bar-h: 0.5,
  )),
  caption: [7B 模型、8 卡、bf16 + AdamW 的每卡常驻显存（不含 activation）。
    DDP 需要 112 GB 直接 OOM；ZeRO-3 降到 14 GB，activation 反而成了主导项。
    注意 `optim` 那一段（$12 Phi = 84$ GB）占了 DDP 的 75%，所以*先切它*收益最大。],
) <fig-zero-mem>

== ZeRO 三级：切什么、省多少、多付多少

三级是递进的，每级在前一级基础上再切一样东西：

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, center, center, center, left, left),
  [*Stage*], [*优化器状态*], [*梯度*], [*参数*], [*每卡显存*], [*每 step 通信量*],
  [DDP], [复制], [复制], [复制], [$16 Phi$], [$2 M$],
  [ZeRO-1], [*切*], [复制], [复制], [$4 Phi + 12 Phi \/ N$], [$2 M$],
  [ZeRO-2], [切], [*切*], [复制], [$2 Phi + 14 Phi \/ N$], [$2 M$],
  [ZeRO-3], [切], [切], [*切*], [$16 Phi \/ N$], [$3 M$],
)

$M = 2 Phi$ 是*一份 bf16 参数（或梯度）的字节数*，7B 时 $M = 14$ GB。通信量都是每卡每 step 的收发量，已经把 $(N-1)\/N approx 1$ 近似掉了。

=== 通信量为什么是这些数

*DDP*：backward 末尾对全量梯度做一次 AllReduce。上一章的公式给出 $2(N-1)\/N dot M approx 2M$。

*ZeRO-1 / ZeRO-2*：关键在于*每张卡只需要自己那片优化器状态对应的梯度*，不需要全量梯度。所以把 AllReduce 拆回它的两个组成部分，各用在不同地方：

+ backward 末尾 *ReduceScatter* 梯度：$M$。rank $i$ 拿到第 $i$ 片梯度的全局平均。
+ 本地更新自己那片 fp32 master weight，cast 成 bf16。
+ *AllGather* 更新后的*参数*片：$M$。每卡拿回完整的新参数，给下一步 forward 用。

#formula[$ D_"ZeRO-1/2" = underbrace(M, "RS 梯度") + underbrace(M, "AG 参数") = 2 M = D_"DDP" $]

*完全等于 DDP*。区别只是 AllGather 的对象从"梯度"换成了"更新后的参数"，总量分文不差。

*ZeRO-3*：参数也切了，所以*用的时候必须先拼回来*：

+ forward 逐层 *AllGather* 参数：合计 $M$
+ backward 逐层*再* AllGather 一次参数：合计 $M$
+ backward 末尾 *ReduceScatter* 梯度：$M$

#formula[$ D_"ZeRO-3" = underbrace(M, "fwd AG") + underbrace(M, "bwd AG") + underbrace(M, "RS 梯度") = 3 M = 1.5 times D_"DDP" $]

#insight[
  *ZeRO-1 和 ZeRO-2 的通信量与 DDP 完全相同——显存收益是"免费"的。只有 ZeRO-3 多付 50% 通信（$3M$ vs $2M$），换来的是显存从 $O(1)$ 变成 $O(1\/N)$。*

  这是本章最高频的一道题。追问"多 50% 为什么还用"的答案是：*你本来就装不下*。DDP 跑 70B 根本起不来，多 50% 通信换来能训，这不是权衡，是没得选。反过来，模型明明装得下还上 `FULL_SHARD`，就是白亏 50% 通信。
]

=== 为什么反向还要 AllGather 一次

ZeRO-3 的 forward 是"AllGather 本层参数 → 算 → 立刻 reshard 释放"。释放之后这块显存就被拿去装下一层的参数了。backward 算这一层的 $partial cal(L) \/ partial W$ 和 $partial cal(L) \/ partial x$ 都需要 $W$ 本身，而 $W$ 已经不在了——只能再 AllGather 一次。

那为什么不留着不释放？*留着就等于没切*：所有层的参数同时驻留，显存回到 $2Phi$，ZeRO-3 退化成 ZeRO-2。这恰好就是 `SHARD_GRAD_OP` 策略在做的事——它是一个显式的"多花 $2Phi$ 显存、省一次 AllGather"的选项。

== FSDP 的执行流程

FSDP 把模型切成若干 *unit*（由 `auto_wrap_policy` 决定，一般一个 Transformer block 一个）。每个 unit 的生命周期：

#table(
  columns: (auto, auto, 1.4fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left),
  [*阶段*], [*集合操作*], [*显存变化*],
  [forward 进入 unit], [`all_gather` 参数], [本 unit 参数从 $2Phi_u \/ N$ 涨到 $2Phi_u$],
  [forward 计算], [无], [产出 activation（要留给 backward）],
  [forward 退出 unit], [无（reshard）], [立刻释放 unshard 的参数，回到 $2Phi_u \/ N$],
  [backward 进入 unit], [`all_gather` 参数], [再次 unshard],
  [backward 计算], [无], [算出本 unit 的全量梯度],
  [backward 退出 unit], [`reduce_scatter` 梯度], [只留自己那片梯度，释放参数和全量梯度],
)

所以*任意时刻 GPU 上只有 1–2 个 unit 的完整参数*，显存峰值是 $16 Phi \/ N + O(2 Phi_u)$ 而不是 $16 Phi \/ N + 2 Phi$。这个 $Phi_u$（单个 unit 的参数量）就是 `auto_wrap_policy` 在控制的量。

*prefetch* 是让通信藏进计算的机制：在算第 $l$ 层的时候就把第 $l+1$ 层的 AllGather 发出去。

#figure(
  align(center, timeline(
    streams: (
      ("AllGather", (("comm", 3), ("comm", 3), ("comm", 3), ("wait", 3))),
      ("compute  ", (("wait", 3), ("compute", 3), ("compute", 3), ("compute", 3))),
    ),
    unit: 0.55, bar-h: 0.5,
    title: "FSDP forward：第 l+1 层的 AllGather 与第 l 层计算 overlap",
  )),
  caption: [只有第一层的 AllGather 暴露在关键路径上，之后每层的通信都藏在前一层的计算里。
    藏得住的前提是*单层计算时间 $>$ 单层 AllGather 时间*——这就是"unit 不能太小"的量化标准。],
) <fig-fsdp-prefetch>

#insight[
  FSDP 能跑得快，全靠"参数 AllGather 与计算 overlap"。这也给出了 unit 大小的判据：unit 太小则单次 AllGather 消息小、带宽利用率低、次数多，藏不住；unit 太大则一次 unshard 的显存峰值高，而且第一层的暴露延迟变长。*一个 Transformer block* 是被反复验证的甜点。
]

== FSDP 的关键配置

```python
import functools, torch, torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy, MixedPrecision, BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,          # = ZeRO-3
    auto_wrap_policy=functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={MyDecoderLayer},             # 按 block 包
    ),
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,       # AllGather 出来的参数用 bf16
        reduce_dtype=torch.float32,       # 梯度规约用 fp32 更稳，见下
        buffer_dtype=torch.bfloat16,
    ),
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    limit_all_gathers=True,
    use_orig_params=True,
    device_id=torch.cuda.current_device(),
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)   # 必须在 wrap 之后
```

=== `sharding_strategy`

#table(
  columns: (auto, auto, 1.5fr),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left),
  [*策略*], [*等价于*], [*什么时候用*],
  [`FULL_SHARD`], [ZeRO-3], [模型装不下。切参数 + 梯度 + 优化器状态，通信 $3M$],
  [`SHARD_GRAD_OP`], [ZeRO-2], [参数装得下但优化器状态紧。参数复制、forward 后不 reshard，通信 $2M$],
  [`HYBRID_SHARD`], [机内 ZeRO-3 + 机间 DDP], [多机大规模，见下一节],
  [`NO_SHARD`], [DDP], [调试对比用；正经跑 DDP 就直接用 DDP],
  [`_HYBRID_SHARD_ZERO2`], [机内 ZeRO-2 + 机间 DDP], [前缀有下划线，是实验性 API],
)

=== `auto_wrap_policy`：最重要的旋钮

不设 `auto_wrap_policy` 是新手最容易犯的错。此时*整个模型是一个 unit*：forward 一开始就 AllGather 全部参数、直到 forward 结束才释放——峰值显存回到 $2Phi$，*显存基本没降，还白付了通信*。

三种包法：

- `transformer_auto_wrap_policy(transformer_layer_cls={MyDecoderLayer})`：按 Transformer block 包。首选。
- `ModuleWrapPolicy({MyDecoderLayer})`：同上，更简洁的新写法。
- `size_based_auto_wrap_policy(min_num_params=1e8)`：按参数量凑。适用于不知道模型结构的通用脚本，但切出的边界可能横穿一个 block，导致通信碎片化。

反面例子：*按 `nn.Linear` 包*。7B 模型每层有 7 个 Linear，一个 block 就要 7 次 AllGather，每次只有几十 MB——小消息带宽利用率低、NCCL launch 开销累积，而且藏不住。

=== `mixed_precision`

三个 dtype 分开设，因为它们的数值需求不同：

- `param_dtype=bf16`：AllGather 出来的参数用什么精度算。用 bf16 直接把参数通信量减半。
- `reduce_dtype`：梯度 ReduceScatter 用什么精度。bf16 只有 7 位 mantissa，大 world size 下累加误差会积累。小规模（百卡以内）bf16 一般够；上千卡建议 fp32，通信量翻倍但数值稳。
- `buffer_dtype`：`register_buffer` 注册的东西（如 rotary embedding 的 cache）。不设的话 buffer 留在 fp32，会和 bf16 的 activation 类型不匹配报错。

=== `limit_all_gathers` 与 prefetch 的取舍

prefetch 越激进，overlap 越好，但*同时 unshard 的 unit 越多、显存峰值越高*。这一组参数就是在调这个取舍：

- `limit_all_gathers=True`（默认）：限制同时 in-flight 的 AllGather 数量，靠一个 rate limiter 让 CPU 不要跑得太快导致 caching allocator 一直在为未来的 unshard 预留显存。*OOM 时确认它是开的*。
- `backward_prefetch=BACKWARD_PRE`（默认）：在第 $l$ 层 backward *开始前*就发第 $l-1$ 层的 AllGather。overlap 最好、显存峰值最高。`BACKWARD_POST` 更保守，`None` 最省显存最慢。

=== 与 activation checkpointing 组合

FSDP 省的是参数/梯度/优化器状态，*activation 一点没省*——forward 时参数是完整的，activation 大小和单卡一样。所以 ZeRO-3 之后 activation 常常变成新的瓶颈（看回 @fig-zero-mem 的 14 GB，一个长序列的 activation 轻松超过它）。

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl,
)

# 顺序：先 AC，再 FSDP wrap
apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=functools.partial(
        checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
    check_fn=lambda m: isinstance(m, MyDecoderLayer),
)
model = FSDP(model, ...)
```

两个要点：*先 AC 再 wrap*，以及 `CheckpointImpl.NO_REENTRANT`（reentrant 版本的 backward 重入行为与 FSDP 的 unshard / reshard 时机配合不好）。recompute 时需要的参数由 FSDP 自动重新 AllGather。

== HYBRID_SHARD：把通信压回机内

`FULL_SHARD` 在 1024 卡上的问题：每层 AllGather 要跨越 128 台机器，走的是 IB（单网卡 12.5–50 GB/s），比 NVLink 慢一个数量级（第 17 章）。而且 $N$ 大到一定程度，每卡那一片参数已经小到 AllGather 全是延迟开销。

`HYBRID_SHARD` 的做法：*机内 shard、机间 replicate*。一台 8 卡机内部做完整的 ZeRO-3，机器之间做 DDP。

```python
from torch.distributed.device_mesh import init_device_mesh

n_nodes = dist.get_world_size() // 8
mesh = init_device_mesh("cuda", (n_nodes, 8),
                        mesh_dim_names=("dp_replicate", "dp_shard"))
model = FSDP(model, sharding_strategy=ShardingStrategy.HYBRID_SHARD,
             device_mesh=mesh, auto_wrap_policy=..., use_orig_params=True)
```

代价与收益：每卡显存变成 $16 Phi \/ 8$ 而不是 $16 Phi \/ 1024$（省得少了），但*所有参数 AllGather 都在 NVLink 域内*，跨机只剩梯度的 AllReduce——而且这个 AllReduce 作用在已经 shard 过的梯度上，量只有 $2M\/8$。大规模训练（Llama 3 405B 就是这个路子）几乎都用它。

== FSDP1 vs FSDP2

FSDP1 的核心数据结构是 *FlatParameter*：把一个 unit 里所有参数拼成一个巨大的一维张量再切片。这个设计带来一串麻烦：

- `state_dict` 里看到的是 FlatParameter，不是原始参数名，要靠一层复杂的映射还原（`use_orig_params=True` 缓解了这一点，但仍是打补丁）。
- 和 TP 组合别扭：TP 已经把单个权重切了，FSDP1 再拼平再切，两层 sharding 元数据无法统一表达。
- `torch.compile` 不友好：FlatParameter 的 view / slice 操作会造成 graph break。
- 逐参数的控制（frozen、per-param LR、不同 dtype）很难做。

FSDP2（`fully_shard`，torch 2.4 起可用，2.6+ 进入 `torch.distributed.fsdp` 公开命名空间，逐步成为推荐路径）改成 *per-parameter sharding*：每个 `nn.Parameter` 独立在 dim-0 上切，用 *DTensor* 表达"这是一个逻辑上完整、物理上分片的张量"。

#table(
  columns: (auto, auto, auto),
  stroke: 0.4pt + gray,
  inset: 5pt,
  align: (left, left, left),
  [*方面*], [*FSDP1*], [*FSDP2*],
  [切分单位], [FlatParameter（unit 内拼平）], [每个 parameter 独立切 dim-0],
  [元数据], [自定义 handle], [DTensor + DeviceMesh],
  [`state_dict`], [需要 `StateDictType` 上下文转换], [直接就是 DTensor，key 是原始参数名],
  [与 TP 组合], [别扭], [两层 sharding 都用 DTensor 表达，自然叠加（第 21 章）],
  [`torch.compile`], [容易 graph break], [友好],
  [API 形态], [包一层 `FSDP(model)`], [原地改造 `fully_shard(module)`，不换类型],
)

```python
import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh("cuda", (dist.get_world_size(),))
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16,
                          reduce_dtype=torch.float32)

# 逐 block 调用，最后包 root —— 顺序很重要，root 必须最后
for block in model.layers:
    fully_shard(block, mesh=mesh, mp_policy=mp)
fully_shard(model, mesh=mesh, mp_policy=mp)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
```

三个 API 差异值得记：

- *`fully_shard` 是原地改造*，返回的还是原来那个 module（类型上多混入了 `FSDPModule`）。所以*没有 `.module` 前缀问题*，`state_dict` 的 key 就是原始名字——这是 FSDP2 最直接的体验改善。
- *`auto_wrap_policy` 没了*，改成你自己循环调用 `fully_shard`。更显式，也更难包错。
- *`reshard_after_forward`* 取代 `sharding_strategy`：`True` = ZeRO-3，`False` = ZeRO-2（forward 后不释放）。默认对 root module 是 `False`（反向马上要用，留着省一次 AllGather），其他 module 是 `True`。传一个 2D mesh 就是 HSDP。

梯度累积的等价写法：FSDP1 用 `model.no_sync()`，FSDP2 用 `model.set_requires_gradient_sync(False)`。

#warn[
  FSDP1 的 `no_sync()` 在 ZeRO-3 下有个反直觉的代价：不做 ReduceScatter 就意味着要*保留全量未分片的梯度*，显存从 $2Phi\/N$ 涨回 $2Phi$。所以 FSDP 下的梯度累积通常*不用* `no_sync`——反正 ReduceScatter 本来就和 backward overlap，每个 micro-step 都做的额外成本远小于多占 $2Phi$ 显存。
]

== checkpoint：分片状态怎么存

FSDP 每卡只有一片参数，`state_dict()` 直接存下来就是"这一片"。两种模式：

- `FULL_STATE_DICT`：所有 rank AllGather 出完整参数，汇聚到 rank 0（通常配 `cpu_offload=True`，否则一张卡装不下）。产物和单卡 checkpoint 格式一致，*可以直接给推理用*，但存的瞬间 rank 0 的 CPU 内存要能装下整个模型，而且很慢。
- `SHARDED_STATE_DICT`：每卡各存自己那片，产物是"$N$ 个分片文件 + 一份元数据"。存得快、内存友好，但直接加载要求相同的 world size。

解法是 *DCP*（`torch.distributed.checkpoint`）：它把分片写成带全局坐标元数据的格式，*加载时可以 resharding*——用 8 卡存的 checkpoint 能在 16 卡上恢复。现代做法是统一走 DCP：

```python
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

model_sd, optim_sd = get_state_dict(model, optimizer)
dcp.save({"model": model_sd, "optim": optim_sd}, checkpoint_id="ckpt/step_1000")
```

完整的存/恢复流程、resharding 细节和常见坑见第 22 章。

== ZeRO-Offload / ZeRO-Infinity

思路是把显存压力转移到更慢更大的存储：*ZeRO-Offload* 把优化器状态和 fp32 master weight 放 CPU 内存，optimizer step 在 CPU 上跑（DeepSpeed 有 SIMD 优化的 CPU Adam）；*ZeRO-Infinity* 再往下放到 NVMe。

代价是带宽：A100 的 HBM 有 2 TB/s 量级，PCIe 4.0 x16 只有 32 GB/s 量级，NVMe 更低一个数量级。每步都要把梯度搬下去、把更新后的参数搬上来，*瓶颈从算力变成 PCIe*。所以它的定位很明确：单卡 / 少卡想跑本来跑不了的模型（单张 24 GB 卡 fine-tune 7B），而不是提升吞吐。FSDP 侧的对应物是 `CPUOffload(offload_params=True)`（FSDP1）/ `CPUOffloadPolicy`（FSDP2）。

== FSDP 常见坑

#warn[
  *1. 没设 `auto_wrap_policy`。* 整个模型成为一个 unit，forward 一开始 AllGather 全部参数，显存基本不降。现象是"上了 `FULL_SHARD` 还是 OOM"。FSDP2 里对应的错是只调了 root 的 `fully_shard` 而没有逐 block 调。

  *2. 优化器在 wrap 之前创建。* `AdamW(model.parameters())` 拿到的是*完整尺寸*的参数引用；FSDP wrap 会把参数替换成分片（FSDP1 是 FlatParameter，FSDP2 是 DTensor）。结果是优化器状态按全量分配（显存没省）而且更新写到了游离的旧张量上，*模型根本不动*。一律 wrap 完再建 optimizer。

  *3. 用 `torch.nn.utils.clip_grad_norm_` 裁剪 FSDP1 的梯度。* 全局 norm 是 $sqrt(sum_i g_i^2)$，需要跨 rank 规约；每卡只有梯度的一片，本地算出来的 norm 是错的（偏小），裁剪系数也就错了。FSDP1 要用它自己的方法 `model.clip_grad_norm_(1.0)`。FSDP2 因为梯度是 DTensor，`torch.nn.utils.clip_grad_norm_` 会自动走 DTensor 的规约路径，可以直接用。

  *4. 直接 `print(param)` 或做自定义初始化。* 分片状态下参数是不完整的。FSDP1 用 `with FSDP.summon_full_params(model):` 临时 AllGather 出完整参数（退出时自动 reshard），适合做初始化、打印、或者调用不理解 sharding 的第三方代码。FSDP2 用 `model.unshard()` 或对 DTensor 调 `param.full_tensor()`。

  *5. `buffer_dtype` 没设。* buffer 留在 fp32、activation 是 bf16，某些 op 会报 dtype 不匹配。

  *6. AC 和 FSDP 的顺序搞反。* 先 AC 再 FSDP wrap，且用 `CheckpointImpl.NO_REENTRANT`。

  *7. frozen 参数和可训练参数混在一个 unit 里。* LoRA / partial fine-tune 时，如果一个 unit 内既有 `requires_grad=True` 又有 `False` 的参数，FSDP1 的 FlatParameter 无法表达"只对一部分做 ReduceScatter"。要么把 frozen 部分单独 wrap，要么用 FSDP2（per-parameter sharding 天然支持）。
]

== DDP 还是 FSDP：决策清单

按顺序问自己：

+ *$16 Phi$ 加上 activation 能装进一张卡吗？* 能 → *DDP*。它最简单、通信最少（$2M$）、没有 unshard 开销。1B 以下的 dense 模型基本都该用 DDP。
+ *参数装得下，但优化器状态挤爆了？* → *`SHARD_GRAD_OP`*（ZeRO-2）。通信量还是 $2M$，显存从 $16Phi$ 降到 $2Phi + 14Phi\/N$，是最划算的一档。
+ *参数也装不下？* → *`FULL_SHARD`*（ZeRO-3）。付 50% 额外通信换 $16Phi\/N$。
+ *卡数超过一台机器，跨机 AllGather 拖慢了？* → *`HYBRID_SHARD`*，把参数通信压回 NVLink 域。
+ *activation 成了新瓶颈？* → 叠 activation checkpointing；序列很长则上 CP / SP（第 20 章）。
+ *单层的权重大到一张卡放不下，或者要压低单 token 延迟？* → 这已经超出 data parallel 的范畴，需要 TP / PP（第 20 章）。

#note[
  一个常被忽略的选项：*先试 ZeRO-2*。很多人一上手就 `FULL_SHARD`，其实 $12Phi$ 的优化器状态占了 DDP 显存的 75%，切掉它（外加梯度）往往就够了，而且*不花额外通信*。
]

更深的推导（ZeRO 与 TP 的分工、HSDP 的通信量矩阵、超大规模下的拓扑感知配置）见仓库的《大模型分布式训练面试通关手册》。

== 面试考点

#interview[
  *Q1*：混合精度 AdamW 训练每个参数要多少字节？为什么？

  A：16 字节。bf16 参数 2 + bf16 梯度 2 + fp32 master weight 4 + Adam 的 $m$ 4 + $v$ 4。fp32 master 是必须的：bf16 只有 8 位 mantissa，当 `lr * update` 比参数本身小 3 个数量级以上时，bf16 加法直接把更新舍掉，参数就再也动不了。7B 模型就是 112 GB，A100 的 80 GB 一步都跑不了——这是 ZeRO 存在的理由。
]

#interview[
  *Q2*：ZeRO 三级各切什么、显存和通信各是多少？

  A：Stage 1 切优化器状态（$12Phi$），每卡 $4Phi + 12Phi\/N$；Stage 2 再切梯度，$2Phi + 14Phi\/N$；Stage 3 再切参数，$16Phi\/N$。通信量：Stage 1/2 都是 $2M$，*和 DDP 完全一样*；Stage 3 是 $3M$，多 50%。所以"先切优化器状态"是最划算的一步——它占了 DDP 显存的 75% 且完全不花额外通信。
]

#interview[
  *Q3*：为什么 ZeRO-1/2 的通信量和 DDP 一样？

  A：因为 AllReduce 本身就是 ReduceScatter + AllGather，各 $M$。DDP 是"RS 梯度 + AG 梯度"；ZeRO-1/2 是"RS 梯度（每卡只要自己那片）+ 本地更新 + AG 更新后的参数"。AllGather 的对象从梯度换成了参数，字节数一样，总量都是 $2M$。显存收益是免费的。
]

#interview[
  *Q4*：ZeRO-3 通信量多 50%，为什么还用？

  A：因为你本来就装不下。DDP 跑 70B 需要每卡 1120 GB，根本起不来；ZeRO-3 把它降到 $16Phi\/N$，64 卡就是 17.5 GB。多 50% 通信换来"能训"，这不是权衡而是没得选。反过来说，模型明明装得下还上 `FULL_SHARD` 就是白亏 50% 通信——那种情况该用 DDP 或 `SHARD_GRAD_OP`。
]

#interview[
  *Q5*：FSDP 的 forward 已经 AllGather 过参数了，为什么 backward 还要再 AllGather 一次？

  A：因为 forward 算完就 reshard 释放了，那块显存拿去装下一层了。backward 求 $partial cal(L)\/partial W$ 和 $partial cal(L)\/partial x$ 都需要 $W$，只能重新拼一次。不释放当然可以，但那就等于没切——所有层参数同时驻留，显存回到 $2Phi$，退化成 ZeRO-2。`SHARD_GRAD_OP` 就是这个显式选项：多花 $2Phi$ 显存换省一次 AllGather。
]

#interview[
  *Q6*：`auto_wrap_policy` 为什么必须按 Transformer block 包？包错了会怎样？

  A：FSDP 的 unshard / reshard 是以 unit 为粒度的，unit 大小决定了显存峰值和单次 AllGather 的消息大小。*不设 policy* → 整个模型一个 unit，forward 开头 AllGather 全部参数直到结束才释放，显存基本不降（最常见的"开了 `FULL_SHARD` 还是 OOM"）。*按 `nn.Linear` 包* → 一个 block 7 次几十 MB 的 AllGather，小消息带宽利用率低、launch 开销累积、藏不住。一个 block 一个 unit 让单次 AllGather 足够大又不会把峰值拉太高。
]

#interview[
  *Q7*：FSDP2 相比 FSDP1 改了什么？为什么这个改动重要？

  A：FSDP1 用 FlatParameter——把一个 unit 内所有参数拼成一维大张量再切；FSDP2 用 *per-parameter sharding*，每个 `nn.Parameter` 独立切 dim-0，用 DTensor 表达。重要性在四点：`state_dict` 的 key 就是原始参数名（不再需要 `StateDictType` 转换，也没有 `module.` 前缀）；和 TP 组合时两层 sharding 都用 DTensor 表达，可以自然叠加；`torch.compile` 不再因 FlatParameter 的 view 操作 graph break；逐参数的 frozen / 不同 dtype / per-param LR 都能做（LoRA 场景）。API 上 `fully_shard` 是原地改造而不是包一层。
]

#interview[
  *Q8*：HYBRID_SHARD 解决什么问题？

  A：`FULL_SHARD` 在多机上每层 AllGather 都要跨 IB，比 NVLink 慢一个数量级，而且 $N$ 很大时每片参数小到 AllGather 全是延迟。`HYBRID_SHARD` 改成机内 ZeRO-3、机间 DDP：所有参数 AllGather 都在 NVLink 域内完成，跨机只剩梯度 AllReduce，且作用在已分片的梯度上（量是 $2M\/8$）。代价是显存只降到 $16Phi\/8$ 而不是 $16Phi\/N$。大规模训练（Llama 3 405B）的标准做法。
]

#interview[
  *Q9*：FSDP 下为什么不能直接用 `torch.nn.utils.clip_grad_norm_`？

  A：全局 norm 是 $sqrt(sum_i g_i^2)$，要跨所有 rank 求和。FSDP1 下每卡只有梯度的一片，本地算出的 norm 偏小，裁剪系数就错了——训练不会报错，只是 clip 失效。FSDP1 要用 `model.clip_grad_norm_(1.0)`，它内部做跨 rank 规约。FSDP2 下梯度是 DTensor，`torch.nn.utils.clip_grad_norm_` 会走 DTensor 的规约路径，可以直接用。同理，`summon_full_params`（FSDP1）/ `unshard()`（FSDP2）用于需要看到完整参数的场景，比如自定义初始化或调用不理解 sharding 的第三方代码。
]

#interview[
  *Q10*：DDP 和 FSDP 怎么选？

  A：按"装不装得下"决策。$16Phi$ + activation 装得下 → DDP，最简单、通信最少（$2M$）、没有 unshard 开销。参数装得下但优化器状态紧 → `SHARD_GRAD_OP`（ZeRO-2），通信还是 $2M$ 但显存降到 $2Phi + 14Phi\/N$，性价比最高的一档。参数也装不下 → `FULL_SHARD`，付 50% 通信换 $16Phi\/N$。跨机后 AllGather 拖慢 → `HYBRID_SHARD`。别忘了 FSDP *不省 activation*，切完状态之后 activation 常常是新瓶颈，要叠 activation checkpointing。
]
