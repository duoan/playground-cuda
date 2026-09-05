#import "../template.typ": *

= ZeRO 与 FSDP：把模型切进 DP 组

ZeRO (Zero Redundancy Optimizer, Rajbhandari 2020) 和 FSDP (Fully Sharded Data Parallel, Zhao 2023) 是同一个思想的两个实现：*把 optimizer state / grad / parameter 沿 DP 组分片，让每卡只持有 1/W*。这是 7B+ 训练的入门砖。

== ZeRO 的三个 stage

标准 AdamW 混合精度显存 = 16 bytes/param，分三块：

- Optimizer state (m, v, master weight)：12 bytes = 75% 
- Gradient：2 bytes = 12.5%
- Weight：2 bytes = 12.5%

ZeRO 的三 stage 依次分片这三块：

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, auto, auto, auto, auto),
    [*Stage*], [*Optim state*], [*Grad*], [*Weight*], [*每卡 mem*],
    [DDP (baseline)],  [replicate], [replicate], [replicate], [$16 P$],
    [ZeRO-1],          [*shard*],   [replicate], [replicate], [$4 P + 12 P / W$],
    [ZeRO-2],          [shard],     [*shard*],   [replicate], [$2 P + 14 P / W$],
    [ZeRO-3 (= FSDP)], [shard],     [shard],     [*shard*],   [$16 P / W$],
  ),
  kind: table,
  caption: [ZeRO 三 stage 分片对象。$P$ 是参数量。$W$ = DP world size。ZeRO-3 显存*线性*降到 $16P/W$——1000 卡就能装 6250B 模型的 state。],
)

*心算*：LLaMA-2 70B 在 8 卡 DDP 每卡 $16 times 70/8 = 140$ GB（放不下，会 OOM）；ZeRO-3 每卡 $16 times 70/64 = 17.5$ GB（64 卡）；再加 activation，能塞进 80GB H100。

#figure(
  align(center, mem-stack(
    // 7B on 8 GPUs, 4 configs
    configs: (
      ("DDP",    (("params", 14.0), ("grads", 14.0), ("opt", 84.0))),
      ("ZeRO-1", (("params", 14.0), ("grads", 14.0), ("opt", 10.5))),
      ("ZeRO-2", (("params", 14.0), ("grads", 1.75), ("opt", 10.5))),
      ("ZeRO-3", (("params", 1.75), ("grads", 1.75), ("opt", 10.5))),
    ),
    width: 9.5, bar-h: 0.5,
  )),
  caption: [7B 模型、8 卡、BF16+Adam 的 per-GPU 显存对比（不含 activation）。DDP 占 112 GB → OOM；ZeRO-3 只 14 GB。],
) <fig-zero-mem-7b>

同一个模型换到 64 卡：

#figure(
  align(center, mem-stack(
    configs: (
      ("DDP-64",    (("params", 14.0), ("grads", 14.0), ("opt", 84.0))),
      ("ZeRO-1-64", (("params", 14.0), ("grads", 14.0), ("opt", 1.3))),
      ("ZeRO-2-64", (("params", 14.0), ("grads", 0.22), ("opt", 1.3))),
      ("ZeRO-3-64", (("params", 0.22), ("grads", 0.22), ("opt", 1.3))),
    ),
    width: 9.5, bar-h: 0.5,
  )),
  caption: [64 卡场景。ZeRO-3 每卡只需 1.7 GB 存 param/grad/opt，activation 反而成为主导。],
) <fig-zero-mem-64>

估算器：`src/distributed_training/estimators.py::mem_estimate()`——传入 zero_stage 0/1/2/3 即可打印每项 breakdown。

== ZeRO 的通信增量

ZeRO 显存换通信。总量分析：

*DDP*：backward 完做一次 grad AllReduce = $2P$ 字节。

*ZeRO-1*：
- Backward：grad 仍需 AllReduce ($2P$)——因为每卡要有完整 grad，才知道自己那份 optim state 该 update 什么
- 优化：可以改成 *ReduceScatter*（每卡只需自己 shard 的 grad） = $P$
- Update 后：每卡只更新自己的 optim state → 只有自己 shard 的 weight 新值 → 需要 AllGather 广播 = $P$
- Total = $P + P = 2P$（同 DDP 通信量）

*ZeRO-2*：
- Backward：`ReduceScatter grad` = $P$（每卡拿到自己 shard 的 grad）
- Update：本地做，只更新 shard 的 weight
- Forward 下一步：需要 AllGather weight = $P$
- Total = $2P$ ✓ 同 DDP

*ZeRO-3*：
- Forward：每层前 AllGather weight → 层结束 discard = $P$
- Backward：每层前 AllGather weight → 层结束 discard = $P$
- Backward end：ReduceScatter grad = $P$
- Total = *$3P$* — 比 DDP 多 50%

*总结*：

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, right, right),
    [*策略*], [*Mem / GPU*], [*Comm / step*],
    [DDP],   [$16 P$],       [$2 P$],
    [ZeRO-1],[$4 P + 12 P/W$], [$2 P$],
    [ZeRO-2],[$2 P + 14 P/W$], [$2 P$],
    [ZeRO-3],[$16 P / W$],   [$3 P$],
  ),
  kind: table,
  caption: [ZeRO 显存 vs 通信。三 stage 里唯有 stage-3 增加通信量 —— 但换来的显存收益让训 100B+ 成为可能。],
)

#insight[
  面试常问："ZeRO-3 比 DDP 通信量多 50%，为什么大家还用它？" 答：因为你*本来就装不下*。DDP 在 70B 模型上根本跑不起来，多 50% 通信换来能训——这是没得选。若模型足够小 (< 5B)，DDP 更快是对的，实际上 PyTorch 的 FSDP `NO_SHARD` 模式就是 DDP。
]

== FSDP：PyTorch 原生实现

FSDP 是 ZeRO-3 的 PyTorch 官方实现，语义等价但 API 更 pythonic。基本用法：

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    MixedPrecision, ShardingStrategy, BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial

mp = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)
wrap_policy = partial(transformer_auto_wrap_policy,
                      transformer_layer_cls={LlamaDecoderLayer})

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,   # = ZeRO-3
    auto_wrap_policy=wrap_policy,
    mixed_precision=mp,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    device_id=torch.cuda.current_device(),
    use_orig_params=True,                             # 推荐
)
```

=== `auto_wrap_policy` 是最重要的旋钮

FSDP 是"逐 unit AllGather"。unit 太大：一次 AG 很多参数，显存峰值高。unit 太小：AG 次数多，overhead 高。

*正确做法*：每 Transformer layer 一个 FSDP unit。这样每层 AG ~140 MB (7B / 32 层)，够大不掉带宽。

*错误做法 1*：整个 model 一个 unit → 相当于 DDP，没切
*错误做法 2*：每个 `nn.Linear` 一个 unit → AG 每次 40 MB，一层 forward 6 次 AG，overhead 爆炸

=== ShardingStrategy 四种

#figure(
  table(
    columns: (auto, 1fr, 1.6fr, 2fr),
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (left, left, left, left),
    [*Strategy*], [*ZeRO 等价*], [*Weight / Grad / Optim*], [*用途*],
    [NO_SHARD],       [DDP],           [replicate / replicate / replicate],       [小模型],
    [SHARD_GRAD_OP],  [ZeRO-2],        [replicate / shard / shard],               [模型能装下，只切 grad+optim],
    [FULL_SHARD],     [ZeRO-3],        [shard / shard / shard],                   [大模型标配],
    [HYBRID_SHARD],   [HSDP],          [intra-node shard, inter DDP],             [多节点大模型],
    [`_HYBRID_SHARD_ZERO2`], [HSDP+ZeRO-2], [intra-node shard weight, inter DDP grad], [新选项],
  ),
  kind: table,
  caption: [FSDP sharding strategy。HYBRID_SHARD 是跨节点场景的关键——见下节。],
)

=== HYBRID_SHARD：跨节点场景的最优解

*问题*：FULL_SHARD 在 1024 卡的集群上做每层 AllGather，跨节点带宽 (IB 50 GB/s) 慢一个数量级。

*思路*：*节点内 FULL_SHARD (ZeRO-3)*，*节点间 DDP*。每节点各持完整模型的 1/8（节点内 8 卡共享），节点间只做 grad 的 AllReduce。

```python
FSDP(
    model,
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
    device_mesh=DeviceMesh("cuda", (num_nodes, 8),
                           mesh_dim_names=("dp", "shard")),
)
```

*通信分析*（每 step，每卡）：
- Intra-node AG (NVLink): $2 P / 8 = P/4$ (fwd + bwd)
- Intra-node RS (NVLink): $P / 8$
- Inter-node AR (IB): $2 P / 8 = P/4$（因为 grad 已经 shard，只 AR 自己那份）

对比 pure FULL_SHARD (1024 卡)：所有 AG/RS 都跨节点 = $3P$ IB volume。HSDP 只 $P/4$。*wall-clock ~5× 加速*。

这是 Llama-3 405B 训练用的方案。

=== `use_orig_params=True` 是什么？

FSDP 内部把多个 param flatten 成一个大 buffer（"FlatParameter"）。旧版 API 里 `optimizer.param_groups` 看到的是 FlatParameter 而不是原 name——想按层设 lr / weight decay 会难。

`use_orig_params=True` (推荐)：optimizer 仍然看到原始的 `named_parameters()`，backward 时 FSDP 内部处理 flatten。付出小的 index bookkeeping 开销，换来 API 一致性。

=== `BackwardPrefetch`

Backward 里第 $l$ 层的 AllGather 什么时候起？

- `BACKWARD_PRE`：在第 $l$ 层的 backward *开始前* prefetch → overlap 与第 $l+1$ 层 backward compute（激进，最快，但显存峰值高）
- `BACKWARD_POST`：在第 $l$ 层 backward 结束后起 → overlap 更保守（少 overlap，显存低）
- `None`：不 prefetch，最省内存最慢

大模型训练建议 `BACKWARD_PRE`；OOM 再降到 `POST`。

== FSDP2 (torchtitan / PyTorch 2.4+)

FSDP1 有几个痛点：FlatParameter、composability 差、动态 shape 支持差。FSDP2 (`torch.distributed._composable.fsdp.fully_shard`) 重写了：

+ *Per-parameter sharding*：每 param 独立 shard，不再 flatten
+ *DTensor 后端*：sharding meta 用 DTensor 表达，与 DeviceMesh 完全集成
+ *更好的 optimizer 交互*：optim state 自动 shard
+ *TP × FSDP composability*：FSDP2 与 TP 组合更自然

```python
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._tensor import DeviceMesh

mesh = DeviceMesh("cuda", (num_nodes, 8), mesh_dim_names=("dp_replicate", "dp_shard"))
for layer in model.layers:
    fully_shard(layer, mesh=mesh["dp_shard"])
fully_shard(model, mesh=mesh["dp_shard"])   # root last
```

torchtitan 的 Llama 3 训练用 FSDP2 + TP + PP。FSDP2 是未来。

== ZeRO-Offload / ZeRO-Infinity：CPU / NVMe 扩展

DeepSpeed 独有：

- *ZeRO-Offload* (Ren 2021)：把 optim state + FP32 master weight 放 CPU，update 在 CPU 用 optimized Adam。GPU 只保留 weight/grad。
- *ZeRO-Infinity* (Rajbhandari 2021)：更进一步 offload 到 NVMe，理论支持万亿参数模型。

*收益*：单卡训 10B 模型（原本装不下）。*代价*：每步 optim update 走 PCIe (63 GB/s)，慢一个数量级。

生产训练现在很少用（有钱直接多买卡）。学术复现 / 单机大模型 fine-tune 场景仍常见。

FSDP 也支持 `CPUOffload(offload_params=True)`——把 param shard 存 CPU，AG 前拉到 GPU。7B fine-tune 单卡 4090 24GB 常用。

== 与 gradient accumulation / activation checkpoint 的组合

*ZeRO-3 + gradient accumulation*：

+ FSDP `no_sync()` 支持——但要小心，`no_sync` 里 grad 是完整 unshard 保存的，*显存翻 W 倍*（本来 shard 存的现在每卡完整）
+ 更好：直接不用 `no_sync`，让每 micro-batch 都做 RS——反正 RS 与 backward compute overlap，与不做 RS 差不多

*ZeRO-3 + activation checkpoint*：完全兼容。checkpoint 层里 backward recompute 时需要 AllGather 参数——FSDP 自动处理，一样 prefetch。

*ZeRO-3 + TP*：FSDP2 支持组合。TP 组内切 tensor，TP 组间 FSDP 切 param。DeviceMesh 二维 `(tp, dp)` 表达。

== ZeRO 与 TP 的分工

面试常问："既然 ZeRO-3 已经把 weight 切 W 份，为什么还要 TP？"

答：*ZeRO-3 切的是"权重存储"，不切"权重使用"*——forward 时还要 AllGather 回来算，然后 discard。TP 是切"权重使用"—— GEMM 直接在切分的 weight 上算，输入/输出走 AllReduce。

区别：

- ZeRO-3：*显存*减少，*compute*不变（因为算之前 gather 回全 weight）
- TP：*显存*减少，*compute*也切分（每卡只算一部分 GEMM）

所以：
- ZeRO-3 一层的 activation *与不 shard 时一样大*（forward 时 param 是完整的）
- TP 一层的 activation *沿 TP 维切分*（GEMM output 是切分的）

对*小模型 + 大 batch*：ZeRO-3 activation 打满显存，TP 反而不会。
对*大模型 + 小 batch*：ZeRO-3 够用，TP 增加通信开销不划算。

一个经验规则：模型 < 20B 时 FSDP + PP 就够；模型 > 20B 或 seq > 4K，加 TP。

== FSDP 常见坑

+ *`optim.step` 后需要 `optim.zero_grad(set_to_none=True)`*：FSDP2 里 grad 存在 shard buffer，`set_to_none` 让 FSDP 内部知道可以释放
+ *`torch.save(model.state_dict())` 是 shard 的*：加载需要 same world size，或用 `torch.distributed.checkpoint` (DCP) 保存 rank-agnostic 格式
+ *混合精度 buffer 的 dtype*：BN running_mean 之类 buffer 默认 FP32，会跟 param BF16 类型不匹配。手动 `MixedPrecision(buffer_dtype=torch.bfloat16)`
+ *AC (activation checkpoint) 与 FSDP 的组合顺序*：正确顺序是 `apply_ac(module)` → `FSDP(module)`。反过来 FSDP 内部 AllGather 与 AC 的 detach 不能配合
+ *frozen parameters*：不参与训练的 param 也需要 FSDP unit 处理，否则 AllGather 语义错。用 `torch.no_grad()` 里的 param 要放独立 FSDP wrap 或者不 wrap

== 一个 FSDP 完整训练脚本骨架

```python
import torch, torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision, ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.device_mesh import init_device_mesh
from functools import partial

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank(); world = dist.get_world_size()
    torch.cuda.set_device(rank % 8)

    # HSDP: 8 卡节点内 shard, 节点间 replicate
    mesh = init_device_mesh("cuda", (world // 8, 8),
                            mesh_dim_names=("dp_replicate", "dp_shard"))

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf",
                                                 torch_dtype=torch.bfloat16)
    # 打 gradient checkpoint
    model.gradient_checkpointing_enable()

    mp = MixedPrecision(param_dtype=torch.bfloat16,
                        reduce_dtype=torch.bfloat16,
                        buffer_dtype=torch.bfloat16)

    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    wrap = partial(transformer_auto_wrap_policy,
                   transformer_layer_cls={LlamaDecoderLayer})

    model = FSDP(model,
                 sharding_strategy=ShardingStrategy.HYBRID_SHARD,
                 device_mesh=mesh,
                 auto_wrap_policy=wrap,
                 mixed_precision=mp,
                 backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                 use_orig_params=True,
                 device_id=rank % 8)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

    for step in range(100000):
        batch = next(loader)
        loss = model(**batch).loss
        loss.backward()
        # optional: grad clip
        model.clip_grad_norm_(1.0)          # FSDP-aware
        optim.step()
        optim.zero_grad(set_to_none=True)
        if step % 1000 == 0 and rank == 0:
            print(f"step {step} loss {loss.item():.3f}")
```

== 面试考点

#interview[
  *Q1*: ZeRO-3 通信量比 DDP 多 50%，为什么还用？

  A: DDP 装不下大模型。ZeRO-3 显存 $16 P/W$ 让 100B+ 模型可训。多 50% 通信换 W 倍显存减少，量纲不对等，在"不切装不下"的场景是没得选。
]

#interview[
  *Q2*: HSDP 相比 pure FSDP 收益在哪？

  A: FSDP AllGather 是每层都做，跨节点走 IB (50 GB/s)。HSDP 让 AG 只在节点内 (NVLink 400 GB/s)，跨节点只做 grad 的 AllReduce（一次，且 grad 已经 shard）。跨节点通信从 $3P$ 降到 $2P/n$（$n$ 节点数），wall-clock ~5-10× 加速。Llama-3 训练用的。
]

#interview[
  *Q3*: FSDP `auto_wrap_policy` 怎么选？

  A: 每 Transformer layer 一个 unit。unit 太大退化到 DDP（一次 AG 打满显存），太小 AG 次数爆炸（overhead 大）。用 `transformer_auto_wrap_policy(transformer_layer_cls={YourLayer})` 是通用做法。
]

#interview[
  *Q4*: FSDP + activation checkpoint 的顺序对吗？

  A: 先 apply AC 再 FSDP wrap。反过来 FSDP 内部会把整层 AllGather 出来后 checkpoint discard，backward 时需要 re-AllGather —— 语义乱。torch.distributed.checkpoint_wrapper 提供 `apply_activation_checkpointing(model, ...)`，之后再 FSDP。
]

#interview[
  *Q5*: 用 `use_orig_params=True` 有什么好处？

  A: optimizer.param_groups 看到的是原始 `named_parameters()` 而不是 FlatParameter，可以按 name 设 lr/wd/frozen。小的 bookkeeping 开销，几乎所有生产训练都开。
]

#interview[
  *Q6*: FSDP 里 `MixedPrecision.reduce_dtype=bf16` 会不会导致数值问题？

  A: 有可能。BF16 mantissa 只 7 bit，大 world_size (1024+) 下 AllReduce 累加误差累积。DeepSeek/Megatron 的做法：grad 用 BF16 通信 AR 后 upcast 到 FP32 累加，或者 `reduce_dtype=torch.float32`——通信量翻倍但精度稳。经验：< 512 卡 BF16 AR 一般 OK，> 1024 卡建议 FP32 grad AR。
]

#interview[
  *Q7*: ZeRO-3 forward 时 AllGather param 之后 discard，反向又要 AllGather 一次——为什么不缓存？

  A: 因为 activation 才是"每层被 fwd/bwd 都用到"的东西，param 缓存下来就等于没 shard。的确 fwd 和 bwd 各 AG 一次，比 DDP 多一次 AG。可以做的 mitigation 是 `limit_all_gathers=True` 里预取更远的层，让 AG 与前面 compute overlap；但绝对通信量省不了。这是 ZeRO-3 与 DDP 的固有 $P$ 差。
]

#interview[
  *Q8*: ZeRO-3 vs TP，哪个更能扩到大 world_size？

  A: ZeRO-3 (FSDP)。TP AllReduce 每层 activation，与 batch/seq 相关；world_size 大了 activation 无法充分利用 TP 组带宽。ZeRO-3 通信量与 param 相关（$3P$，$W$ 无关），带宽利用率高。所以 Llama-3 405B 用 FSDP2 主切，TP 只用 8（NVLink 域内）。
]
