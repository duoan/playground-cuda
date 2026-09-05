#import "../template.typ": *

= Activation Checkpoint 与 Offload

Activation 是训练显存的大头（DeepSeek-V3 Table 3 里占 66%）。这一章讲清楚 activation checkpoint 的原理、selective / fine-grained 版本、以及 offload 到 CPU/NVMe 的时机。

== Activation 从哪来

一层 forward 计算图里，每个中间张量都被 backward 需要（求梯度要 input）。粗略 accounting，一层 Transformer 的 activation：

- LN input & output：$2 B S H$
- Q, K, V projection output：$3 B S H$
- Attention scores (FA 后省)：$B S^2 A$ (naive) 或 $0$ (FA)
- Attention output：$B S H$
- FFN input：$B S H$
- FFN intermediate：$B S I approx 4 B S H$
- FFN output：$B S H$
- Residual：saved for grad flow

一共约 $17-34 B S H$ per layer（因是否 fused、是否 Flash 而异）。80 层 = 上 TB。

== Activation Checkpoint (recompute) 的基本思想

Chen et al. 2016 提出。核心：*只存少量 activation (每 layer 的 input)，backward 时重算前向*。

```python
from torch.utils.checkpoint import checkpoint

class TransformerLayer(nn.Module):
    def forward(self, x):
        return checkpoint(self._forward, x, use_reentrant=False)

    def _forward(self, x):
        # 完整的 attention + FFN
        ...
```

Backward 到这层时，PyTorch 会*重新跑一遍 forward*（不存中间），拿到 activation 后算 grad，再 free 掉。

*收益*：activation 从 $17 B S H$ 降到 $B S H$（只存 input）。80 层模型省 16× activation。

*代价*：forward 跑 2 次（一次原本，一次 backward 重算），总 compute + ~30-35% (backward 一般是 forward 的 2×，加一遍 forward = 33%)。

*MFU 代价*：25-30%。

== Selective Recompute

Megatron 2022 (Korthikanti et al.) 提出：*只对 attention 层做 checkpoint*。

Attention 的 activation 里 $B S^2 A$（未 FA 时）是超大项 —— 选择性 checkpoint attention 能省最大头，同时 FFN 部分（$4 B S H$，也不小）不 recompute 保持 forward 效率。

Megatron flag: `--recompute-granularity selective` (默认 recompute attention only) 或 `full` (整层)。

用 FlashAttention 之后 attention activation 已经很小 ($B S H$ 而非 $B S^2 A$)，selective 收益变小。但仍有意义：Q/K/V matmul 的 activation、softmax 的 stats 还是几十 GB per layer。

== Fine-grained Recompute (DeepSeek / Megatron-Core MoE)

进一步细化：不 recompute 整层，只 recompute *几个特定的 cheap 子模块*，避免 recompute 引发的重通信。

Megatron-Core flag:
```
--recompute-granularity selective
--recompute-modules mla_up_proj layernorm moe_act moe mlp core_attn
```

各模块含义：
- `mla_up_proj`：DeepSeek MLA (Multi-head Latent Attention) 的 up-projection，特别耗 activation
- `layernorm`：LN 内部的 mean/var stat activation
- `moe_act`：MoE expert 里的 SwiGLU intermediate（$B S I / "EP"$ 大头）
- `core_attn`：FA 内部的 lse
- `mlp`：dense MLP intermediate

*为什么不整层 recompute MoE*：MoE 层含 dispatch/combine all-to-all——整层 recompute 会*重触发 a2a*，通信量翻倍。fine-grained 只 recompute 计算部分。

DeepSeek-V3 Table 3-4 报告：fine-grained recompute 省 activation ~42 GB/GPU，compute overhead $\< 5%$。相比整层 recompute (30% overhead) 是巨大进步。

== Activation Offload

不 recompute，而是把 activation *放到 CPU DDR 或 NVMe*。fwd 时 D2H，bwd 时 H2D。

*PCIe 带宽*：Gen4 32 GB/s，Gen5 63 GB/s。CPU DDR5 ~90 GB/s per socket。所以 offload 的瓶颈是 PCIe。

*何时值得*：
+ activation 太大装不下，且 recompute 也没帮到显存需求
+ 有闲置 PCIe 带宽（TP/EP 通信主要用 NVLink）
+ 单 layer activation < CPU DDR (~几百 GB，通常满足)

*何时不值得*：
+ recompute 已经够
+ PCIe 已经被别的用（peer memory copy, dataloader 拉数据）
+ compute 太快，D2H 来不及

=== Fine-grained Offload (Megatron-Core)

Megatron 2026 tech report 提出：*不整层 offload*，而是选特定模块（`expert_fc1`, `moe_act`, ...）offload。用独立 CUDA stream 与 1F1B 的 a2a 兼容 overlap。

Flag:
```
--fine-grained-activation-offloading
--offload-modules expert_fc1 moe_act
```

DS-V3 场景报告 mem -10.7%, throughput -1.6%——几乎白赚 10% 显存。Qwen3-235B 换 module mapping 后甚至 +15% 吞吐（因为 offload 让 batch 更大, kernel 效率提升）。

=== PyTorch Native Offload (FSDP CPUOffload)

FSDP 支持:
```python
FSDP(model, cpu_offload=CPUOffload(offload_params=True))
```

粗粒度：整个 param shard 都 offload，需要时拉回。适合单机 fine-tune 大模型（4090 24GB 训 7B）。不适合多卡训练（PCIe 竞争）。

== Optimizer Offload

*ZeRO-Offload* (DeepSpeed) / *Optimizer CPU Offload* (Megatron `--optimizer-cpu-offload`)。

Optimizer state 12P 是大头。放 CPU：
- Fwd/bwd：不需要 optim state
- Step：把 grad 传给 CPU (D2H)，CPU 更新 (fused CPU-Adam)，master weight 传回 (H2D)
- Latency：一次 optim step 从 ~5 ms 变 ~100 ms

*收益*：约 12P 显存。DS-V3 场景 15-20 GB。iter time +0.1-0.2 s（小 batch 时无所谓）。

生产建议：先 FSDP + Precision-aware optim（optim state BF16 存 GPU），OOM 再上 CPU offload。

== Precision-Aware Optimizer (Megatron-Core)

DeepSeek/Megatron 观察：Adam m/v 用 FP32 存的必要性不高 —— m/v 只用于 update 计算，用 BF16 存 + step 时 upcast FP32 计算，误差可忽略。

Flag:
```
--use-precision-aware-optimizer
--exp-avg-dtype bf16
--exp-avg-sq-dtype bf16
```

*收益*：optim state 从 12P/param 降到 ~6P/param，*显存省 50%*。这是最"零成本"的 memory 优化。

*代价*：fused kernel 里多几次 dtype cast，微小 latency。精度：Megatron 官方 benchmark 显示训练曲线基本一致。

== 组合策略

真实训练里几种都可以叠：

```
Precision-Aware Optim (m/v BF16)  → optim state -50%
   +
Selective Recompute (attn only)   → activation -30%
   +
Fine-grained Recompute (moe_act)  → activation 再 -20%
   +
Fine-grained Offload              → activation 再 -15%
   +
FP8 Activation                    → activation storage -50%
```

DeepSeek-V3 stack 起来把 199.5 GB/GPU 压到 H100 80GB 能训。

== Recompute vs Offload：怎么选

#figure(
  table(
    columns: (auto, auto, auto, 1fr),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, auto, auto, left),
    [*方法*], [*Compute overhead*], [*Bandwidth overhead*], [*适用*],
    [Full recompute],    [+30%], [0],           [简单 baseline，短 seq 用],
    [Selective recompute],[+5-15%], [0],        [FA 之后 dense 训练标配],
    [Fine-grained recompute], [+3-5%], [0],     [MoE 训练标配 (避免重触发 a2a)],
    [Full offload],      [0],   [~PCIe bound], [PCIe 有余量时],
    [Fine-grained offload], [0], [~PCIe overlap], [MoE + 1F1B stream 有余量时],
    [FP8 activation],    [+2-3%], [0],         [Hopper+, DeepSeek recipe],
    [Precision-aware optim], [$approx 0$], [0], [几乎白赚，先开],
  ),
  kind: table,
  caption: [各种 memory 优化对比。生产训练通常混用：Precision-aware + Fine-grained recompute + FP8 activation 组合起来能省 60%+ 显存。],
)

*选择顺序*：
+ 打开 Precision-aware optim (`--use-precision-aware-optimizer`)
+ 打开 FSDP / ZeRO-3 切 weight
+ 打开 activation recompute (先 selective，OOM 再 fine-grained)
+ 如果还 OOM，开 fine-grained offload
+ 如果仍 OOM，开 FP8 activation
+ 都开满还 OOM，减 GBS 或加卡

== 面试考点

#interview[
  *Q1*: Activation checkpoint 的 compute overhead 为什么是 33%？

  A: 一次 backward 计算量约 forward 的 2×（对 input 求导 + 对 weight 求导）。加上 checkpoint 需要重算一次 forward → total = 1F + 1F(重算) + 2B = 4F。原本 = 1F + 2B = 3F。overhead = 4/3 - 1 = 33%。selective recompute 只 checkpoint 部分层，overhead 小。
]

#interview[
  *Q2*: FlashAttention 出来后 activation checkpoint 还有必要吗？

  A: 有。FA 只把 attention matrix $B S^2 A$ 那一项省掉；activation 还有 $B S H$ (LN input, residual)、$B S I$ (FFN intermediate)、Q/K/V 各 $B S H$。长 seq (8K+) 时这些累积起来仍 GB 级。selective recompute 对 attention layer 内的非-FA 部分（Q/K/V matmul） checkpoint 也有用。
]

#interview[
  *Q3*: 为什么 MoE 场景整层 recompute 不好？

  A: MoE 一层含 dispatch + combine all-to-all。整层 recompute 意味着 backward 时 forward 全跑一遍——包括*重触发 a2a 通信*。a2a 是 MoE 的主要通信开销，翻倍代价太大。fine-grained recompute 只重算 compute 部分（moe_act, expert intermediate），跳过 dispatch/combine 的 activation。
]

#interview[
  *Q4*: `--use-precision-aware-optimizer` 有什么代价？为什么不默认开？

  A: 代价：几乎为 0。fused kernel 里多几次 BF16↔FP32 cast，微小 latency。收益：optim state -50%。为什么不默认：这是 Megatron 2024 才加的 flag，需要 fused optimizer kernel。DS-V3 用；Llama-3 405B 训练时还没这功能。现在的新项目默认开就对。
]

#interview[
  *Q5*: Activation offload 到 CPU 什么时候会拖慢训练？

  A: 三个场景：(1) 每层 activation 太大，D2H+H2D 时间超过 compute；(2) PCIe 被别的用（peer copy for TP, dataloader batch fetch）；(3) CPU DDR 带宽不够（多 GPU 共享一个 socket，聚合 D2H > 400 GB/s 超 DDR 上限）。诊断：nsys 看 GPU 是否等 memcpy_HtoD kernel。
]

#interview[
  *Q6*: FP8 activation 与 recompute 冲突吗？

  A: 不冲突，正交。FP8 activation 只是"更小的存储"，recompute 是"不存重算"。两者可以叠：只存少量关键 activation 用 FP8，其他 recompute。DeepSeek-V3 stack 里都有。
]

#interview[
  *Q7*: 你怎么决定要不要开 activation checkpoint？

  A: 先 profile 看单卡 activation 显存。如果 activation < 30% 总显存，不用 checkpoint（compute overhead 不值）。activation > 50%，先 selective recompute。> 70% 且 OOM，fine-grained recompute + offload + FP8。目标：让 batch size 尽量大（→ MFU 高），activation 恰好卡在 80% 显存以内。
]
