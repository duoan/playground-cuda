#import "../template.typ": *

= 精度：从 AMP 到 BF16 到 FP8 / FP4

从 2018 年的 FP16 AMP 到 2024 的 FP8 全链路训练，"精度"是每次硬件迭代都会重打一遍的战场。这一章讲清楚：每种格式的数值范围、什么会溢出/下溢、GradScaler 的作用、Transformer Engine 的 FP8 recipe、DeepSeek-V3 的 blockwise FP8。

== 数值格式速览

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (left, right, right, right, right, right),
    [*Format*], [*Bits*], [*Sign*], [*Exp*], [*Mantissa*], [*Range*],
    [FP32],  [32], [1], [8],  [23], [$plus.minus 3.4 times 10^38$],
    [TF32],  [19 (\*)], [1], [8], [10], [FP32 range, FP16 mantissa],
    [BF16],  [16], [1], [8],  [7],  [$plus.minus 3.4 times 10^38$],
    [FP16],  [16], [1], [5],  [10], [$plus.minus 6.5 times 10^4$],
    [FP8 E4M3], [8], [1], [4], [3], [$plus.minus 448$],
    [FP8 E5M2], [8], [1], [5], [2], [$plus.minus 5.7 times 10^4$],
    [FP4 E2M1], [4], [1], [2], [1], [$plus.minus 6$],
    [MX FP8], [8+], [1], [4/5], [3/2], [E4M3/E5M2 + shared FP8 scale per 32-elem block],
    [MX FP4], [4+], [1], [2],   [1],   [E2M1 + shared FP8 scale per 32-elem block],
    [NVFP4],  [4+], [1], [2],   [1],   [NVIDIA 变种, 16-elem block + double scale],
  ),
  kind: table,
  caption: [常见数值格式。TF32 是 19-bit "假 32"（NVIDIA A100+ TensorCore 内部用，用户看到的是 FP32 tensor）。FP8/FP4 都是 block-scaled 或 tile-scaled，独立的 scalar 存 exponent。],
)

*关键 tradeoff*：exponent 决定动态范围（会不会 overflow/underflow），mantissa 决定精度（同一 exponent 下能表示多少个不同值）。BF16 = FP32 range + FP16 mantissa：range 够用（训练稳定），但精度差 → 累加误差大。FP16 反过来：精度好但 range 小（gradient underflow）。

== 混合精度训练的 hierarchy

*基本原则*：以*最低够用的精度做 compute，以最高精度做 accumulate*。

一次 Adam 更新的完整 dtype 流：

#figure(
  align(center, op-stack(steps: (
    ("weight (存储)",         "BF16",              "shard-h"),
    ("matmul fwd/bwd",        "BF16 in, FP32 accum", "full"),
    ("activation (存/传)",    "BF16",              "shard-h"),
    ("loss (upcast)",         "FP32",              "full"),
    ("backward → grad",       "BF16 (accum FP32)", "shard-h"),
    ("all-reduce grad",       "FP32",              "comm"),
    ("m, v update",           "FP32",              "full"),
    ("master weight update",  "FP32",              "full"),
    ("cast → weight",         "BF16",              "shard-h"),
  ), width: 7.2, cell-h: 0.55)),
  caption: [Adam step 的 dtype 流。TensorCore 天然做 BF16 → FP32 accumulate；optim state 全 FP32；master weight FP32 保留低精度更新的累积精度；只有 forward compute 与 activation 存储走 BF16。],
) <fig-adam-dtype-flow>

optim step 伪代码：

```
grad_fp32 = grad.to(FP32) / world_size
m = beta1 * m + (1-beta1) * grad_fp32          (FP32)
v = beta2 * v + (1-beta2) * grad_fp32.square() (FP32)
master_weight -= lr * m / (sqrt(v) + eps)      (FP32)
weight = master_weight.to(BF16)                (broadcast to BF16 replica)
```

FP32 master weight 存在的原因：BF16 mantissa 只 7 bit，`weight - lr * update` 里 `lr * update` 通常比 `weight` 小 1000+ 倍，直接 BF16 减会 round 到 0。FP32 master 累积小更新，broadcast BF16 用于 compute。

== FP16 与 GradScaler

FP16 因 exponent 只 5 bit，$"exp"(x)$ 里 $x$ 稍大就 overflow (65504+ = inf)。梯度小时反过来 underflow 到 0。

*GradScaler 的做法*：loss × scale (e.g. 65536)，反向传播的梯度也 ×scale，进入 FP16 表示范围。step 前 unscale 回去，检查 inf/nan 决定是否 skip step。

```python
scaler = GradScaler(init_scale=65536)
for x in data:
    with autocast(dtype=torch.float16):
        loss = model(x).loss
    scaler.scale(loss).backward()          # loss × scale
    scaler.unscale_(optim)                 # grad /= scale
    torch.nn.utils.clip_grad_norm_(...)    # 现在 grad 是真值
    scaler.step(optim)                     # 内部检查 inf, decide skip
    scaler.update()                        # dynamic scale adjust
```

*BF16 完全不需要 GradScaler*——BF16 与 FP32 exponent 相同 (8 bit)，range 一致。所以 A100/H100 时代 fp16 训练几乎绝迹。

== Transformer Engine 的 mixed precision

TE 是 NVIDIA 官方 mixed-precision library，几个关键点：

+ *fused `LayerNormLinear`*：LN + Linear 一个 kernel，中间不落 FP32 activation
+ *`LayerNormMLP`*：整个 MLP (LN + Linear + GELU + Linear) 融成一个 kernel，内部保持 high precision accumulate
+ *`DotProductAttention`*：FlashAttention wrapper，支持 BF16 和 FP8
+ *`Float8Tensor`*：管 FP8 scale 的抽象，透明地处理 cast

用法：直接替换 nn.Linear：

```python
import transformer_engine.pytorch as te

self.linear = te.Linear(H, H, bias=False)   # replaces nn.Linear
self.norm_linear = te.LayerNormLinear(H, 4*H, eps=1e-5)
```

TE 内部 kernel 是 CUTLASS 手写，比 PyTorch 默认路径快 15-30%。生产 Llama/DeepSeek 训练都用 TE。

== FP8 训练：Hopper 的杀手锏

H100 Tensor Core 支持 FP8 GEMM，peak 是 BF16 的 2×（989 TFLOPS BF16 → 1979 TFLOPS FP8）。理论算力翻倍——如果能不掉精度用起来。

难度：FP8 精度极低（mantissa 3 bit = 8 个 unique 值 per exponent），需要*精细的 scale 管理*。

=== 三种 FP8 recipe

+ *TE Delayed Scaling* (default, TE v0-v1)：每 tensor 一个 scale，用上一次的 amax 估计当前 scale。简单但对 outlier 敏感。
+ *TE Current Scaling*：每次都用当前 tensor 的 amax，无 lag，但每次要多一次 reduce 求 amax。
+ *DeepSeek Blockwise FP8*：weight 沿 (128, 128) tile 各存一个 FP32 scale；activation 沿 (1, 128) tile scale。granularity 更细，对 outlier 强 robust。DeepSeek-V3 训练用的 recipe。
+ *MXFP8* (Blackwell): hardware-native block scaling，每 32 元素共享一个 FP8 scale。B200/GB200 支持，不需要软件层管理。

=== FP8 GEMM 的 forward-backward 全流程

标准 GEMM $Y = X W$ 在 FP8 下：

*Forward*：
```
X_fp8 = quantize(X_bf16, amax_X)      # per-tile scale
W_fp8 = quantize(W_bf16, amax_W)      # per-column scale (or blockwise)
Y_fp32 = fp8_gemm(X_fp8, W_fp8)       # TensorCore FP8 in, FP32 out
Y_bf16 = Y_fp32.to(bf16)              # cast for storage
```

*Backward*（3 个 GEMM 都可以 FP8）：
```
# dX = dY · W.T (dgrad)
dY_fp8 = quantize(dY_bf16)
dX_fp32 = fp8_gemm(dY_fp8, W_fp8.T)

# dW = X.T · dY (wgrad)
X_fp8_T = X_fp8.T
dW_fp32 = fp8_gemm(X_fp8_T, dY_fp8)
```

DeepSeek-V3 三路 GEMM (Fprop/Dgrad/Wgrad) 全 FP8，只有 optimizer state 和 master weight 保持 FP32/BF16。

=== FP8 activation 存 checkpoint

Activation 用 FP8 存（比 BF16 省 2×），backward 时反量化。DeepSeek Table 3 里 activation 12% → 用 FP8 后 6%，配合其他技术总显存降 30%。

代价：quantize/dequantize kernel。Hopper 上有硬件加速，overhead < 3%。

=== FP8 vs BF16 精度差

DeepSeek-V3 tech report 报告：14T tokens 训练，FP8 vs BF16 val loss 差 *< 0.25%*。基本无损。Meta 的 Llama-3 405B 训练用 BF16 但内部实验 FP8，结论类似。

*不适合 FP8 的场景*：
+ Optimizer state (要 FP32)
+ LayerNorm / softmax（中间 accumulate 用 FP32）
+ Loss 计算
+ Attention softmax 内部（TE 里 FA-FP8 只把 GEMM 用 FP8，softmax 是 FP32）

== FP4 训练（Blackwell）

B200 支持 FP4 GEMM，peak 是 BF16 4×。目前只有 inference 广泛用，训练是研究前沿。

*NVFP4* (NVIDIA 变种)：16-elem block scaling + double scale (per-block FP8 scale + per-tensor FP32 scale)，缓解 FP4 精度不足。Nemotron 405B 训练测试报告 FP4 可行但需要精细 recipe。

生产用 FP4 训练可能要等 2026-2027。

== BF16 训练的精度坑

BF16 训练的隐藏问题：

+ *AllReduce accumulate 误差*：BF16 grad 在 AllReduce 时按 BF16 累加。$W$ 大 (1000+) 时累加误差累积，导致 loss curve 抖动。解决：`--grad-reduce-in-fp32` (Megatron)，或 FSDP `MixedPrecision(reduce_dtype=torch.float32)`。通信量翻倍，稳定性提升。
+ *LayerNorm mean/var*：BF16 mantissa 7 bit 不够表示 $sum x_i^2 / N$ 的精度。TE / Megatron 的 LN 内部用 FP32 accumulate。
+ *Softmax 内部*：$"softmax"(x) = "exp"(x - "max") / sum "exp"$，$sum$ BF16 累加会误差累积。FA/TE 内部 FP32。
+ *长序列 attention*：$Q K^T$ 的 $sum$ 沿 $d_h$ 累加，BF16 里 $d_h=128$ 累加 128 项误差在 $10^-2$ 级。FA 内部 FP32 accumulate 是标配。

*所以*："BF16 训练"实际是"weight/activation BF16，中间关键操作 FP32 accumulate"。全 BF16 会训炸。

== Loss Scale (BF16 版)

BF16 不需要 GradScaler 但依然会遇到 vanishing gradient——尤其 rank 大 (1000+ 卡) 时。补救：

+ *Loss shift*：`loss = loss * loss_scale`, `loss_scale = 1024` 之类常数。有效放大梯度，减小累加误差
+ *Z-loss* (PaLM 2)：`z_loss = 1e-4 * log(sum exp(logits))^2`，防 logits 爆炸，间接稳定梯度尺度

== 面试考点

#interview[
  *Q1*: BF16 与 FP16 都是 16 位，为什么大模型训练都选 BF16？

  A: exponent 位数。BF16 8 bit exp (与 FP32 同) 动态范围 $10^-38$ 到 $10^38$；FP16 5 bit exp 只有 $6 times 10^-5$ 到 $6 times 10^4$。梯度小时 FP16 underflow 到 0，梯度大时 overflow 到 inf。要用 FP16 得配 GradScaler；BF16 直接开箱用，代价是 mantissa 少 3 bit (7 vs 10)，精度稍差但训练无影响。
]

#interview[
  *Q2*: FP32 master weight 为什么必需？

  A: BF16 mantissa 7 bit，$"weight" - "lr" times "update"$ 里 $"lr" times "update"$ 通常比 weight 小 1000+ 倍。BF16 直接减 round 到 0，权重不动。FP32 master 累积小更新，broadcast BF16 用于 fwd/bwd。Adam 的 m/v 也在 FP32 累加，同理由。
]

#interview[
  *Q3*: GradScaler 里 dynamic scale 怎么调？

  A: 初始 65536。每步 unscale 后检查 grad 是否含 inf/nan：
  - 有 inf → skip step，scale × 0.5
  - 连续 N 步都无 inf → scale × 2
  目标是把 scale 维持在"最大不 overflow" 附近。稳态下大概每 2000 步更新一次。
]

#interview[
  *Q4*: FP8 训练里 Delayed Scaling 与 Current Scaling 的差别？

  A: Delayed 用上一次的 amax 估计当前 scale，无额外 reduce，快但对 outlier 敏感（一个 spike 让下一步 scale 猜错）。Current 每次求当前 amax，需要一次 tensor-wide reduce，overhead 3-5%，但精度更稳。TE 里 delayed 是默认，DeepSeek 用 blockwise（每 tile 一个 scale，反正每 tile 都要 reduce）。
]

#interview[
  *Q5*: DeepSeek 的 blockwise FP8 是什么？为什么比 tensor-wide FP8 稳？

  A: Weight 按 $(128, 128)$ tile 各存一个 FP32 scale；activation 按 $(1, 128)$ tile scale。粒度细，一个 outlier 只影响自己 tile 而不是整个 tensor。tensor-wide scale 里一个 spike 让全 tensor 的 scale 猜错 → 其他 tile quantize 到很低分辨率。blockwise 完全隔离。代价：scale 存储稍多、kernel 复杂化，但 CUDA-native 支持。
]

#interview[
  *Q6*: FSDP + BF16，什么时候要用 `reduce_dtype=torch.float32`？

  A: 世界大小 > 512 卡，或看到 loss curve 有 non-decreasing 抖动。BF16 grad AllReduce 里 $W$ 大时累加误差 $prop W$。$W=1024$ 时误差可能到 $10^-3$ 级别，影响 optim update。FP32 reduce 通信量翻倍但精度高。生产 Llama-3 405B 用 FP32 reduce。
]

#interview[
  *Q7*: FP8 训练里 attention 内部为什么不用 FP8？

  A: softmax 需要 max + exp + sum + divide，FP8 mantissa 3 bit 精度不足以做 exp 累加。FlashAttention-FP8 只把 $Q K^T$ 和 $P V$ 两个 GEMM 用 FP8，softmax 中间 stats 全 FP32。BF16 版 FA 里 softmax stats 也是 FP32 accumulate。
]

#interview[
  *Q8*: 一个 405B 模型用 FP8 训比 BF16 快多少？成本降多少？

  A: TensorCore FP8 peak 是 BF16 2×，但实际 MFU 会略降 (量化 overhead + 精度校验)。理论 2× wall-clock 加速，实际 1.5-1.7×。参数存储也少一半（若激活也 FP8）。整体 GPU-hour 成本 -35% 到 -45%。DeepSeek-V3 671B on 2048 H800 花了 180K GPU-hrs/T tokens = ~\$5.3M，若 BF16 需要 ~\$8-9M。
]
