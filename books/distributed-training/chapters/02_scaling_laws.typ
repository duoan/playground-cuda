#import "../template.typ": *

= 显存、算力与三堵墙

在讲任何具体并行策略之前，你必须能*心算*一个模型在特定 setup 下的显存和算力占用。这一章给三张速查表 + Roofline + 三堵墙模型，为后续所有优化提供诊断框架。

== 参数量速查

一个标准 GPT-style Transformer，参数量：

$ N approx L times (12 H^2) + V H $

（$L$ 层，$H$ hidden dim，$V$ vocab size；不含 LN/bias 的小项）

其中每层的 $12 H^2$ 来自：

- $Q K V O$ 四个 projection：$4 H^2$
- FFN 两个 linear ($H -> 4H, 4H -> H$)：$8 H^2$

GLU-family (SwiGLU / GeGLU) 用三个 linear ($H -> (8/3) H, H -> (8/3) H, (8/3) H -> H$)，参数量还是约 $8 H^2$。

*典型模型速查*：

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (left, right, right, right, right, right),
    [*Model*], [*L*], [*H*], [*A*], [*V*], [*Params*],
    [GPT-2 (small)],  [12], [768],   [12], [50257],  [124M],
    [GPT-3],          [96], [12288], [96], [50257],  [175B],
    [LLaMA-2 7B],     [32], [4096],  [32], [32000],  [7B],
    [LLaMA-2 70B],    [80], [8192],  [64], [32000],  [70B],
    [LLaMA-3 405B],   [126],[16384], [128],[128256], [405B],
    [Mixtral 8×7B],   [32], [4096],  [32], [32000],  [47B (13B active)],
    [DeepSeek-V3],    [61], [7168],  [128],[129280], [671B (37B active)],
    [Qwen3-235B (MoE)],[94],[4096],  [64], [151936], [235B (22B active)],
  ),
  kind: table,
  caption: [常见模型参数速查。MoE 括号里是每 token 激活量。],
)

== 显存组成：单卡训练的四头

不做任何并行时，训练一个模型显存 = weight + grad + optim state + activation：

#figure(
  table(
    columns: (auto, auto, 1fr),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, right, left),
    [*组成*], [*字节 / 参数*], [*说明*],
    [Weight (BF16)],  [2],  [模型主权重，混合精度下 fwd/bwd 用这个],
    [Weight (FP32 master)], [4],  [Adam 更新用；如果不用 master 则省掉],
    [Grad (BF16 或 FP32)], [2 或 4], [和 update 精度一致],
    [Adam m (FP32)],  [4],  [一阶动量],
    [Adam v (FP32)],  [4],  [二阶动量],
    [Total (标准 AdamW BF16 混合精度)], [*16*], [$2 + 4 + 4 + 4 + 2 approx 16$ per param],
    [Total (Precision-aware, m/v BF16)], [10], [DeepSeek/Megatron-Core 用],
    [Total (SGD momentum)], [8], [$2 + 4 + 2 = 8$],
  ),
  kind: table,
  caption: [每参数字节数。标准 AdamW 混合精度是 *16 bytes / param*，这是所有显存讨论的基线。],
)

*一句话公式*（AdamW 混合精度）：

$ "mem"_"model" " (GB)" = N " (B)" times 16 $

*所以*：
- LLaMA-2 7B：7 × 16 = 112 GB → *单卡 A100 80GB 装不下*
- LLaMA-2 70B：70 × 16 = 1120 GB → *需要 14+ 卡*
- DeepSeek-V3 671B：671 × 16 = 10736 GB → *需要 135+ 卡起步*

*Activation* 更麻烦：与 batch/seq 强相关。粗估（BF16，不 recomp）：

$ "act" approx s dot L dot B dot S dot H $

$s$ 大约在 30-60 之间（Megatron 2022 activation recomp paper 里给了精确公式），$L$ 是层数，$B$ micro-batch，$S$ seq len，$H$ hidden dim。

LLaMA-2 7B，$L=32, B=1, S=4096, H=4096$：

$ "act" approx 34 times 32 times 1 times 4096 times 4096 times 2 approx 34 "GB" $

Recomp 之后可以降到 $2 L B S H$（每层只存 input）约 2 GB。

$ "act"_"recomp" approx 2 L B S H $

== 算力速查

*Forward FLOPs* per token（Chinchilla 公式简化版）：

$ "FLOPs"_"fwd" "per token" approx 2 N $

（$N$ = 参数量。系数 "2" 因为一个 matmul 是 $2 M N K$ FLOPs，而每个参数在 fwd 中恰好乘 1 次。忽略 attention 的 $S H$ 项——在 $S << H$ 时成立。）

*Forward + Backward*：

$ "FLOPs"_"train" "per token" approx 6 N $

（fwd 一次、bwd 输入梯度一次、bwd 权重梯度一次；每次都是 $2 N$ FLOPs。）

*Attention 修正*：$S$ 大时 attention 的 $O(S^2 H)$ 不可忽略。精确：

$ "FLOPs"_"train" "per token" approx 6 N + 12 L S H $

举例，LLaMA-3 70B 在 $S=8192$：$6 times 70 dot 10^9 + 12 times 80 times 8192 times 8192 approx 4.2 dot 10^11 + 6.4 dot 10^10 approx 4.9 dot 10^11$，attention 项占 13%。

*训练一个模型总 FLOPs*：$C = 6 N D$（$D$ = tokens）。GPT-3 175B on 300B tokens = $3.14 times 10^23$ FLOPs。

*MFU (Model FLOPs Utilization)*：

$ "MFU" = ("achieved FLOPs") / ("peak FLOPs of hardware") $

H100 BF16 peak = 989 TFLOPS。如果 step time 里每卡处理 $B S$ tokens：

$ "achieved TFLOPS" = (6 N B S) / ("step_time" times 10^12) $

*行业参考*：
- Dense LLM 训练：40-55% MFU 是好的（GPT-4-class）
- MoE：35-50% MFU（多了 dispatch overhead）
- FP8 训练：30-45% MFU（peak 变 2×，绝对 TFLOPS 更高但 utilization 降）

== Roofline：一个 kernel 值不值得优化

Roofline 说 kernel 性能受两个屋顶限制：算力 (compute) 或带宽 (memory)。

$ "achievable perf" = min(pi_"peak", beta_"peak" times I) $

- $pi_"peak"$：硬件峰值 (H100 BF16: 989 TFLOPS)
- $beta_"peak"$：HBM 带宽 (H100: 3.35 TB/s)
- $I$：arithmetic intensity (FLOPs / byte)

*ridge point*：$I^* = pi / beta = (989 dot 10^12) / (3.35 dot 10^12) approx 295$ FLOPs/byte。

意思：kernel 每读一 byte 至少要做 295 FLOPs，才能跑到峰值算力。低于这个就是 memory-bound，高于就是 compute-bound。

*常见 kernel 的 intensity*：

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (left, right, left),
    [*Kernel*], [*Intensity (FLOP/byte)*], [*状态*],
    [Element-wise (add, gelu)],  [0.25], [memory-bound],
    [LayerNorm],                 [1-2],  [memory-bound],
    [Softmax],                   [3-5],  [memory-bound],
    [Attention (MHA, S=4K)],     [50-100], [memory-bound (无 Flash)],
    [FlashAttention v2 (S=4K)],  [100-200], [compute-bound ~50%],
    [GEMM (M=N=K=1024)],         [~340], [compute-bound],
    [GEMM (M=N=K=4096)],         [~1300], [compute-bound],
    [GEMM (M=8, N=K=8192)],      [~15],  [memory-bound (small-M GEMM)],
  ),
  kind: table,
  caption: [常见 kernel 的算术强度。GEMM 的 intensity 随 $min(M,N,K)$ 线性增长——小 M（decode / MoE per-expert）就是 memory-bound，这是 MoE 训练 compute wall 的根源。],
)

#insight[
  面试常问："为什么 batch size 越大 MFU 越高？" 答：MFU 直接对应 arithmetic intensity。BS 大意味着 attention 和 FFN 的 GEMM M 维大，越过 ridge point，越接近 compute-bound。BS 小时是 memory-bound，无论怎么优化算法都拿不到峰值。
]

== 三堵墙：训练系统的诊断框架

我们借用 Megatron-Core MoE Tech Report 的"三堵墙"框架，扩展到所有大规模训练。任何一次训练卡顿最终都能归为：

+ *Memory Wall*：显存装不下。症状：OOM，或被迫用小 batch。
+ *Communication Wall*：通信占太多时间。症状：nsys 里 comm 占 step > 30%。
+ *Compute Efficiency Wall*：算力没用满。症状：MFU < 40%，SM 利用率低。

三堵墙*互相耦合*：解决内存墙常常增加通信（切分权重需要 AR/AG）；overlap 通信需要更多 batch（增加显存）；小 M GEMM 让 compute 掉率，恰恰是切碎权重的副产物。

*每堵墙的武器谱系*，本书对应章节：

#figure(
  table(
    columns: (auto, 1.5fr, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*墙*], [*武器*], [*本书章节*],
    [Memory],
    [ZeRO / FSDP; TP; PP; Activation recomp; Activation offload; Precision-aware optim; FP8 activation; CPU offload; Selective recompute],
    [第 4-6, 9-10 章],
    [Communication],
    [DDP bucket; ZeRO overlap; TP-SP; Ring/Ulysses attention; Hierarchical a2a; DualPipe; FWD-BWD merged; FP8 dispatch; NCCL tuning],
    [第 1, 3-8, 11 章],
    [Compute Efficiency],
    [Grouped GEMM; FlashAttention; TE fused kernel; CUDA graph; Sync-Free MoE; SM carve-out policy; batch size / seq packing 让 M 变大],
    [第 8-11, 12 章],
  ),
  kind: table,
  caption: [三堵墙的武器谱系。定位问题时先 profile 判定属于哪堵墙，再挑对应工具。反过来"这个技术很酷"式的堆叠通常没收益。],
)

== 一个"能不能塞下"的心算 checklist

拿到一个新模型 + 新集群时，30 秒决定"能不能跑起来"：

+ *模型 param 量 $N$*：查论文或 `sum(p.numel() for p in model.parameters())`
+ *单卡显存*：$16 N$（AdamW BF16 混精）；MoE 记 $N_"total"$ 不是 active
+ *总卡数需求（纯 DP，不切模型）*：$"cards" = ceil(16 N / 80 "GB")$；这是绝对下限，任何切分方案都能省
+ *切完之后 activation*：$B S H L times "factor" times 2$，其中 factor 视 recomp 策略
+ *算 step time*：$6 N B S / ("MFU" times 989 "TFLOPS")$ 每卡，加通信估算
+ *总训练时间*：$6 N D / ("cards" times "MFU" times 989 "TFLOPS")$

例：LLaMA-2 70B on 512 × H100, D=1T tokens, MFU=50%:

$ (6 times 70 dot 10^9 times 10^12) / (512 times 0.5 times 989 dot 10^12) = 1.66 dot 10^6 " s" approx 19.2 " 天" $

一个数量级估算 30 秒能得出。这个能力是分布式训练面试的"入场券"。

#figure(
  align(center, stacked-bar(
    entries: (("compute", 15.0), ("TP AR", 3.2), ("DP AR (overlap)", 0.1),
              ("PP bubble", 0.8), ("overhead", 0.5)),
    width: 10, bar-h: 0.7,
    title: "典型 70B/TP4/DP4/PP4 一次 step 时间构成 (估算, 秒)",
  )),
  caption: [一次 step 时间的组成。TP AR 是不能 overlap 的显式通信；DP AR 100% 被 backward 隐藏；PP bubble 与 (P,m) 有关。完整推导见「面试数学」章 (M)。],
) <fig-step-time-breakdown>

#insight[
  拿到面试题时先按上面这张 stacked-bar 的四项报数。90% 情况下你能给出 ±30% 的答案，比背公式实用。完整 estimator：`src/distributed_training/estimators.py`。
]

== 面试考点

#interview[
  *Q1*: 为什么 AdamW 混合精度是 16 bytes/param？

  A: BF16 weight (2) + FP32 master weight (4) + BF16/FP32 grad (2 或 4) + FP32 momentum (4) + FP32 variance (4)。用 grad=BF16 就是 $2+4+2+4+4=16$。如果不保留 FP32 master 是 12。Precision-aware optimizer (m/v BF16) 是 10。
]

#interview[
  *Q2*: FLOPs = 6 ND 里的"6"是怎么来的？

  A: 每个参数在 forward 中乘 1 次 activation (2 FLOPs：乘 + 累加)。Backward 里对 input 和 weight 各求一次导 = 2×2 = 4 FLOPs。总 6 FLOPs/param/token → $6 N D$ tokens 训完。
]

#interview[
  *Q3*: 一个 70B 模型 MFU 40%，你觉得"正常"吗？

  A: 依赖 setup。dense 70B on 512 H100 with BF16, TP=8/PP=2/DP=32, S=8K：40% 偏低（GPT-3 era 已经能 50%+）。可能原因：通信没 overlap 好、activation recomp 太激进、data loading 慢。可以问 profile 看 comm / compute ratio。
]

#interview[
  *Q4*: 为什么 MoE 训练比 dense MFU 低 5-10%？

  A: 三个来源：(1) all-to-all 通信 overlap 不完全；(2) grouped GEMM 的 M 维小（每 expert 只有 1/E 的 tokens），小-M GEMM 未饱和 TensorCore；(3) permute / bincount / cross-rank sync 的 launch overhead。用 fine-grained MoE + DeepEP + capacity padding 能把差距压到 ~3%。
]

#interview[
  *Q5*: Roofline 上一个 kernel 是 memory-bound，你说"用更大 batch"—— 那 attention 呢？attention 的复杂度是 $O(S^2)$，batch 大 memory 反而爆了。

  A: 分两层看：*Q/K/V 的 projection matmul* 是 batch-dominated，BS 大 → M 大 → compute-bound。*QK^T 和 PV 的 attention matmul* 是 seq-dominated（每 head 的 M = S），S 大就有 M 大 → 也 compute-bound。所以 FlashAttention 的 tile 沿 S 而非 BS 切——单个 tile 里 M = block_size 已经越过 ridge point，与 BS 无关。这解释了为什么 FA 对小 batch decode 也很快。
]

#interview[
  *Q6*: activation 显存怎么估？给一个 LLaMA 7B, B=1, S=8192 的数字。

  A: 精确要看 Megatron 2022 paper 公式 5-6。简易估：$"act" approx (s dot L dot B dot S dot H)$，$s approx 34$ (unfused) 或 $17$ (fused, TE)。7B: $L=32, H=4096$。$s=17$ (TE): $17 times 32 times 1 times 8192 times 4096 times 2 = 36 "GB"$。开 selective recompute 后 attention 部分省 $10-15 "GB"$，剩 ~20 GB。
]

#interview[
  *Q7*: 什么是"three walls"？举一个案例说明它们互相耦合。

  A: Memory / Communication / Compute Efficiency Wall。案例：为解决 memory wall 引入 EP=64 切 MoE → 暴露 communication wall (a2a 变主要开销) → 上 DualPipe overlap → 需要 2× 权重副本 → 又拉高 memory wall → 用 FP8 activation 压缩 → 引入 quantize/dequantize kernel + 小 group 计算 → 拉低 compute efficiency wall → 用 grouped GEMM + CUDA graph 缓解。整个链条就是"三堵墙互相制约"的具体展开。
]
