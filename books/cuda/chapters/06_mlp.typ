#import "../template.typ": *

= MLP (Fused Linear + Activation)

MLP（多层感知机）是深度学习里*最朴素、也最常被 fusion 讨论*的结构：两层线性变换夹一个非线性激活。它看起来比 matmul 简单——本质上就是两个 GEMM 串起来——但工程上 90% 的优化空间不在"把 matmul 写对"，而在*中间激活要不要 materialize*、*epilogue 能不能 fuse*、*SwiGLU 这种变体怎么 numerically stable*。

这一章我们要把它讲透：

- 标准 MLP 与 SwiGLU 的数学结构，以及它们和 GEMM 的关系（第 4 章 matmul 的直接延伸）。
- Roofline：小 batch 时 memory-bound 且带宽打不满，大 batch 时 compute-bound。
- Naive 版本：两个独立 matmul + 中间激活的全局读写。
- Fused epilogue：在寄存器里做 bias + activation，省一次 HBM 往返。
- GEMM fusion 的极限：为什么 matmul1 + matmul2 一般不 fuse。
- Bias 广播、激活函数选择、backward 时的 dgelu 融合思路。
- 面试里怎么估算 fusion 收益、怎么答 GELU 近似误差。

对应源码：`src/cuda/06_mlp.cu`。本书示例用 $B=4, D_"in"=8, D_"hidden"=16, D_"out"=4$ 的小规模 batch MLP，方便对照 CPU reference；生产里 hidden 维度是 4096、12288 量级，但*优化 ladder 的结构完全一样*。

== 问题定义

=== 标准两层 MLP

给定 batch 输入 $X in RR^(B times D_"in")$，权重 $W_1 in RR^(D_"in" times D_"hidden")$, $W_2 in RR^(D_"hidden" times D_"out")$，偏置 $b_1 in RR^(D_"hidden")$, $b_2 in RR^(D_"out")$，激活函数 $"act"$：

$ H = "act"(X W_1 + b_1), quad Y = H W_2 + b_2 $

row-major 存储（与源码一致）：

```cpp
// X: B×D_in,   X[batch * d_in + i]
// W1: D_in×D_hidden,  W1[i * d_hidden + j]
// H: B×D_hidden
// W2: D_hidden×D_out
// Y: B×D_out
```

每个 batch 样本独立——$B$ 个样本可以并行，等价于 *batched GEMM*：第一层 $C_1 = X W_1$，第二层 $C_2 = H W_2$。

#note[
  PyTorch 里 `nn.Linear` 默认 $Y = X W^T + b$，权重是 $(D_"out" times D_"in")$。本书和源码用 $W_1[i,j] = W^T[j,i]$ 的 row-major 布局，和 matmul 章一致。面试手写时先问清楚 layout，再写索引。
]

=== SwiGLU 变体（LLM FFN 主流）

Llama、Mixtral 等模型的 FFN 不用单一 MLP，而用 *SwiGLU*：

$ H = "SiLU"(X W_"gate") ⊙ (X W_"up"), quad Y = H W_"down" + b $

其中 $"SiLU"(x) = x dot sigma(x)$，$⊙$ 是逐元素乘（Hadamard product）。展开后是*三个 GEMM + 一个 elementwise gate*，比标准 MLP 多 50% 的 matmul FLOP，但 empirically 效果更好。

设 $D_"in" = D_"out" = D$，SwiGLU 中间维 $D_"ff" = 4D$：标准 MLP 权重约 $2 D^2$；SwiGLU 约 $3 D D_"ff" = 12 D^2$——参数量与 weight 流量约为标准 MLP 的 6 倍。Llama 用更宽 FFN 换模型质量，推理优化首要是 weight 量化与 GEMM fusion，不是改激活公式本身。

#insight[
  SwiGLU 的 fusion 难点比 ReLU MLP 更高：gate 和 up 两个 GEMM 可以并行 launch，但 elementwise 乘依赖两者都完成；$W_"down"$ 又依赖 gate 结果。工程上常见做法是 fuse gate+up 的 epilogue（都只做 bias，无激活），再 fuse SiLU×up 为一个 elementwise kernel，最后 $W_"down"$ 走 cuBLAS + epilogue fusion。
]

== Roofline：小 batch memory-bound，大 batch compute-bound

=== 单层 GEMM 的 AI

对 $C = A B$，$A in RR^(M times K)$, $B in RR^(K times N)$，总 FLOP $= 2 M N K$。Tiled GEMM 的有效 AI（第 4 章公式）：

$ "AI" approx frac(B_M B_N, 2(B_M + B_N)) "FLOP/B" $

当 tile 足够大（如 $128 times 128$），AI $approx 32 "FLOP/B"$，超过 A100 ridge point（$approx 13 "FLOP/B"$）——*compute-bound*。

=== MLP 端到端的 AI 随 batch 变化

两层 MLP 总 FLOP（忽略激活）：

$ "FLOPs" = 2 B D_"in" D_"hidden" + 2 B D_"hidden" D_"out" $

Naive 实现（中间 $H$ materialize 到 global）的内存流量：

- 读 $X, W_1, W_2, b_1, b_2$
- 写 + 读 $H$（pre-activation 和 post-activation 各一次，naive 三 kernel 版甚至写 pre、读 pre、写 post、读 post）

以 $D_"in" = D_"hidden" = D_"out" = D$、只算 $H$ 的读写为例，每层 hidden 占 $2 B D times 4 "B"$（写 pre + 读 post）。当 $B$ 很小：

$ "AI"_"MLP,small B" approx frac(4 B D^2, c dot B D + "weight bytes") approx O(D) "FLOP/B" $

$B = 1$, $D = 4096$ 时，FLOP $approx 2 times 4096^2 approx 67 "MFLOP"$，但 weight 就有 $2 times 4096^2 times 4 "B" approx 128 "MB"$——*算一遍要把所有 weight 从 HBM 读一遍，计算密度极低*。

#insight[
  小 batch inference（$B=1$~$8$）的 MLP/Linear 层*极度 memory-bound*：瓶颈是 weight 带宽，不是 FMA。A100 2.04 TB/s 带宽下，128 MB weight 读完就要 $approx 85 mu s$，而 67 MFLOP 在 19.5 TFLOPS 上只要 $approx 3.4 mu s$——*带宽利用率看起来"打不满"*，是因为算力过剩、内存才是瓶颈。优化方向：量化（INT8/FP8 减 weight 体积）、weight 缓存到 L2、epilogue fusion 减 $H$ 读写。
]

大 batch（$B = 1024$+）时，同一份 weight 被 $B$ 个样本复用，有效 AI 随 $B$ 线性增长，逐渐进入 compute-bound——这时 tensor core GEMM tuning 才是主战场。

=== 手算一遍：decode vs prefill

Llama-2 70B 的 FFN 单层（SwiGLU，$D_"hidden" = 28672$，$D_"inter" = 8192$，三个 GEMM）：

*Prefill*：$B times S = 2048$ tokens 一起算。第一层 GEMM 的 $M = 2048$，$N = 8192$，$K = 8192$——grid 足够大，cuBLAS 能喂饱 SM，AI 高，*compute-bound 倾向*。

*Decode*：$B times S = 1$（单 token 自回归）。$M = 1$，同样的 $N, K$——grid 只有 $O(N)$ 个 block，SM 大量空闲；weight 读 $8192 times 8192 times 4 times 3 approx 768 "MB"$（三个矩阵），FLOP 只有 $approx 400 "MFLOP"$。*算力利用率接近 0，内存是全部故事*。

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left, left),
    [*场景*], [*Batch $M$*], [*瓶颈*], [*首要优化*],
    [Prefill / 训练], [$1024$+], [Compute（tensor core）], [GEMM tile tuning、FP8],
    [Decode $B=1$], [$1$], [Memory（weight BW）], [量化、KV/weight cache、fusion],
    [小模型 teaching], [$4$], [Launch + 访存], [Epilogue fuse、单 kernel],
  ),
  caption: [*Table:* MLP 在不同 batch 规模下的瓶颈类型与首要优化方向。对比 Prefill/训练（大 $M$）、Decode 单 token（$M=1$）与本书 teaching 规模三档；瓶颈列区分 compute-bound（tensor core）与 memory-bound（weight 带宽）及 launch 开销主导。],
  kind: table,
)

*Observation*：Decode $B=1$ 时 $M=1$ 使 grid 极小、读一遍 weight 的时间远超 FLOP 时间——优化首要是量化与 epilogue fusion 减 HBM 流量；Prefill/训练在 $M >= 1024$ 时有效 AI 随 batch 升高，瓶颈转向 tensor core GEMM tuning。这与前文 Roofline 一致：不能笼统说「MLP 是 compute-bound」，batch 维 $M$ 决定故事。

#note[
  面试说 "MLP 是 compute-bound" 时要加前提：*大 batch matmul 是*；*小 batch inference 的 Linear/FFN 几乎总是 memory-bound*。这和 matmul 章 ridge 分析不矛盾——AI 公式里的 $M$ 就是 batch 维。
]

=== 性能 ladder 概览

$B=4, D_"in"=8, D_"hidden"=16, D_"out"=4$（本书教学尺寸），A100 上 `ncu` 实测*端到端* kernel time 加总（见下文 `== 实测`）：

#ladder(
  ("naive (3 kernels)",     "linear1 → ReLU → linear2，H 写/读 global", "~10 μs"),
  ("fused epilogue (2)",    "fused linear1+ReLU + linear2",             "~6.7 μs"),
  ("tiled fused (1)",       "x tile + smem H + 单 kernel 两层",         "4.5 μs"),
)

#warn[
  上表是*玩具规模*上的 launch 加总，不是 $D=4096$ 生产 benchmark。端到端 tiled fused（4.5 μs）确实比 naive 三 kernel（~10 μs）快，但*单看 epilogue fusion*（fused-linear1-relu 3.14 vs naive-linear1 3.23）只有 ~3%——远小于 ladder 直觉。$D=4096$ 时差距主要来自 cuBLAS tensor core + fused epilogue，不是本章 smem tile Teaching kernel 能代表的。
]

== v1: naive — 三个 kernel，中间结果落 global

源码把 naive MLP 拆成三个 launch：

```cpp
__global__ void mlp_naive_linear1_kernel(
    const float* x, const float* w1, const float* b1, float* hidden) {
  const int batch = blockIdx.x;
  const int hidden_idx = threadIdx.x;
  if (batch >= kBatch || hidden_idx >= kHiddenDim) return;

  float acc = b1[hidden_idx];
  for (int input_idx = 0; input_idx < kInputDim; ++input_idx) {
    acc += x[batch * kInputDim + input_idx] *
           w1[input_idx * kHiddenDim + hidden_idx];
  }
  hidden[batch * kHiddenDim + hidden_idx] = acc;  // 写 pre-activation
}

__global__ void relu_kernel(float* values, int count) { /* ... */ }

__global__ void mlp_naive_linear2_kernel(
    const float* hidden, const float* w2, const float* b2, float* y) {
  /* 读 hidden，写 y */
}
```

Launch：`linear1 <<<kBatch, kHiddenDim>>>` → `relu <<<...>>>` → `linear2 <<<kBatch, kOutputDim>>>`。

=== 数据流 timeline

```
Kernel 1 (linear1):  读 X, W1, b1  →  写 H_pre 到 global
Kernel 2 (relu):     读 H_pre       →  写 H_post 到 global（或 in-place）
Kernel 3 (linear2):  读 H_post, W2, b2  →  写 Y
```

每个箭头 crossing global memory 都是 HBM 带宽消耗。ReLU 作为独立 kernel 时，即使 in-place，仍要*完整读一遍 + 写一遍* $H$——对 elementwise 操作来说，memory traffic 和计算量完全不成比例（AI $approx 0.08$ FLOP/B，比 vector add 还低）。

=== 三个问题

*1. 中间 tensor $H$ 的冗余读写*

`linear1` 写出 $H_"pre"$（pre-activation），`relu` 读 $H_"pre"$ 写 $H_"post"$，`linear2` 再读 $H_"post"$。对 $B times D_"hidden"$ 的 hidden 层，至少 *3 次 global 访问*（1 写 + 1 读 + 1 写 + 1 读，若 ReLU in-place 则 1 写 + 2 读）。

每层 hidden 占 $B D_"hidden" times 4 "B"$。$B=128, D_"hidden"=4096$ → 2 MB 一次 forward 光 $H$ 就来回 4~8 MB traffic——对 memory-bound 小 batch 这是显著开销。

*2. 三次 kernel launch*

每次 launch $approx 5 mu s$ 固定成本。小 batch、小 hidden 时 launch 占比不可忽视（和 vector add 章同理）。

*3. 没有 input 复用*

`linear1` 里每个 thread 独立读整行 $X[b, :]$——$D_"hidden"$ 个 thread 各读一遍完整的 $X$ row，*零复用*。和第 4 章 naive matmul 一样的问题，只是规模小不明显。

#note[
  naive 的价值：建立正确性。CPU reference 和三个 kernel 结果必须 bit-exact（源码 `check_output` 容差 $10^(-4)$）。优化 ladder 的每一步都在保持这个不变量。
]

=== ncu 实测

#ncu-snapshot(
  version: "naive (3 kernels)",
  size: [$B = 512$, $D_"in" = 512$, $D_"hidden" = 1024$, $D_"out" = 512$],
  rows: (
    ("linear1_kernel duration", "261.6 µs", "第一层 GEMM，占用最多时间"),
    ("relu_kernel duration",    "5.9 µs",   "elementwise，微秒级"),
    ("linear2_kernel duration", "140.3 µs", ""),
    ("*Total 3-launch*",        "*~408 µs*", "含 3 次 launch overhead"),
    ("linear1 Memory SOL",      "61.7 %",   "第一层未打满 HBM，其实 mid-tensor 写没吃满"),
    ("relu Memory SOL",         "17.7 %",   "*太短*——15 μs kernel 起不了 HBM 稳态"),
    ("linear2 Memory SOL",      "48.1 %",   ""),
    ("Achieved Occupancy",      "91.4 %",   "linear kernels 都塞满 SM"),
  ),
)

三个 kernel 序列告诉了我们两件事：

- *linear1 + linear2* 都是 GEMM，各自能拿到 memSOL 50-60%，但 relu 只有 17.7%——`relu` kernel 只跑 5.9 μs，*HBM 甚至来不及进入稳态*。这是"activation 单独作为 kernel"的经典诟病：算术极其轻，全被 launch overhead + HBM ramp-up 淹没。
- *中间 tensor `hidden[B × Dh] = 512 × 1024 × 4B = 2 MB`* 在 linear1 写 → relu 读 → linear2 读*三次* HBM。这是可以省掉的字节。

#verdict(
  problem: [三个独立 kernel 里 relu 只跑 5.9 μs 却付了完整的 launch + HBM ramp-up 成本；中间 tensor 走 HBM 三次],
  evidence: [relu 单独 memSOL 17.7%（远低于 vector add 84%）；3 次 launch overhead \~15 μs 相对小 kernel 显著；hidden tensor 2 MB 在 kernel 之间 round-trip HBM],
  next: [v2 (fused epilogue) 把 bias + ReLU 融进 linear1 —— 计算完 `matmul` 后不写 HBM，直接在 register 里 `+ bias`、`max(0, ·)`，再写出 —— 消除 relu kernel 和它的 launch overhead]
)

== v2: fused epilogue — bias + ReLU 在寄存器里完成

```cpp
__global__ void mlp_fused_linear1_relu_kernel(
    const float* x, const float* w1, const float* b1, float* hidden) {
  const int batch = blockIdx.x;
  const int hidden_idx = threadIdx.x;
  if (batch >= kBatch || hidden_idx >= kHiddenDim) return;

  float acc = b1[hidden_idx];
  for (int input_idx = 0; input_idx < kInputDim; ++input_idx) {
    acc += x[batch * kInputDim + input_idx] *
           w1[input_idx * kHiddenDim + hidden_idx];
  }
  hidden[batch * kHiddenDim + hidden_idx] = relu(acc);  // fuse 在这里
}
```

Launch 从 3 个减到 2 个：`fused_linear1_relu` + `linear2`。

=== Fused epilogue 的实现方式

在 CUTLASS / cuBLAS 语境里，*epilogue* 是 GEMM mainloop 完成累加器 $C_"acc"$ 之后、写 global 之前的那段逻辑。Teaching 版在 thread 的 `acc` register 上直接做：

```
mainloop: acc = sum_k X[batch,k] * W1[k, hidden_idx]
epilogue: acc = relu(acc + b1[hidden_idx])   // 或 acc += b1 再 relu
store:    hidden[...] = acc
```

*Tiled GEMM 版*（第 4 章 register tile）：每个 thread 维护 `acc[TM][TN]`，K-dimension 循环结束后，对 tile 内每个元素：

```cpp
#pragma unroll
for (int mi = 0; mi < TM; ++mi) {
  for (int ni = 0; ni < TN; ++ni) {
    acc[mi][ni] = relu(acc[mi][ni] + bias[col_base + ni]);  // bias 广播
    c[(row_base + mi) * n + col_base + ni] = acc[mi][ni];
  }
}
```

关键：*激活在 register 里完成，pre-activation 从未出现在 global memory*。省掉的是 $H_"pre"$ 的一次写和 ReLU kernel 的一次读+写。

=== CUTLASS 视角：Epilogue Visitor

生产 GEMM 的 epilogue 在 CUTLASS 里常写成 *Epilogue Visitor* 链：

```
Accumulator → +bias → ×alpha → activation → ×beta + C → store D
```

每个 Visitor 是编译期模板参数，nvcc 内联进 store loop——*和 Teaching 版在 `acc` 上做 `relu()` 是同一件事*，只是 scale/bias/residual 更通用。cuBLASLt 的 `cublasLtMatmul` 通过 `CUBLASLT_EPILOGUE_RELU_BIAS` 等枚举暴露同一能力，无需手写 kernel。

第二层 `$Y = H W_2 + b_2$` 同样可以 fuse bias epilogue（ReLU 已在第一层做完）。源码 `mlp_linear2_kernel` 在 register 里加 `b2[tid]` 再 store——*bias-only epilogue fusion*，只是没有 activation。

#insight[
  Epilogue fusion 有效的原因：GEMM 的累加器本来就在 register 里，bias/activation 是 $O(1)$ 逐元素操作——*marginal compute 为零，但省下的 global traffic 是实打实的*。对 memory-bound 小 batch，这往往比"再优化 mainloop 5%"更有用。
]

=== 省多少内存？手算

只 fuse 第一层 epilogue（ReLU），省 $H_"pre"$ 的一次写 + in-place ReLU 的一次读：

$ Delta "bytes" approx 2 times B D_"hidden" times 4 "B" $

$B=128, D_"hidden"=4096$：$Delta approx 4 "MB"$。A100 带宽 2.04 TB/s → 理论节省 $approx 2.7 mu s$。大模型 FFN 每层都有，stack 几十层很可观。

#interview[
  *如何估算 epilogue fusion 收益？* 数 materialize 的中间 tensor 大小和访问次数，乘以 HBM 带宽倒数。若 kernel 已 compute-bound（大 batch + tensor core），fusion 收益比例下降——因为总时间 dominated by FMA 而非 epilogue traffic。
]

=== ncu 实测

#ncu-snapshot(
  version: "fused_epilogue (bias+ReLU into linear1)",
  size: [$B = 512$, $D_"in" = 512$, $D_"hidden" = 1024$, $D_"out" = 512$],
  rows: (
    ("fused_linear1_relu",      "261.1 µs", "跟 naive linear1 几乎持平"),
    ("linear2",                 "143.0 µs", "跟 naive 一样"),
    ("*Total 2-launch*",        "*~404 µs*", "总时间只降 ~1%"),
    ("fused Memory SOL",        "62.9 %",   "vs naive linear1 61.7% —— 几乎不变"),
    ("linear2 L2 Hit",          "111.9 %",  "L2 hit > 100% 是 ncu 报告方式（含 sector 分摊）"),
  ),
)

*总时间几乎没变*——为什么？

- Fusion 消除了 `relu_kernel`（本身 5.9 μs）+ 1 次 launch overhead（~5 μs），共节省 10-11 μs。
- 但 `fused_linear1_relu` 每 thread 多几个 `+ bias`、`max(0,·)`——如果编译器 pipeline 不好，会挤占其他 issue slot。
- 更本质：*hidden tensor $H$ 依然被写到 HBM，然后被 linear2 从 HBM 读回来*。fused 版只是把 ReLU 从"独立 kernel"合进了 linear1 的 epilogue 里，*没有*让 $H$ 不落 HBM。

这个 kernel 的存在证明了一个反直觉的事实：*epilogue fusion 的性能收益，主要来自减少访存*——不是减少 launch。$B = 512, D_"hidden" = 1024$ 时 hidden = 2 MB，往返 HBM 依然是主导。

#verdict(
  problem: [fused epilogue 只消掉了 relu kernel 的启动开销，*没有*让 hidden tensor 留在片上],
  evidence: [total 从 408 μs 降到 404 μs（仅 1%）；hidden = 2 MB 依然走 HBM 一读一写],
  next: [v3 (tiled_fused) 把整个 MLP 融进*一个 kernel*：linear1 结果留在 shared memory 里，linear2 直接从 smem 读——彻底消除 hidden 的 HBM 往返（前提是 $D_"hidden"$ 装得下 smem）]
)

== GEMM fusion 的极限：为什么不 fuse matmul1 + matmul2

理想情况：一个 kernel 算完 $Y = (X W_1 + b_1) W_2 + b_2$，$H$ 永远不落 global。源码 `mlp_tiled_fused_kernel` 在*小尺寸*上部分做到了——hidden 放 smem。但 $D_"hidden" = 4096$ 时为什么不行？

=== 1. 中间维度太大，smem 放不下

Fuse 两个 GEMM 需要同时持有：

- $X$ tile（或 streaming 读）
- $W_1$ tile
- $H$ 完整中间结果：$B_M times D_"hidden"$ —— $D_"hidden" = 4096$, $B_M = 128$ → $128 times 4096 times 4 "B" = 2 "MB"$
- $W_2$ tile

A100 每 SM shared memory 上限 164 KB（与 L1 共享），典型 CTA 预算 48~96 KB。*一个 hidden 向量都放不下*，更别说 tile。

=== 2. 两个 GEMM 的 tile 布局不同

第一层：$C_1[m, n] = sum_k A[m,k] B_1[k,n]$ —— $A$ 按 row 复用，$B_1$ 按 column 复用。

第二层：$C_2[m, n] = sum_k H[m,k] B_2[k,n]$ —— $H$ 的 row 变成左矩阵的 row，$B_2$ 仍是 column-major 访问。

Mainloop 的 smem swizzle、K-dimension pipeline 是为*第一个* GEMM 的 operand 布局 tuned 的。第二个 GEMM 的 $H$ 作为左矩阵，访问 pattern 不同——*不能简单复用同一套 smem staging*，除非专门设计 dual-GEMM fused kernel（CUTLASS 有 `GemmFusion` 研究原型，但极不通用）。

=== 3. 累加维度与依赖

第一层沿 $D_"in"$ 累加，第二层沿 $D_"hidden"$ 累加。第一层*完整算完* $H$ 的一个 row（或 tile row）才能开始第二层的对应 row——*层间有 hard dependency*。Fuse 意味着要么：

- 算完整个 $H$ 再算 $Y$（需要存整个 $H$ → 回到 smem 不够），或
- 按 output tile 做 *persistent* 计算（FlashFFN 思路，极复杂）。

#warn[
  面试不要说"两个 matmul 一定能 fuse"。正确表述：*epilogue fusion*（bias+act）几乎总是值得做；*inter-GEMM fusion* 只在 hidden 足够小、或 specialized shape（如 MLP 最后一层接 softmax）时才 practical。
]

== Bias 广播：每 row 加同一个 bias

Bias $b in RR^N$ 在 $C = A B + b$ 里沿 *column*（输出维）广播：$C[i, j] += b[j]$，与 row $i$ 无关。

=== 错误做法

```cpp
// 每个 (row, col) thread 都从 global 读 bias[col]
float acc = b[col];  // 看起来没问题，但...
```

在 tiled GEMM 里，一个 block 的 $N$ 方向 tile 只有 $B_N$ 列。正确做法是 *epilogue 里每个 thread 只读一次它负责的 column 对应的 bias*，或在 block 协作阶段把 $b[j_0 : j_0 + B_N]$ 搬进 smem：

```cpp
__shared__ float bias_tile[kTileN];
if (threadIdx.x < kTileN) {
  bias_tile[threadIdx.x] = b[col_base + threadIdx.x];
}
__syncthreads();
// epilogue: acc[mi][ni] += bias_tile[ni];
```

#insight[
  Bias 广播的优化核心：*同一条 bias 元素服务一整列输出*——在 block 内 load 一次、复用 $B_M$ 次（每个 row 的 epilogue 都用同一个 `bias_tile[ni]`）。不要把它当成和 $A/B$ 一样大的矩阵去读。
]

Teaching kernel 更简单：一个 thread 负责一个 hidden/output 元素，`acc = b1[hidden_idx]` 只读一次——天然正确。

=== 和 weight 广播的对比

Bias 是 *length-$N$ 向量沿 row 广播*；weight 是 *full matrix*。面试有时会混淆：bias 只占 $N times 4$ B，相对 $K times N$ 的 weight 可忽略——但*错误实现*（每个 output element 重复 load 同一 `b[j]` 从 global）会在 epilogue 循环里放大成 $B_M$ 次冗余 load。正确做法：bias 进 smem/register 一次，复用整个 tile。

== Activation 的选择

=== 对比表

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left, left),
    [*激活*], [*Epilogue 代价*], [*Backward 保存*], [*典型场景*],
    [ReLU], [1 条 fmaxf], [1 bit mask], [Teaching、旧 CNN],
    [GELU tanh], [tanh + 几次 FMA], [input $x$ 或 mask], [BERT、GPT FFN（非 SwiGLU）],
    [SiLU / Swish], [div + expf], [$x$], [SwiGLU gate、部分 ConvNet],
    [SwiGLU gate], [SiLU + Hadamard 乘], [gate + up 输出], [Llama、Mistral FFN],
  ),
  caption: [*Table:* 常见激活函数在 epilogue fusion 中的逐元素代价、backward 需保存的中间态与典型部署场景。Epilogue 代价指 register 内完成 bias+激活的指令数；Backward 保存列说明训练时需额外 materialize 的缓冲类型。],
  kind: table,
)

*Observation*：ReLU 仅一条 `fmaxf`、backward 只需 1 bit mask——Teaching 版选它的原因；GELU/SiLU 引入 `tanh`/`expf` 级运算，epilogue fusion 仍可行但 register 压力与 latency 更高；SwiGLU 在 SiLU 之外还需 Hadamard 乘且保存 gate+up 双路输出，fusion 链比标准两层 MLP 多一步 elementwise 依赖。

=== ReLU

$ "ReLU"(x) = max(0, x) $。实现：一条 `fmaxf` 或 predicated select。Backward：$g = g_"out" dot 1_(x > 0)$。Epilogue fusion 零成本——*这是 MLP teaching 版用 ReLU 的原因*。

=== GELU

Transformer 常用。精确式：

$ "GELU"(x) = x dot Phi(x) = x/2 dot (1 + "erf"(x/sqrt(2))) $

`erf` 在 GPU 上慢。*Tanh 近似*（PyTorch 默认 `"none"` 以外的 `"tanh"` 版本）：

$ "GELU"_"approx"(x) = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))) $

```cpp
__device__ float gelu_tanh(float x) {
  const float k = 0.7978845608028654f;  // sqrt(2/pi)
  const float c = 0.044715f;
  return 0.5f * x * (1.0f + tanhf(k * (x + c * x * x * x)));
}
```

*误差*：$| "GELU"_"approx"(x) - "GELU"(x) | < 0.001$ 对 $|x| < 6$；尾部 $x arrow.r -oo$ 时近似趋向 0 比精确式慢，但对训练影响极小（梯度区域更重要）。面试常问：*为什么可以用近似？* 答：LLM 训练对激活精度不敏感，$10^(-3)$ 误差远小于 FP16 noise；推理 INT8 量化误差更大。

*Erf 精确式*：用 `erff` 或 rational approximation（CUDA `normcdf` 风格），慢 3~5×，用于需要 bit-exact 对齐的单元测试。

=== SwiGLU 与 GEGLU 数值稳定性

$ "SiLU"(x) = x / (1 + e^(-x)) $。$x$ 很大时 $e^(-x) arrow.r 0$，$"SiLU"(x) approx x$——*不会 overflow*。$x$ 很负时 $e^(-x) arrow.r +oo$，但分母也大，结果 $arrow.r 0$——用 `expf` 前 clamp 或 `log1p` 技巧更安全：

```cpp
__device__ float silu(float x) {
  return x / (1.0f + expf(-x));  // 生产代码可能对 x 做 clamp(-30, 30)
}
```

*GEGLU*：$ "GELU"(X W_1) ⊙ (X W_2) $。GELU 输出可正可负，gate 乘 up 时*不像 SwiGLU 那样天然非负 gate*——训练时 gradient 行为不同。Llama 选 SwiGLU  partly 因为 SiLU gate 平滑且非负区间有更好 empirical 表现，不是单纯数值稳定性。

#note[
  Epilogue fusion GELU 比 ReLU 贵：tanh 里有 `expf` 级运算，still $O(1)$ per element，但 register 压力和 latency 更高。cuBLASLt 的 epilogue 枚举里 `CUBLASLT_EPILOGUE_GELU` 是单独 tuned path。
]

== v3: tiled fused — 单 kernel 两层，hidden 在 smem

```cpp
__global__ void mlp_tiled_fused_kernel(
    const float* x, const float* w1, const float* b1,
    const float* w2, const float* b2, float* y) {
  __shared__ float x_tile[kInputTile];
  __shared__ float hidden_shared[kHiddenDim];

  const int batch = blockIdx.x;
  const int tid = threadIdx.x;
  if (batch >= kBatch) return;

  float hidden_acc = (tid < kHiddenDim) ? b1[tid] : 0.0f;

  for (int tile = 0; tile < kInputDim; tile += kInputTile) {
    if (tid < kInputTile) {
      const int input_idx = tile + tid;
      x_tile[tid] = (input_idx < kInputDim) ? x[batch * kInputDim + input_idx] : 0.0f;
    }
    __syncthreads();

    if (tid < kHiddenDim) {
      #pragma unroll
      for (int i = 0; i < kInputTile; ++i) {
        const int input_idx = tile + i;
        if (input_idx < kInputDim) {
          hidden_acc += x_tile[i] * w1[input_idx * kHiddenDim + tid];
        }
      }
    }
    __syncthreads();
  }

  if (tid < kHiddenDim) {
    hidden_shared[tid] = relu(hidden_acc);
  }
  __syncthreads();

  if (tid < kOutputDim) {
    float acc = b2[tid];
    for (int hidden_idx = 0; hidden_idx < kHiddenDim; ++hidden_idx) {
      acc += hidden_shared[hidden_idx] * w2[hidden_idx * kOutputDim + tid];
    }
    y[batch * kOutputDim + tid] = acc;
  }
}
```

Launch：`<<<kBatch, kTiledThreadsPerBlock>>>`，一个 block 负责一个 batch 样本。

=== 结构解读

1. *Input staging*：$X$ 按 `kInputTile=4` 分块搬进 `x_tile`，$D_"hidden"$ 个 thread 复用同一块 $X$——和第 4 章 smem tiling 同构。
2. *Layer1 epilogue*：`relu(hidden_acc)` 在 register，结果写 `hidden_shared`——*跳过 global $H$*。
3. *Layer2*：从 smem 读 `hidden_shared`，写 `y` 到 global。$W_2$ 仍从 global 读（weight 太大不 staging）。

=== 适用边界

`kHiddenDim = 16` → `hidden_shared` 只有 64 B。$D_"hidden" = 4096$ → 16 KB smem，*一个 block 可行*。$D_"hidden" = 16384$ → 64 KB，occupancy 开始受影响。再大必须回 global $H$ + cuBLAS GEMM——*Teaching kernel 展示的是 fusion 思路，不是 production FFN*。

#insight[
  Tiled fused 版的额外收益：$X$ 被 $D_"hidden"$ 个 thread 复用（smem），第二层 $H$ 在 smem 被 $D_"out"$ 个 thread 读——*小模型里连 $H$ 的 global 读写都省了*。这是 inter-GEMM fusion 在 smem 能装下时的特例。
]

=== ncu 实测

#ncu-snapshot(
  version: "tiled_fused (single kernel)",
  size: [$B = 512$, $D_"in" = 512$, $D_"hidden" = 1024$, $D_"out" = 512$],
  rows: (
    ("Duration (1 kernel)",     "732.5 µs", "*比 v2 慢 1.8×！*"),
    ("Memory SOL",              "39.4 %",   ""),
    ("Compute SOL",             "17.5 %",   ""),
    ("Achieved Occupancy",      "41.4 %",   "*↓* —— shared memory + register 都用得多"),
    ("Static SMEM / block",     "~4 KB",    "hidden_shared: D_hidden × 4B = 4096 B"),
  ),
)

*慢了！*为什么？这是本章最重要的教学时刻：*fusion 不是免费的*。

- Teaching kernel 是"每个 block 处理一个 batch row"设计——`gridDim = kBatch = 512`。每 block 只有 `max(kHiddenDim, kOutputDim) = 1024` 个 thread 单独承担一整个 output row 的所有工作。
- $D_"hidden" = 1024$ 时 hidden_shared 占 4 KB smem——不算大。但 kernel 里 *thread 复用 X* 依赖大量 `for k in kInputDim` 的串行 load，*没有 tile 化*，也*不是 GEMM-friendly kernel*。这是 "prove fusion is possible" 的教学 kernel，不是 "prove fusion is fast" 的 production kernel。
- 生产上 MLP fusion 走的是 cuBLASLt / CUTLASS：先做 GEMM 主循环（保 Tensor Core 效率），epilogue 里 fuse bias + ReLU。inter-GEMM fusion 只在 $D_"hidden"$ 极小时才可能有收益（LoRA、adapter 场景）。

#final-verdict(
  status: [教学 ladder 展示了 epilogue fusion（v2）和 inter-GEMM fusion（v3）两种思路。],
  note: [*不要复制 tiled_fused kernel 到生产环境*。v2 (epilogue fusion) 是可以直接进 CUTLASS / cuBLASLt 的模板；v3 只在 hidden 很小时才 make sense。真正的 MLP 加速在 Tensor Core 上 —— 见第 04 章"CUTLASS 分层"节讨论。]
)

== Backward：dgelu 融合思路（简述）

Forward epilogue fusion 的 mirror：backward 时激活的梯度也在 *GEMM backward 的 epilogue* 里做，避免 materialize $g_"pre"$。

ReLU backward：$g_"in" = g_"out" dot 1_(x > 0)$，需要 forward 时保存 *mask* 或 *sign bit*（1 bit/element，比存完整 $H$ 便宜）。

GELU backward（对 tanh 近似）：

$ (partial "GELU"_"approx")/(partial x) = 0.5(1 + tanh(...)) + 0.5 x (1 - tanh^2(...)) dot (partial tanh(...))/(partial x) $

*Fusion 做法*：weight gradient 的 GEMM（$partial L / partial W = X^T partial L / partial Y$）mainloop 完成后，在 epilogue 里对 $partial L / partial Y$ 应用 dgelu，*直接得到 $partial L / partial H$* 用于 input gradient GEMM——或更常见：先 elementwise `g_pre = dgelu(g_out, x_saved)` fuse 进一个 kernel，再 launch input gradient GEMM。

PyTorch 的 `torch.compile` / cuDNN fused ops 就是把 `dAct → dGemm` 链合成最少 kernel。

=== 两层 MLP backward 数据流

Forward 存：$H$（或 ReLU mask）、$X$（第一层 input grad 需要）。Backward 顺序：

1. $partial L / partial H = (partial L / partial Y) W_2^T$ —— GEMM
2. $partial L / partial H_"pre" = "dAct"(partial L / partial H)$ —— elementwise（可 fuse 进上一步 epilogue）
3. $partial L / partial W_2 = H^T (partial L / partial Y)$，$partial L / partial b_2 = "rowsum"(partial L / partial Y)$ —— GEMM + reduction
4. $partial L / partial X = (partial L / partial H_"pre") W_1^T$ —— GEMM
5. $partial L / partial W_1 = X^T (partial L / partial H_"pre")$ —— GEMM

*Fusion 机会*：步骤 1 的 GEMM epilogue 直接做 dReLU/dGELU，输出作为步骤 4 的 input——省 $partial L / partial H$ 的 global round-trip。训练框架里这常和 *gradient checkpointing* 权衡：checkpoint 不存 $H$，backward 重算 forward 的 $H$ 和激活，用 compute 换 memory。

== 生产栈：cuBLASLt 与 torch.compile

手写 Teaching kernel 是为了理解结构。Production FFN 路径：

1. *cuBLASLt*：`cublasLtMatmul` + `CUBLASLT_EPILOGUE_GELU_BIAS` / `RELU_BIAS`，一次 launch 完成 $X W + b + "act"$。
2. *CUTLASS*：自定义 epilogue functor，支持 SwiGLU 的 multi-GEMM schedule。
3. *torch.compile / TensorRT*：graph 层自动识别 `linear → relu → linear` 模式，匹配 fused pattern。

#note[
  面试问 "你怎么优化 MLP"：先答 epilogue fusion + 量化 + batching；再说手写 kernel 只在 shape 特殊或框架 overhead 太大时考虑。能调用 cuBLASLt fused epilogue 就不要重写 GEMM mainloop。
]

== 实测

$B=4, D_"in"=8, D_"hidden"=16, D_"out"=4$（与源码一致），A100 80GB SXM4，`ncu` 抓取。表中是*各 kernel 单次 launch* 的 `gpu__time_duration.sum`——naive 路径三次 launch，fused epilogue 两次，tiled fused 一次。

Launch 配置决定 warp lane 利用率：

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (left, left, left, right),
    [*version*], [*grid*], [*block*], [*threads*],
    [naive-linear1 / fused-linear1-relu], [(4, 1, 1)], [(16, 1, 1)], [64],
    [naive-linear2 / linear2], [(4, 1, 1)], [(4, 1, 1)], [16],
    [tiled-fused], [(4, 1, 1)], [(32, 1, 1)], [128],
  ),
  caption: [*Table:* ch6 MLP 各 kernel 版本的 launch 配置。*grid* / *block* 对应 `<<<grid, block>>>`（ncu `launch__grid_size` / `launch__block_size`）；*threads* 为 grid 内 block 线程总数。naive/fused-linear1 用 `<<<kBatch, kHiddenDim>>>`，linear2 用 `<<<kBatch, kOutputDim>>>`，tiled-fused 用 32-thread block。],
  kind: table,
)

*Observation*：linear2 的 block 仅 4 thread（$D_"out"=4$），远小于 warp 32——diag 表 `issued/32` 会低至 4.0；linear1/fused 各 16 thread 占半 warp（`issued/32 = 16.0`）。Teaching kernel「一 thread 一输出元素」在小 output/hidden 维上是结构性 lane 浪费，生产 $D >= 1024$ 时 block 才喂饱 warp。

linear1 用 `<<<kBatch, kHiddenDim>>>`——一 thread 一 hidden 元素，block 只有 16 thread。linear2 用 `<<<kBatch, kOutputDim>>>`——block 只有 *4 thread*。Teaching kernel 的"1 thread per output element"写法，在 output 维 $< 32$ 时会把整颗 warp 的空 lane 浪费掉。

#include "../bench/06_mlp.typ"

#warn[
  这一章的问题规模是教学 default（$B times D_"hidden" approx 64$ 个 float），kernel 单次运行只有 3–5 μs。ncu 的定性指标（`issued/32`、`bank conflicts`、`barrier stall`）仍能反映 kernel 结构，但*绝对数字对生产规模不完全可信*：
  - HBM % 会偏低（分母 elapsed time 含冷启动窗口）
  - dram_bytes 可能被 L2 消化，`GB/s (实测/逻辑)` 两列差距明显
  想拿到生产规模的数字，把主参数（rows/cols/hidden dim）加到让工作集远超 L2 (40 MB)。
]

*先看 perf 表（compute 章看 TC %，不是 HBM %）：*

- *TC % 全部为 0.0*——Teaching 版用 `float` + CUDA core FMA，没有 `wmma::` / `mma.sync`。生产 cuBLASLt GEMM 应 > 60%。
- *warp % 全部 0.0*——4 block × 16 thread 填不满 108 个 SM；decode $B=1$ FFN 是同一类并行度问题。
- *HBM % 全部 0.1%*——工作集 < 2 KB，全在 L2；*不能*用 HBM % 判断 fusion 收益。
- *time*：fused-linear1-relu 3.10 μs vs naive-linear1 3.10 μs——epilogue fusion 在 device 时间里*测不出差*；tiled-fused 4.51 μs 比两 kernel 加总（6.65 μs）快，但 barrier 开销让它比单层 linear1 还慢 45%。

*再看 diag 表——本章最重要的教学点：*

*a) naive-linear2：`issued/32 = 4.0`——全书最极端的 lane 浪费*

output 维 $D_"out" = 4$，launch 为 `<<<4, 4>>>`。每个 block 只有 4 个 thread——凑不满一个 warp（32 lane）。硬件仍按 warp 粒度发射指令，*28 个 lane 每拍都在空转*。

#figure(
  warp-lanes(active: range(4), cell: 0.34,
             title: "naive-linear2：blockDim = D_out = 4，仅 lane 0–3 有活干"),
  caption: [绿色 = 负责一个 output 元素的 thread。灰色 = 占 issue slot 但无 output 可算——*不是* predication 能藏起来的浪费。]
)

`pred_on/32 = 3.9`，与 `issued/32 = 4.0` 几乎重合——*没有* "issued 高但 pred_on 低" 的 predication 假象。和 softmax / reduce 章不同：那些 kernel 里 `if (tid < n)` 编译成 predicated instruction，ncu 仍显示 `issued/32 approx 32`，predicated-off lane 藏在 `issued − pred_on` 的 gap 里。这里 gap 几乎为零——*空 lane 是 launch 配置造成的结构性浪费*，issue slot 真的只喂了 4 个 lane。

#insight[
  这是全书最强证据："1 thread per output element" 在 output 维 $< 32$ 时是坏主意。$D_"out" = 4$ 意味着每 warp 有效吞吐只有 $4/32 = 12.5%$。生产 decode 里 $D_"out" = 4096$ 时 block 够大，问题消失——但 teaching default 故意把小维暴露出来。
]

*b) naive-linear1：`issued/32 = 16.0`——半个 warp*

hidden 维 $D_"hidden" = 16$，block 16 thread。每个 warp 只有 16 个 lane 参与每条 issued 指令——另外 16 个 lane 空转。fused-linear1-relu 同样是 16.0 / 15.6——ReLU epilogue fusion *不改变* lane 利用率，只省 global traffic（在本规模测不出）。

#figure(
  warp-lanes(active: range(16), cell: 0.34,
             title: "naive-linear1：blockDim = D_hidden = 16，半个 warp 有活干"),
  caption: [16 个 active lane 对应 16 个 hidden neuron；另外 16 个 lane 结构性 idle。]
)

*c) tiled-fused：`issued/32 = 18.3`，`pred_on/32 = 12.6`——tile 边界 predication*

block 32 thread，比 linear1/2 饱满，但 `kInputTile = 4`、`kHiddenDim = 16`、`kOutputDim = 4` 都不能整除 32。kernel 里大量 `if (tid < kHiddenDim)`、`if (tid < kOutputDim)` 编译成 predicated 指令——`issued/32 = 18.3` 表示平均每条指令约 18 个 lane 参与；其中只有 12.6 个 lane 的 predicate 为 on。gap $18.3 - 12.6 = 5.7$ 是*真正的 predicated-off lane*，占 issue slot 但不干活。

`barrier stall = 0.76`——input tile 循环里两次 `__syncthreads`，32-thread block 比 16/4-thread block 的 sync 开销更实。`smem conf. = 0`——80 B smem 无 bank conflict（不能从源码猜，要以 metric 为准）。

*d) fusion 路径的 device 时间都在 noise 里*

naive-linear1 与 fused-linear1-relu 同为 3.10 μs、同为 `issued/32 = 16.0`——省掉的 pre-activation write 在 64 元素规模下不可见。端到端收益在*少一次 launch*（~5 μs host 侧），不在 `ncu` 表的 kernel time。linear2 与 naive-linear2 同为 `issued/32 = 4.0`——bias-only epilogue 不改变 lane 故事。

#warn[
  不要把"省 $2 B D_"hidden" times 4$ B HBM traffic"直接翻译成"epilogue fusion 快 40%"。本章 micro-benchmark 的 device 时间差在 noise 里；大 hidden + memory-bound decode 时，同一公式才给出 μs–ms 级节省。
]

*端到端加总（launch 次数不同）：*

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*path*], [*launch 数*], [*device time 加总*],
    [naive], [3（linear1 + ReLU + linear2）], [~10 μs（ReLU 未单独 profile，按 ~3 μs 估）],
    [fused epilogue], [2（fused-linear1-relu + linear2）], [6.65 μs],
    [tiled fused], [1], [4.51 μs],
  ),
  caption: [*Table:* MLP 三条优化路径的 launch 次数与 device time 加总（μs，`gpu__time_duration.sum`）。naive 三次 launch；fused epilogue 两次；tiled fused 一次。ReLU 在 naive 路径未单独 profile，按 ~3 μs 计入加总。],
  kind: table,
)

*Observation*：玩具规模上 tiled fused（4.51 μs）端到端快于 naive 三 kernel 加总（~10 μs），但单看 epilogue fusion（fused-linear1 与 naive-linear1 同为 3.10 μs）device 时间几乎无差——收益主要来自少 launch 与省 $H$ 的 global 读写；放大 hidden dim 且 memory-bound decode 时，同一 fusion 公式才给出 μs–ms 级 HBM 节省。

tiled fused 端到端最快，但绝对时间都在 launch overhead 量级。面试和生产决策应基于 $D >= 1024$ 的 cuBLASLt + `dram__bytes` 差分，不是本章表格。

== ncu 该看什么

```bash
ncu --set full --section SpeedOfLight ./build/06_mlp
```

关键 metric（compute 章优先看 lane 利用率，不是 HBM %）：

- `smsp__thread_inst_executed_per_inst_executed.ratio`（issued/32）：naive-linear2 = *4.0*（全书最低），naive-linear1 = 16.0——output/hidden 维 $< 32$ 时 "1 thread per element" 的结构性浪费。
- `smsp__average_thread_inst_executed_pred_on_per_inst_executed.ratio`（pred_on/32）：与 issued/32 几乎重合（linear2: 3.9 vs 4.0）⇒ *真实空 lane*，不是 predication 能藏起来的 gap。tiled-fused: 12.6 vs issued 18.3 ⇒ ~40% predicated-off。
- `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`（TC %）：全 0——Teaching 版无 tensor core；生产 GEMM 应 > 60%。
- `sm__warps_active.avg.pct_of_peak_sustained_elapsed`（warp %）：0.0——4 block 填不满 GPU。
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld+st.sum`（smem conf.）：tiled-fused = 0——无 bank conflict。
- `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio`（barrier stall）：tiled-fused = 0.76——input tile 循环的 `__syncthreads` 开销。
- `dram__bytes.sum.pct_of_peak_sustained_elapsed`（HBM %）：~0.1%——L2 全命中；对比 fusion 收益要把 hidden 加大到远超 L2。

== 面试白板 code

面试官说"手写一个 MLP forward（Linear + GELU + Linear）"——不要一个个 kernel 单独启（会被追问 fuse）。给这份：把 activation fuse 到 GEMM epilogue，Linear2 独立算，最后再 fuse residual add + LayerNorm。

```cpp
// Linear1 + GELU 融合成一个 kernel（GEMM 骨架同 ch4）。
// h1 = GELU(x @ W1 + b1)，x: [B, D], W1: [D, F], b1: [F], h1: [B, F].
constexpr int TILE = 32;

__device__ __forceinline__ float gelu_tanh(float x) {
  constexpr float k0 = 0.7978845608f;   // sqrt(2/pi)
  constexpr float k1 = 0.044715f;
  return 0.5f * x * (1.f + tanhf(k0 * (x + k1 * x * x * x)));
}

__global__ void linear_gelu(const float* x, const float* W1, const float* b1,
                            float* h1, int B, int D, int F) {
  __shared__ float xs[TILE][TILE];
  __shared__ float ws[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;   // batch 行
  int col = blockIdx.x * TILE + threadIdx.x;   // 输出通道 (F 维)
  float acc = 0.f;

  for (int kt = 0; kt < D; kt += TILE) {
    xs[threadIdx.y][threadIdx.x] =
        (row < B && kt + threadIdx.x < D) ? x [row * D + kt + threadIdx.x] : 0.f;
    ws[threadIdx.y][threadIdx.x] =
        (kt + threadIdx.y < D && col < F) ? W1[(kt + threadIdx.y) * F + col] : 0.f;
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) acc += xs[threadIdx.y][k] * ws[k][threadIdx.x];
    __syncthreads();
  }

  // ==== Epilogue fuse: bias + GELU. 直接写最终值，避免多一次 gmem read/write ====
  if (row < B && col < F) {
    float v = acc + b1[col];      // bias 沿 F 维 broadcast
    h1[row * F + col] = gelu_tanh(v);
  }
}

// ==== Launch config ====
// 和普通 tiled GEMM (ch4) 完全一样——epilogue fuse 不改变 launch shape.
// blockDim = (TILE, TILE) = (32, 32); gridDim = ((F+31)/32, (B+31)/32).
// 关键点: bias b1 沿 F 维 broadcast——warp 内 lane 沿 col (threadIdx.x) 分布,
// 32 lane 读 b1[col..col+31] = 32 连续 fp32 → coalesced 1 次 gmem transaction.
dim3 block(TILE, TILE);
dim3 grid((F + TILE - 1) / TILE, (B + TILE - 1) / TILE);
linear_gelu<<<grid, block>>>(x, W1, b1, h1, B, D, F);

// Linear2 用普通 GEMM (同 ch4 骨架)，输出 h2 = h1 @ W2 + b2.
// 训练时再来一个 fused kernel: y = LayerNorm(h2 + x_residual)——见 ch5.
```

*核心考点*（追问顺序）：

- *"为什么 GELU fuse 进 Linear1、Linear2 却不 fuse activation？"* → Linear1 的 output 就是 activation 输入，写完 acc 立即用 GELU 消掉、不写 gmem 中间量，省一次 read/write。Linear2 后面没有 activation（是 residual add，另一个 fuse 单元）。
- *"epilogue fuse 的边界是什么？"* → 只能 fuse *element-wise* 和 *broadcast*（bias、activation、scale）。需要 reduce 的（softmax、LayerNorm）不能在 GEMM epilogue 里 fuse——它们需要跨 tile 的信息。
- *"Linear1 + Linear2 能不能一起 fuse？"* → 不行。Linear2 的 K 维 = Linear1 的输出维 F，两次 GEMM 中间要 *跨 tile* 完整看到 h1——这需要 grid barrier 或分成两 kernel。除非 F 很小（能塞进 smem），可以做 "back-to-back GEMM"（cuBLASLt 支持）。
- *"backward 怎么 fuse？"* → $partial L / partial "acc" = partial L / partial h_1 dot "GELU"'("acc")$——GELU derivative 在 backward 时也 fuse 到 Linear1 的 backward epilogue。参数 grad ($partial L / partial W_1, partial L / partial b_1$) 是 GEMM $x^T dot "grad"$，独立 kernel。
- *"SwiGLU 呢？"* → gate 和 up 两条 Linear 并行算（可以 fuse 成一个 GEMM 输出双倍列），然后 element-wise `silu(gate) * up`——见附录 D 第 8 题。
- *"epilogue fuse 后 register 压力会不会爆？"* → GELU tanh 版几条指令、几个额外 register，占用增加约 5-10 个 reg，可以接受。fuse LayerNorm 那种需要 row reduction 的就不行——需要 smem 存中间量 + `__syncthreads`，和 GEMM 的 K 循环结构冲突。

== 面试考点

#interview[
  *Q1*: 为什么 epilogue fusion（bias + ReLU）有效？

  A: GEMM 累加器已在 register；bias/激活是 O(1) 逐元素操作，marginal compute 为零，但省去 pre-activation 的 global write 和 activation kernel 的 global read/write。小 batch memory-bound 时收益按 `$2 B D_"hidden" times 4$ B` / bandwidth 估算。
]

#interview[
  *Q2*: 为什么不把两个 matmul fuse 成一个 kernel？

  A: (1) 中间 $H$ 维度大，smem 放不下；(2) 两层 GEMM operand layout / swizzle 不同；(3) 层间 hard dependency——第一层完整输出才是第二层输入。Epilogue fusion 可行，inter-GEMM fusion 只在 hidden 小或 specialized kernel 可行。
]

#interview[
  *Q3*: GELU tanh 近似和 erf 精确式差多少？能用在训练吗？

  A: $|x| < 6$ 时误差 $< 10^(-3)$；训练完全可用，误差远小于 FP16/BF16 noise。推理若 INT8 量化，激活近似更不重要。
]

#interview[
  *Q4*: SwiGLU vs GEGLU？

  A: SwiGLU 用 SiLU gate，输出平滑、非负区间多；GEGLU 用 GELU gate，可负。Llama 选 SwiGLU 是 empirical + 三 GEMM 结构，不是唯一数值稳定选择。SwiGLU 需三个 weight matrix，FLOP 比标准 MLP 多 50%。
]

#interview[
  *Q5*: Bias 广播在 tiled GEMM epilogue 里怎么实现？

  A: bias 长度 = 输出列数 N；每个 block 把负责的 $b[j:j+B_N]$ 搬进 smem 或 register，tile 内每个 row 复用。不要 per-element 重复读 global bias。
]

#interview[
  *Q6*: 小 batch MLP 为什么 bandwidth "打不满"？

  A: 不是带宽坏了，是*算力相对内存太多*——weight 读一遍的时间远大于 FLOP 时间，kernel memory-bound，SM 大量时间在等 HBM。看起来 compute util 低、memory util 也"不够满"，因为 problem 太小、并行度不够饱和 memory controller。
]

#interview[
  *Q7*: 如何估算 MLP fusion 端到端收益？

  A: 列出每个 intermediate tensor 的 shape 和 read/write 次数 × 4 B × elem_count，加总 delta bytes；除以 peak BW 得 latency 节省。加上减少的 launch 数 × 5 μs。大 batch compute-bound 时比例缩小，用 ncu 实测为准。
]

#interview[
  *Q8*: 第二层 linear 的 bias 要不要 fuse？和第一层有何不同？

  A: 要 fuse。第二层没有 activation 时 epilogue 只剩 `acc += b2[j]; store`——同样应在 register 完成，避免先写 bare matmul 结果再读回加 bias。Teaching 版 `mlp_linear2_kernel` 已在 dot 循环后加 `b2[tid]`，等价于 bias-only epilogue fusion。
]

#interview[
  *Q9*: SwiGLU 比标准 MLP 多哪些 kernel / 内存压力？

  A: 三个 GEMM（gate、up、down）+ 一次 SiLU + 一次 Hadamard 乘。比两层 MLP 多 50% matmul FLOP 和一份 gate/up 中间态；fusion 策略是 gate/up 并行 GEMM，SiLU×up elementwise fuse，down 走 fused bias epilogue。
]
