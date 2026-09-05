#import "../template.typ": *

= Attention (Naive Scaled Dot-Product)

scaled dot-product attention 是 transformer 的心脏。面试里它几乎必考——不是让你背公式，而是追问：*中间矩阵有多大、HBM 读写几次、mask 怎么写、multi-head 怎么并行、为什么长序列 naive 版本直接 OOM*。这一章只讲*教科书式*的三 pass 分解：$Q K^T arrow.r "softmax" arrow.r @ V$，把 $S = Q K^T$ *完整 materialize* 到 global memory。Flash-Attention 的分块 fused 路径留到第 8–10 章；本章末尾会说明为什么那条路*不得不*走。

对应源码：`src/cuda/07_attention.cu`（教学规模 $N = 8, d = 8$；正文用 LLM 典型 shape 推导）。

本章 optimization ladder（naive 路径）：

#ladder(
  ("scores (QK^T)",     "1 block / query row, dot product",  "0.1%"),
  ("softmax",           "1 block / row, smem + 3 pass",      "0.1%"),
  ("value (P @ V)",     "1 block / query row, weighted sum", "0.1%"),
  ("tiled (fused)",     "1 kernel, smem + online softmax",   "0.1%"),
  ("+ causal mask",     "scores kernel 内置 -1e30",          "—"),
  ("+ multi-head",      "grid.y = head index",               "—"),
)

ladder 里百分比是 A100 上 `sm__warps_active`（warp %），不是相对 naive 的加速比——$N = 8$ 时三 pass 各 ~3.2–3.6 μs、合计 ~9.9 μs，fused tiled 5.5 μs（见 `=== 实测`）。三 pass 各自独立 launch；瓶颈不在单 kernel 的 occupancy，而在 *$O(N^2)$ 中间态 + 三次 HBM round-trip*（生产规模）或 *launch 累加*（本章 micro-benchmark）。

本章在全书中的位置：第 3 章 softmax 提供 row-wise 归一化与 online 公式；第 4 章 matmul 提供 $Q K^T$ / $P V$ 的 GEMM 视角与 tiling 直觉；本章把二者*按教科书拼接*，暴露 naive 拼接的 IO 代价；第 8 章起用 FlashAttention 拆掉 $S$ 这一层 global buffer。读代码顺序建议：先 `./build/07_attention` 看 naive vs tiled 输出一致，再对照本章三 kernel 逐步 host launch。

== 问题定义

=== 数学

对单个 head，给定 $Q, K, V in RR^(N times d)$（$N$ = sequence length，$d$ = head dimension），scaled dot-product attention：

$ "Attention"(Q, K, V) = "softmax"(frac(Q K^T, sqrt(d))) V $

等价地，令 $S = Q K^T / sqrt(d) in RR^(N times N)$，$P = "softmax"(S) in RR^(N times N)$（按行 softmax），输出 $O = P V in RR^(N times d)$。

*Scaled* 的 $1/sqrt(d)$ 来自 Vaswani et al. (2017)：$d$ 增大时点积方差变大，不 scale 则 softmax 进入饱和区、梯度变小。

=== 矩阵形式（单 head）

把 $Q$ 的每一行当作一个 query 向量 $q_i^T in RR^(1 times d)$，$K$ 的每一行当作 key $k_j^T$，则

$ S[i,j] = frac(q_i^T k_j, sqrt(d)) = frac(sum_(ell=0)^(d-1) Q[i,ell] K[j,ell], sqrt(d)) $

Softmax 对*固定 query 行 $i$* 在列 $j$ 上归一化：$P[i,j] = exp(S[i,j] - m_i) / sum_(j') exp(S[i,j'] - m_i)$，其中 $m_i = max_j S[i,j]$。输出 $O[i,d] = sum_j P[i,j] V[j,d]$。整个流程对 $i = 0..N-1$ 独立——*行与行之间无依赖*，并行轴是 $(B, H, N)$ 上的 query 行。

#note[
  与 matmul 对照：$S = Q K^T$ 是 $(N times d)(d times N)$；$O = P V$ 是 $(N times N)(N times d)$。Attention 不是「一个新算子」，而是*两个 GEMM 夹 softmax*——优化时分别套用第 3、4 章工具，再面对二者之间的 $N^2$ 中间态。
]

=== Multi-head 与 batch 形状

生产代码用 4D tensor，本书记号与前言一致：

- $B$：batch size
- $H$：head 数
- $N$（或 $S$）：序列长度
- $d$：每 head 维度（通常 $d = "hidden" / H$）

$Q, K, V$ shape 均为 $(B, H, N, d)$。一个 head 的 attention 是 $(N, d) times (N, d)^T arrow.r (N, N) arrow.r (N, d)$。*所有 head 共享同一套算法*，head 之间无数据依赖——天然并行轴。

#note[
  PyTorch `nn.MultiheadAttention` 里 $Q/K/V$ 常先 reshape 成 $(B, N, H dot d)$ 再 split head；GPU kernel 更常见 layout 是 $(B, H, N, d)$ 或 $(B, N, H, d)$。写 kernel 前必须确认 stride——本书用 row-major $(B, H, N, d)$：`q[b*H*N*d + h*N*d + i*d + k]`。
]

=== 与 GEMM / Softmax 的关系

Attention 可拆成两个 batched GEMM + 一次 row-wise softmax（第 3、4 章）：

1. $S = Q K^T$：对每个 $(b, h)$，$S_(b,h) = Q_(b,h) K_(b,h)^T$，shape $(N, N)$。
2. $P = "softmax"(S / sqrt(d))$：把 $S$ 看成 $B dot H dot N$ 行、每行 $N$ 列的矩阵，复用 softmax 章的 block-per-row 模板。
3. $O = P V$：对每个 $(b, h)$，$O_(b,h) = P_(b,h) V_(b,h)$，shape $(N, d)$。

Naive 实现的关键特征：步骤 1 的 $S$ *写回 global*，步骤 2 *读 $S$、写 $P$*（源码里 in-place 覆盖 $S$），步骤 3 *读 $P$、写 $O$*。三步之间无 fusion，每步都是完整 global round-trip。

#insight[
  面试第一问往往是「attention 计算复杂度？」——FLOPs 是 $O(B H N^2 d)$（两个 $N times N times d$ 的 matmul），但 naive *内存*是 $O(B H N^2)$ 存 $S/P$。*长序列瓶颈是内存，不是 FLOPs*。  第二问常跟「那为什么 FA 有用」——答 IO 复杂度从 $O(N^2)$ 降到 $O(N)$，不是 FLOPs 复杂度。
]

== Roofline：$O(1)$ AI 与 $N^2$ 内存灾难

=== 算术强度（端到端 naive）

忽略 $Q, K, V$ 的输入只读一次（训练时它们来自上一层，往往已在 cache），*增量*成本集中在 $S$ 和 $P$：

- 写 $S$：$N^2$ floats
- 读 $S$、写 $P$（softmax）：约 $2 N^2$ floats traffic
- 读 $P$、写 $O$：$N^2 + N d$ floats

对固定 $N, d$，总 FLOPs $approx 4 B H N^2 d$（两个 matmul 各 $2 N^2 d$），总 HBM traffic 主导项 $approx 4 B H N^2 times 4 "B"$（$S/P$ 各读写一遍）。

有效 AI（按 $N^2$ 中间矩阵摊销）：

$ "AI" approx frac(4 N^2 d "FLOP", 16 N^2 "B") = frac(d "FLOP", 4 "B") $

$d = 64$ → AI $approx 16 "FLOP/B"$，在 A100 ridge ($approx 13$) 附近——*看似 compute-bound*。但：

1. *实际更 memory-bound*：$Q, K, V$ 的读取、softmax 的 exp/div、三次 launch 同步，都会拉低有效 AI。
2. *$N^2$ 内存与带宽同阶增长*：$N = 8192$ 时单 head 的 $S$ 就有 $8192^2 times 4 "B" approx 256 "MB"$；32 heads × batch 8 → *数十 GB* 仅中间矩阵——还没算 activation checkpoint 的反向。

#warn[
  Roofline 上的 AI 用 FLOPs/$N^2$ 字节有时「看起来还行」，但 *materialize $N times N$ 矩阵* 本身在长序列下不可行。面试要说清：*复杂度分 FLOPs、内存、带宽三条线*；naive attention 死在内存 $O(N^2)$ 和 HBM 读写 $O(N^2)$，不是死在 FLOPs。
]

=== 内存占用手算

单 head、FP32：

$ "mem"(S) = N^2 times 4 "B" $

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, center, center),
    [*$N$*], [*单 head $S$*], [*$H=32$, $B=1$*],
    [1024],  [4 MB],   [128 MB],
    [4096],  [64 MB],  [2 GB],
    [8192],  [256 MB], [8 GB],
    [32768], [4 GB],   [128 GB],
  ),
  caption: [*Table:* attention score 矩阵 $S in RR^(N times N)$ 的 FP32 显存占用（$N^2 times 4$ B）。左列为序列长度 $N$；中列为单 head；右列为 $H=32$、$B=1$ 时所有 head 的 $S$ 缓冲合计。],
  kind: table,
)

*Observation*：$N$ 每翻 2×，$S$ 显存翻 4×——$N=32768$ 单 head 即 4 GB，32 head 合计 128 GB，远超单卡容量。Roofline 上 AI 看似接近 ridge，但 materialize $N times N$ 矩阵本身在长序列下不可行；这是 Flash-Attention 的动机，不是 matmul tiling 能解决的。

LLM 上下文 32K 时，*仅一层* attention 的中间矩阵就可占满 A100 40GB 显存的一部分；多层、反向、optimizer state 叠加 → naive 训练/推理都不可接受。

#insight[
  memory-bound 的直接后果在这里体现为 *$S/P$ 字节数随 $N$ 平方增长*。算 AI 时若只数 FLOPs 不看 $N^2$ buffer，会误判「$d=64$ 已超 ridge，attention 该 compute-bound」——这是 naive 分析最常踩的坑。
]

理论上界（仅 $S/P$ traffic，不含 $Q,K,V$）：

$ T_"min, map" approx frac(4 B H N^2 times 4 "B", 2.04 "TB/s") $

$N = 4096, B = 8, H = 32$：$T_"min, map" approx 45 "ms"$。任何 naive 实现若端到端 attention 远低于此仍 OOM，说明问题在*分配*而非带宽。

#insight[
  这就是为什么工业界默认 Flash-Attention / fused attention：目标不是把 AI 从 15 提到 20，而是 *把 $O(N^2)$ HBM traffic 降到 $O(N)$*（第 8 章 IO 分析）。本章 naive 的价值是让你*亲手感受* $S$ 矩阵的存在。
]

=== 手算一遍：$N = 4096$, $d = 64$, $B = 8$, $H = 32$

固定典型 LLM 一层 self-attention（忽略 KV cache 增量解码，只看 full forward）：

*FLOPs*：两个 matmul 各 $2 B H N^2 d$ → $2 times 2 times 8 times 32 times 4096^2 times 64 approx 2.2 times 10^12$ FLOP（2.2 TFLOP / layer）。A100 FP32 峰值 19.5 TFLOPS，*纯算力*约 0.1 s——看起来很快。

*中间矩阵 HBM*：$S$ 或 $P$ 占 $B H N^2 times 4 "B" = 8 times 32 times 4096^2 times 4 approx 17 "GB"$。A100 40GB 上*一层 forward* 就吃掉 40%+ 显存给 attention map，还没算 $Q,K,V$ 和 MLP。

*$S/P$ 相关 HBM traffic*（naive 三 pass，in-place softmax）：

- Pass 1 写 $S$：$17 "GB"$
- Pass 2 读+写 $S arrow.r P$：$approx 34 "GB"$
- Pass 3 读 $P$：$17 "GB"$

合计 $approx 68 "GB"$ *仅 attention map 的读写*（不含 $Q,K,V$）。A100 HBM 2.04 TB/s → 68 GB 需要 $approx 45 "ms"$ *下限*——还没算 matmul 读 $Q,K,V$ 和 exp/div。实测 naive PyTorch attention 在 4K 序列上往往*数百 ms*，和这张账一致。

#note[
  手算时把 FLOPs 和 HBM GB 分开写。面试官给 $N=8192$ 时，先算 $N^2 times 4 times B times H$ 是否 OOM，再谈 kernel 优化——顺序反了会答偏。
]

== v1: Step 1 — $S = Q K^T / sqrt(d)$

源码 `attention_scores_kernel`：每个 query 行一个 block，每个 key 列一个 thread，做 $d$ 维点积。

```cpp
constexpr int kSeqLen = 8;
constexpr int kHeadDim = 8;
constexpr float kScale = 1.0f / sqrtf(static_cast<float>(kHeadDim));

__global__ void attention_scores_kernel(
    const float* q, const float* k, float* scores, bool causal) {
  const int row = blockIdx.x;   // query index
  const int col = threadIdx.x;  // key index
  if (row >= kSeqLen || col >= kSeqLen) return;

  if (!causal || col <= row) {
    float acc = 0.0f;
    for (int d = 0; d < kHeadDim; ++d) {
      acc += q[row * kHeadDim + d] * k[col * kHeadDim + d];
    }
    scores[row * kSeqLen + col] = acc * kScale;
  } else {
    scores[row * kSeqLen + col] = -1e30f;
  }
}
```

Launch：`<<<kSeqLen, kSeqLen>>>`——$N$ 个 block，每 block $N$ threads。$N=8$ 时只有 64 个 thread 在算 scores；$N=4096$ 时 grid 有 4096 blocks × 4096 threads/block，*远超*单 SM 容量，但每个 thread 只做 $d$ 次 FMA，算术强度极低。

=== 并行模型诊断：为什么 scores kernel 不能上生产

把这个 kernel 当成「每个输出元素一个 thread 的 dot product」，和第 4 章 matmul naive 同构：

*1. $Q$ 行复用为零*

`blockIdx.x = row` 固定时，block 内 32 个 thread 读*不同* $K[j, :]$（$j$ = key index），但读*同一* $Q[i, :]$（$i$ = row）——$Q[i,:]$ 被 $N$ 个 thread 各读一遍，本应 block 协作读一次进 smem。

*2. $K$ 访问不合并*

同一 warp 内 `col` 连续 → 读 $K[j, d]$ 时 $d$ 循环内合并，但不同 `col` 之间 $K$ 行起点间隔 $d times 4$ B——warp 并行读不同行时，对 global memory 仍较友好；然而 $N$ 很大时 block 有 4096 thread，远超一个 warp，occupancy 和 launch 调度才是首要问题。

*3. 算术强度*

每 thread：读 $2d$ float，写 1 float，算 $2d$ FLOP → AI $approx d/6$ FLOP/B。$d=64$ → $approx 10.7$ FLOP/B，仍低于 matmul tiled 后水平，且*没有任何 smem 复用*。

#note[
  教学 kernel 用「row × col 一个 thread」是为了一目了然 $S[i,j] = Q[i,:] dot K[j,:]$。生产 $Q K^T$ 走 cuBLAS / CUTLASS batched GEMM + tensor core——本章重点是 *$S$ 写回 global* 这一下，不是 tune 这个 dot product kernel。
]

=== 这就是一个 batched matmul

把 $Q$ 看成 $(N, d)$，$K^T$ 看成 $(d, N)$，则 $S = Q K^T$ 是 $(N, N)$ GEMM。cuBLAS 调用形态：

```cpp
// 单个 head: C = alpha * A * B + beta * C
// A = Q (N×d), B = K^T 逻辑上 (d×N)，C = S (N×N)
cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            N, N, d,
            &alpha, K, d, Q, d, &beta, S, N);
// alpha = 1/sqrt(d) 可融进 epilogue
```

*Batched*：对 $(B, H)$ 的每个 slice 调一次，或用 `cublasGemmStridedBatched` / `cublasGemmBatched`，stride 为 $N dot d$（相邻 head 在内存中连续时）。

Strided batched 示例（layout $(B, H, N, d)$，同一 batch 内 head 连续）：

```cpp
const int64_t stride_q = num_heads * seq_len * head_dim;
const int64_t stride_s = num_heads * seq_len * seq_len;
const float alpha = 1.0f / sqrtf(static_cast<float>(head_dim));

cublasGemmStridedBatched(
    handle,
    CUBLAS_OP_T, CUBLAS_OP_N,
    seq_len, seq_len, head_dim,
    &alpha,
    K, head_dim, stride_q,   // 每个 batch slice 起点跳 B*stride？实际按 layout 设 stride
    Q, head_dim, stride_q,
    &beta_zero,
    S, seq_len, stride_s,
    batch_size * num_heads);
```

*Shape 注意事项*：

- *Leading dimension (lda/ldb/ldc)* 是 row-major 下*一行*的字节跨度对应的元素数，不是 tensor 总长度。$Q$ 的 lda $= d$（$(N, d)$ 连续行）。
- *Batch count* $= B times H$ 时，每个 batch item 是独立 $(N, d) times (N, d)^T arrow.r (N, N)$，*无 cross-batch 归约*。
- *Causal mask* 不在 GEMM 里——cuBLAS 算完 $Q K^T$ 后还要 bias / mask kernel，或像源码一样在 scores kernel 里分支。FlashAttention 把 mask 融进 tile 循环（第 8 章）。

#note[
  Row-major 下 $Q[i, :] dot K[j, :]$ 等价 GEMM 的 $C[i, j] = sum_d Q[i,d] K[j,d]$。$K$ 在内存里是 $(N, d)$ row-major，GEMM 里常对 $K$ 做 `OP_T`。面试手写 launch 时，*先画 $Q, K$ 的 shape，再对应 cuBLAS 的 trans 标志*。
]

=== Scale 放在哪？

源码在 scores kernel *写回前*乘 `kScale`（$1/sqrt(d)$）。三种等价位置：

#figure(
  table(
    columns: (auto, 1fr, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*位置*], [*做法*], [*备注*],
    [QK 乘后], [`acc * kScale` 写 $S$], [源码路径；与 cuBLAS alpha 一致],
    [softmax 前], [读 $S$ 时乘 scale], [多一次读或 fuse 进 softmax kernel],
    [融进 $Q$], [launch 前 $Q *= 1/sqrt(d)$], [改变 $Q$；multi-layer 里 $Q$ 还要给别的 op 用时不方便],
  ),
  caption: [*Table:* scaled dot-product attention 中 $1/sqrt(d)$ 缩放因子的三种等价放置位置。数值上 FP32 下 ULP 差可忽略；工程取舍在于是否多一次 global 读写或修改 $Q$ 供后续 op 复用。],
  kind: table,
)

*Observation*：三处放置数学等价，但 QK epilogue 或 softmax 入口 fuse scale 最常见——少一次 global write/read；融进 $Q$ 会改变 $Q$ 本体，multi-layer 里 $Q$ 还要给别的 op 用时不如在 scores 或 cuBLAS `alpha` 里一次性缩放。

#insight[
  数值上三者等价（FP32 下 ULP 差可忽略）。工程上 *QK epilogue 或 softmax 入口 fuse scale* 最常见——少一次 global write/read。Naive 三 pass 若在 scores 里 scale，softmax 读到的已是 scaled logits，与 PyTorch `scaled_dot_product_attention` 默认行为一致。
]

=== Causal mask 在 scores 阶段

Decoder self-attention：query 位置 $i$ 只能 attend key $j <= i$。源码在 `col > row` 时写 `-1e30f`（近似 $-oo$），softmax 后权重 $approx 0$。

#warn[
  不要用 `0` 代替 $-oo$ mask——0 会参与 max，当其他 logits 都是负数时 max 变成 0，softmax 分布错误。PyTorch 用 `masked_fill(..., -inf)` 再 softmax，与此一致。
]

Padding mask：变长 batch 里无效 token 的 key 列同样置 $-oo$（或 skip 不参与 max/sum，见 softmax 章 `softmax_masked_kernel`）。Causal + padding 可叠加：无效位或 future 位都 excluded。

=== Padding mask 实现（与 scores 解耦 vs 融合）

*路径 A — scores 阶段加 bias*（与 causal 同类）：

```cpp
if (!padding_mask[col]) {
  scores[row * seq_len + col] = -1e30f;
  return;
}
```

*路径 B — softmax 阶段 skip*（第 3 章 `softmax_masked_kernel`）：mask=0 的列不参与 online 的 max/sum 更新，写回时置 0。好处：causal 已写 $-oo$ 的 future 位不必再读 mask；padding 位单独处理。

*路径 C — 加性 attention bias*：$S[i,j] += "bias"[i,j]$，ALiBi 把相对位置编码成每行不同的斜率，不 materialize 完整 $N times N$ bias 矩阵时可只传 slope 参数——那是 fused kernel 的事。

#insight[
  Mask 的面试要点：*excluded 位置不能参与 softmax 归一化分母*。写 $-oo$ 等价于 exp 后权重 0；skip 等价于不参与 max/sum。Padding 全 0 行（无有效 token）要特判，否则 $m=-oo, sum=0$ 除零（第 3 章 warn 框）。
]

== v2: Step 2 — Row-wise Softmax

源码 `attention_softmax_kernel`：一行一个 block，$N$ 个 thread 协作（$N <=$ block size 的教学假设）。

```cpp
__global__ void attention_softmax_kernel(float* scores) {
  const int row = blockIdx.x;
  if (row >= kSeqLen) return;

  __shared__ float shared[kSeqLen];
  const int col = threadIdx.x;
  shared[col] = scores[row * kSeqLen + col];
  __syncthreads();

  float row_max = shared[0];
  for (int i = 1; i < kSeqLen; ++i) {
    row_max = fmaxf(row_max, shared[i]);
  }

  float row_sum = 0.0f;
  for (int i = 0; i < kSeqLen; ++i) {
    shared[i] = expf(shared[i] - row_max);
    row_sum += shared[i];
  }

  scores[row * kSeqLen + col] = shared[col] / row_sum;
}
```

Launch：`<<<kSeqLen, kSeqLen>>>`。In-place 把 $S$ 变成 $P$——仍占 $N^2$ global memory，只是语义从 logits 变成概率。

=== 与第 3 章的衔接

生产环境 $N = 4096$ 时不能用 `__shared__ float shared[4096]`（16 KB 尚可，32K 则 128 KB 超 SM 限额）。应换第 3 章的 `softmax_block_kernel`（256 threads + grid-stride 扫列 + tree reduction）或 online softmax。

Naive attention 的 softmax 步 *单独占一次 kernel launch + 整表 HBM 读写*。第 3 章 online merge 在这里还*用不上*——因为整行 $S$ 已经在 global memory 里，没有「分 tile 流式」的压力。Online 的价值在 *不 materialize 整行* 的 Flash-Attention（第 8 章）。

=== 大 $N$ 时 softmax kernel 的 smem 陷阱

教学 kernel 用 `__shared__ float shared[kSeqLen]`，要求 `blockDim.x >= N` 且 smem $= N times 4 "B"$。$N = 4096$ → 16 KB smem，尚可；$N = 32768$ → 128 KB，*超过默认 48 KB/SM 限额*，kernel 无法 launch。

生产路径必须换第 3 章 `softmax_block_kernel`：`<<<B H N, 256>>>`，256 threads grid-stride 扫 $N$ 列 + tree reduction。Attention 的 rows 数 $= B H N$（每个 query token 一行 logits），$B H N$ 很大时 grid 足够；但每行 $N$ 也 huge，softmax 自身 dram traffic $= 2 B H N^2 times 4 "B"$（读 $S$ + 写 $P$）——与 scores/value 叠加，*带宽主导*。

#note[
  Sum 用 float 累加在 $N$ 很大时可能丢精度；第 3 章 online 版用 `double` 累加 `row_sum`。Attention logits 动态范围大时，softmax 数值稳定性（subtract-max）比 attention 特有逻辑更关键——复习第 3 章 subtract-max 推导。
]

== v3: Step 3 — $O = P V$

源码 `attention_value_kernel`：每个 query 行一个 block，每个输出维度 $d$ 一个 thread。

```cpp
__global__ void attention_value_kernel(
    const float* probs, const float* v, float* out) {
  const int row = blockIdx.x;
  const int d = threadIdx.x;
  if (row >= kSeqLen || d >= kHeadDim) return;

  float acc = 0.0f;
  for (int col = 0; col < kSeqLen; ++col) {
    acc += probs[row * kSeqLen + col] * v[col * kHeadDim + d];
  }
  out[row * kHeadDim + d] = acc;
}
```

Launch：`<<<kSeqLen, kHeadDim>>>`。这是第二个 batched GEMM：$P (N times N) times V (N times d) arrow.r O (N times d)$。

=== GEMM 视角与 launch

```cpp
// O = P * V,  P: N×N, V: N×d, O: N×d
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            d, N, N,
            &alpha, V, d, P, N, &beta, O, d);
// 注意 cuBLAS column-major：参数顺序与 row-major 思维要对调
```

*Shape 易错点*：

- $P$ 的行 = query index，列 = key index；$V$ 的行 = key index（与 $P$ 的列对齐）。
- $O[i, d]$ 是 query $i$ 的输出向量第 $d$ 维，等于 $sum_j P[i, j] dot V[j, d]$。
- Batched 时 $P, V, O$ 在 $(b, h)$ slice 上各是一块连续 $(N, N)$ / $(N, d)$。

#insight[
  两个 matmul 的 naive GPU 实现（scores / value）都是 *memory-bound、零 tile 复用*——和第 4 章 matmul naive 同一类问题。工业界 $Q K^T$ 和 $P V$ 走 cuBLAS/tensor core；瓶颈往往在 $S/P$ 的 materialize，不在 matmul 本身。
]

=== ncu 实测（naive 三 kernel chain）

#ncu-snapshot(
  version: "naive (scores → softmax → value)",
  size: [seq=256, head_dim=64（batch=1, head=1）],
  rows: (
    ("scores_kernel",   "48.8 µs",  "$Q K^T$，$O(N^2 d)$ ops"),
    ("softmax_kernel",  "32.6 µs",  "row-wise softmax on $N times N$ 矩阵"),
    ("value_kernel",    "20.6 µs",  "$P V$，$O(N^2 d)$ ops"),
    ("*Total (3 launches)*", "*~102 µs*", "含 3 次 launch overhead + HBM 中间 tensor 往返"),
    ("scores Memory SOL",  "71.2 %",   "$Q$、$K$ 从 HBM 读，$S$ 写回 HBM"),
    ("softmax Memory SOL", "32.7 %",   "$S$ 读一次，$P$ 写回一次"),
    ("value Memory SOL",   "10.3 %",   "*非常低*——$V$ 每 row 被多个 output col 复用"),
    ("Achieved Occupancy", "7-23 %",   "小 seq 时都撑不满 GPU"),
  ),
)

*三个 kernel 加起来的开销*和它们对应的 workload 有关：

- *scores kernel* 是 $O(N^2 d)$ 的 GEMM——`memSOL 71%` 说明 HBM-bound（跟第 4 章 naive matmul 同构，$V$ 未被复用）。
- *softmax kernel* 处理 $N times N$ 的 `S` 矩阵：`memSOL 32.7%`——比 03_softmax 的 12% 高，因为 seq=256 让 grid 更大，SM 塞得更满。
- *value kernel `memSOL 10.3%`* 是 attention naive 的最大痛点——$V$ 应该被 $N$ 个 query 复用，但 naive 每个 output 独立读 $V$ 的一整列。

*中间 tensor $S$、$P$ 各是 $N times N times 4 = 256$ KB*（seq=256），一路走 HBM 三次。这个 $O(N^2)$ 显存 + 带宽开销随 seq 平方增长——是 FlashAttention 出现的根本动因。

#verdict(
  problem: [三个 kernel 之间通过 HBM 传递 $O(N^2)$ 中间 tensor $S$、$P$；value kernel 又没有 tile 化 $V$ 的复用],
  evidence: [total 102 μs，其中 $S/P$ 在 HBM 里往返约 1 MB（$2 times 512$ KB）；value memSOL 10.3% 说明大量 memory pipeline 时间在等 $V$ 的重复读；seq $= 4096$ 时 $S$ 会变成 64 MB，直接 OOM],
  next: [v2 (tiled) 把三步融进一个 kernel——把 $S$ / $P$ tile 化后存 shared memory，用 online softmax 做增量合并，让 $S$ 永远不落 HBM]
)

== Host 侧：三次 Launch 与中间缓冲

源码 `run_naive_attention` 的数据流：

```cpp
cudaMalloc(&device_scores, kSeqLen * kSeqLen * sizeof(float));

attention_scores_kernel<<<kSeqLen, kSeqLen>>>(device_q, device_k, device_scores, causal);
attention_softmax_kernel<<<kSeqLen, kSeqLen>>>(device_scores);
attention_value_kernel<<<kSeqLen, kHeadDim>>>(device_scores, device_v, device_out);
```

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left, left),
    [*Step*], [*Kernel*], [*Grid*], [*Global 读写*],
    [1], [`attention_scores_kernel`], [`(N, 1, 1)` × `(N, 1, 1)`], [读 $Q,K$；写 $S$],
    [2], [`attention_softmax_kernel`], [`(N, 1, 1)` × `(N, 1, 1)`], [读+写 $S arrow.r P$],
    [3], [`attention_value_kernel`], [`(N, 1, 1)` × `(d, 1, 1)`], [读 $P,V$；写 $O$],
  ),
  caption: [*Table:* naive attention 三 pass 的 kernel、launch 配置（grid × block）与每步 global memory 读写模式。Step 1 写 $S$；Step 2 in-place 将 $S$ 变为 $P$；Step 3 读 $P,V$ 写 $O$。],
  kind: table,
)

*Observation*：三步各一次完整 HBM round-trip，中间 $S/P$ 占 $N^2$ global buffer——$N=8$ 时每 pass ~3 μs、时间≈launch 固定成本；$N=4096$ 时 matmul/softmax body 进毫秒级、$S/P$ traffic 按 $N^2$ 放大，三步 launch 仍可忽略但 HBM 读写成为主导项。

*三次 launch*：第 1 章测 ~5 μs 量级固定成本。本章 $N = 8$ 实测每 pass 总时间仅 3–4 μs——*kernel body 可忽略，时间 ≈ launch*（见 `=== 实测`）。$N$ 大到单次 matmul/softmax 进毫秒级时 launch 可忽略；小 batch 推理 × 多层 × 三 pass 仍会堆成 measurable latency。Fused attention 的动机之一是一次 kernel 搞定（第 8 章）。

`device_scores` 分配：$B times H times N^2 times 4$ B。反向传播还需存 $P$ 或 recomputation 策略——naive 训练通常 *checkpoint $P$*，显存再翻倍。

=== 训练反向：为什么 naive 更惨

Forward 三 pass 已 materialize $S/P$。Backward 经典路径（简化）：

1. $d V = P^T d O$ — 又一个 $(N, N) times (N, d)$ GEMM。
2. $d P = d O V^T$ — $(N, d) times (d, N) arrow.r (N, N)$。
3. $d S = "softmax_backward"(d P)$ — 读 $P$ 和 $S$，写 $d S$。
4. $d Q = d S K$, $d K = d S^T Q$ — 两个 batched GEMM。

*峰值显存*：forward 的 $P$ + backward 的 $d S$ + 激活 $Q,K,V$ — 仍含 $O(N^2)$ 项。FlashAttention 用 recomputation（反向时不存 $P$，按 tile 重算）把显存降到 $O(N)$——第 8 章。

#warn[
  面试问「attention backward 瓶颈」：forward 的 $N^2$ 内存已够致命；backward 再多 1–2 次 $N^2$ 量级的 GEMM 和 softmax backward，*带宽和显存双杀*。答 recomputation + fused backward 是标准解法。
]

== Multi-Head 并行：head 进 grid.y

单 head 源码用 `blockIdx.x = row`。推广到 $(B, H)$：

```cpp
const int b = blockIdx.z;
const int h = blockIdx.y;
const int row = blockIdx.x;
const int head_offset = ((b * num_heads + h) * seq_len) * head_dim;
const float* q_ptr = q + head_offset;
// scores 偏移: ((b * num_heads + h) * seq_len + row) * seq_len + col
```

Launch 示例：

```cpp
dim3 grid(seq_len, num_heads, batch_size);
dim3 block(seq_len, 1, 1);  // scores / softmax
attention_scores_kernel<<<grid, block>>>(...);
```

#insight[
  *Head 维度是零同步成本的并行轴*：不同 head 读写不同的 $S$ slice，无 atomic、无 shared memory 跨 head。Grid 在 y/z 上扩到 $B times H$，SM 利用率随 batch × heads 线性涨——LLM 推理 batch=1 时 heads=32 仍可能填不满 GPU，要靠 sequence 或 layer 并行。
]

$Q, K, V$ 的 layout 决定 offset 公式；$(B, H, N, d)$ 下同一 head 内 $Q[i, :]$ 连续，$K[j, :]$ 在 scores kernel 里对不同 `col` 是*跨行*访问——与 matmul 章 naive $B$ 列访问类似，*scores 的 thread-per-element 写法在大 $N$ 上不合并*。生产 $Q K^T$ 必走 tiled GEMM。

=== 完整 multi-head launch 骨架

```cpp
// 假设 layout (B, H, N, d)，row-major
__global__ void attention_scores_mha_kernel(
    const float* q, const float* k, float* scores,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool causal) {
  const int b = blockIdx.z;
  const int h = blockIdx.y;
  const int row = blockIdx.x;
  const int col = threadIdx.x;
  if (b >= batch_size || h >= num_heads || row >= seq_len || col >= seq_len) return;

  const int head_elems = seq_len * head_dim;
  const int batch_stride = num_heads * head_elems;
  const int head_stride = head_elems;
  const float* q_head = q + b * batch_stride + h * head_stride;
  const float* k_head = k + b * batch_stride + h * head_stride;

  const int score_stride = seq_len * seq_len;
  const int score_batch_stride = num_heads * score_stride;
  float* s_head = scores + b * score_batch_stride + h * score_stride;

  // ... 同单 head dot product，写 s_head[row * seq_len + col]
}

dim3 grid(seq_len, num_heads, batch_size);
dim3 block(seq_len, 1, 1);
attention_scores_mha_kernel<<<grid, block>>>(...);
```

Softmax 与 value 步同理：`blockIdx.y = h`, `blockIdx.z = b`，scores 指针偏移 $((b H + h) N^2)$。Grid 三维 $(N, H, B)$ 总 block 数 $B H N$——LLM 典型 $8 times 32 times 4096 approx 10^6$ blocks，CUDA 可调度，但 *每 block 仅 $N$ thread 做 softmax* 在大 $N$ 上仍不 optimal。

#note[
  Inference KV cache：解码 step 只新增 1 个 query token，$N_q = 1$，$N_k =$ 已缓存长度。Naive 仍分配 $(1, N_k)$ 的 score 行（或可省略 full matrix 只算一行——那是另一个优化点）。*长上下文 inference* 的瓶颈从「方阵 $N^2$」变成「每 step 读全长 $K,V$ cache」——本章 $O(N^2)$ 分析仍适用 score 行长度 $N_k$ 增长。
]

== Masked Attention 小结

#figure(
  table(
    columns: (auto, 1fr, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*Mask 类型*], [*实现*], [*典型场景*],
    [Causal], [`col > row` → $-oo$ 或 softmax 只扫 `col <= row`], [GPT decoder],
    [Padding], [无效 key → $-oo$ 或 masked softmax skip], [BERT batch 变长],
    [Custom], [bias 加在 $S$ 上再 softmax], [ALiBi、局部 window],
  ),
  caption: [*Table:* attention 中三类 mask 的实现方式与典型部署场景。Causal 排除 future keys；Padding 排除无效 token 的 key 列；Custom 通过加性 bias 修改 logits 再 softmax。],
  kind: table,
)

*Observation*：三类 mask 的共同要求是 excluded 位置不能参与 softmax 归一化分母——写 $-oo$ 与 skip 等价；Causal 与 Padding 可叠加，但 padding 全 0 行需特判防 $m=-oo, sum=0$ 除零（第 3 章 warn）。Fused kernel 应在 online 循环内应用 mask，避免单独 mask launch。

Causal 可在 scores 写 $-oo$，或在 softmax 只累加允许列（第 3 章 `softmax_causal_kernel`）。Padding 常在 softmax 用 mask tensor（0/1），*不参与 max/sum*。Fused kernel 里 mask 应在 online 循环内应用，避免单独 mask kernel（第 3 章 v6）。

== Naive 的三宗罪

=== (a) $O(N^2)$ 内存：长序列 OOM

上文内存表：$N=32K$ 单 head $S$ 约 4 GB。Materialize $S$（或 $P$）是 *架构级* 问题，不是「再优化 2× 带宽」能解决的。

=== (b) 三次 kernel launch

Scores → Softmax → Value 各一次同步。Transformer 每层 × 每个 forward；小 batch 推理时 launch 占比高。CUDA Graph / `torch.compile` 可录制三次 launch 序列，但不减 $S$ 内存。

=== (c) $S$ 矩阵 HBM 读写占满带宽

Traffic 主导项：写 $S$ + 读 $S$ 写 $P$ + 读 $P$ ≈ *3× $N^2$ 量级的 attention map 流量*（in-place softmax 省一次 buffer，不省读）。$Q, K, V$ 相对 $N^2$ 项是 $O(N d)$，$N >> d$ 时可忽略。

*与 FLOPs 的对比*：$N=4096, d=64, B=8, H=32$ 时 matmul FLOPs $approx 2.2$ TFLOP，而 $S/P$ 相关 HBM $approx 68$ GB。即使 GPU 算力无限，仅搬运 $S/P$ 就要 $approx 45$ ms——*内存墙*先于算力墙。优化 matmul 的 tiling 只能减少 $Q,K,V$ 读次数，*无法*去掉 $S$ 的写与 $P$ 的读，除非 fuse（FlashAttention）。

#warn[
  实测「attention 慢」时，先用 `ncu` 看 `dram__bytes` 是否和 $4 N^2 H B$ 同量级。若是，换 tiled GEMM 只能优化 matmul 部分；*要根治必须去掉 $S$ 的 materialize*（Flash-Attention）。
]

运行：`make build/07_attention && ./build/07_attention`。默认 $N = 8, d = 8$，与 CPU reference 对齐，容差 $10^(-4)$。源码同时含 `attention_tiled_kernel`（单 kernel 预告 FA），本章正文聚焦 naive 三 pass。建议实验：把 `kSeqLen` 改为 64 观察 `device_scores` 的 `cudaMalloc` 大小（16 KB → 16 KB 仍小），再用 host 脚本估算 $N=4096$ 时 64 MB/head 的分配——*数字比背公式更有体感*。

== 端到端数据流与常见错误

=== 三步之间的 tensor 生命周期

```text
Q, K, V  (global, 输入)
    │
    ▼  attention_scores_kernel
S = QK^T/√d  (global, N×N)  ← O(N²) 瓶颈
    │
    ▼  attention_softmax_kernel (in-place)
P = softmax(S)  (global, 同一块 buffer)
    │
    ▼  attention_value_kernel
O  (global, N×d 输出)
```

*In-place softmax* 把 $S$ 覆写成 $P$——省一份 $N^2$ 分配，但 forward 若 backward 需要 $P$，仍得在 softmax 后 `cudaMemcpy` 备份或重算。PyTorch autograd 通常*保存 $P$ 或 $S$* 用于 backward，显存不减。

=== 错误 1：Scale 漏乘或乘两次

漏 scale：$d=128$ 时 logits 过大，softmax 极端 one-hot，梯度消失。乘两次（scores 和 softmax 各乘一次 $1/sqrt(d)$）：等价 $1/d$ 缩放，分布过平。单元测试：对比 `torch.nn.functional.scaled_dot_product_attention` 的 max abs error。

=== 错误 2：Causal 与 padding 混淆

Causal 只 mask *future keys*（列 index > 行 index）。Padding mask *无效 token 的 key 列*，与 query 行无关——一个 query 可以 attend 到 batch 内另一序列的有效 key（cross）或同序列有效 key（self + padding）。把 padding 当成 causal 会错 mask 整列。

=== 错误 3：Softmax 在 mask 前做

若先 softmax 再 zero future，分母含 future 的 exp，*有效权重和 $< 1$*。必须在 max/sum 前 exclude masked 位置。

=== 错误 4：$V$ 的行与 $P$ 的列不对齐

$O[i,d] = sum_j P[i,j] V[j,d]$。$P$ 的列 $j$ 必须对应 $V$ 的第 $j$ 个 token 向量。Layout $(B,H,N,d)$ 下 $V[j,d]$ 的偏移是 `j * head_dim + d`，不是 `i` 的函数——写反成 $P[i,j] V[i,d]$ 是经典 off-by-one 级 bug。

=== 错误 5：忽略 $N^2$ 分配失败

`cudaMalloc(..., N*N*4)` 在 $N=65536$ 时单 head 需 16 GB——`cudaMalloc` 返回 `out of memory` 而非 kernel 算错。长序列第一件事：算 buffer 大小，不是 launch grid。

#warn[
  这五个错误在 interview coding 里出现频率高于「会不会写 online softmax」。能口述数据流 + 指出 $S$ 在哪分配，已经比只会背公式强一档。
]

== 铺垫 Flash-Attention：为什么必须 fuse

第 3 章 online softmax 维护 $(m, d)$（running max + sum），支持 *chunk merge*：

$ d' = d dot exp(m - m') + exp(x - m') $

FlashAttention 对 $K/V$ 按 sequence 分块：每块在 shared memory 算局部 $Q K_"tile"^T$，用 online 公式更新全局 softmax 状态，*同时*累加 $O$ 的分子——*从不写出完整 $S$*。

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*维度*], [*Naive（本章）*], [*Flash-Attention（第 8 章）*],
    [中间 $S$ HBM], [$O(N^2)$ 读写], [$O(N)$（tile 在 smem）],
    [Softmax], [整行 materialize 后算], [online merge 按 tile],
    [Kernel 数], [3+ launches], [1 fused kernel],
    [核心数值工具], [subtract-max（第 3 章）], [online merge + $O$ 重标定],
  ),
  caption: [*Table:* naive 三 pass attention 与 Flash-Attention 在 HBM 流量、softmax 策略、launch 次数与数值工具上的对比。Naive 列对应本章教科书路径；FA 列预告第 8 章 fused 方案。],
  kind: table,
)

*Observation*：四行对比的核心差异是中间 $S$ 的 HBM 复杂度——naive $O(N^2)$ 读写 vs FA $O(N)$ tile 在 smem；online merge（第 3 章公式）是 FA 正确性前提，使分块 softmax 与整行等价。本章 `attention_tiled_kernel` 是机制原型，$N=8$ 上 ~1.8× 加速主要来自少 launch，生产规模 IO 节省按 $N^2$ 放大。

源码里 `attention_tiled_kernel` 是*预告*：单 kernel、sequence 维分 `kTileTokens`、online 更新 `running_max` / `running_sum`——与 naive 三 pass 对照跑通 CPU reference。$N = 8$ 上 fused 版 5.47 μs vs 三 pass 合计 9.92 μs（~1.8×，见 `=== 实测`）——*机制原型*，不是生产加速比。第 8 章会在更大 tile、warp 分工、$Q/K/V$ 双缓冲上展开。

=== ncu 实测（tiled，机制原型）

#ncu-snapshot(
  version: "tiled (fused single kernel)",
  size: [seq=256, head_dim=64],
  rows: (
    ("Duration (1 kernel)",     "135.6 µs", "*比 naive 三 kernel 总和 102 μs 慢！*"),
    ("Memory SOL",              "15.1 %",   ""),
    ("Compute SOL",             "16.8 %",   ""),
    ("Achieved Occupancy",      "7.4 %",    "one-block-per-query 设计"),
    ("Grid Size",               "256",      "$N$ 个 block（一个 query 一个 block）"),
  ),
)

*慢了！*—— 这个反直觉结果需要正视。教学 tiled kernel 有几个明显的次优点：

- *one-block-per-query*：每 block 处理 $N$ 个 key 全走一遍。seq=256, block=256 → 一个 query 一个 block，但 kernel 内是纯粹串行的 outer loop over tiles。没有 warp specialization，也没有 Tensor Core。
- *smem 使用简陋*：`Q_tile[HEAD_DIM] + K_tile[TILE × HEAD_DIM] + V_tile[TILE × HEAD_DIM]`——K/V tile 在 smem 里没有 swizzle，也没有 async load。
- Naive 三 kernel 版反而在 seq=256 时占了便宜：scores kernel 是 `<<<seq, seq>>>`，256×256 threads 有 65536 个 thread 撑满 GPU；tiled 版只有 256 blocks × 256 threads/block（当然还有 block 内的 tile-serial loop 让 SM 一直有活干，但并行度反而低）。

真正 fast attention 在下一章 FlashAttention v1 展开：把 `attention_tiled_kernel` 的骨架换成 warp-per-row 结构 + `cp.async` + Tensor Core。

#final-verdict(
  status: [教学 tiled kernel 展示了 "1 kernel + online softmax + $S$ 不落 HBM" 的核心思想，是 FlashAttention 的最简 prototype。],
  note: [但*机制原型不是性能实现*。seq=256 的小尺寸下，三 pass 的高并行度反而更快；到 seq $= 4096+$ 时 naive 因为 $S = 64$ MB OOM 或严重带宽瓶颈，tiled 的 $O(N)$ smem 就体现价值。真正的性能实现请看第 8 章 FlashAttention v1。]
)

=== Self-attention vs Cross-attention（shape 不变）

*Self-attention*：$Q, K, V$ 来自同一序列，$N_q = N_k = N$。Causal mask 只出现在 decoder self 路径。

*Cross-attention*（encoder-decoder）：$Q$ shape $(B, H, N_q, d)$，$K,V$ shape $(B, H, N_k, d)$。$S$ 变为 $(N_q, N_k)$，内存 $O(N_q N_k)$——通常 $N_k << N_q$（encoder 短）或二者同阶。Naive 三 pass *结构不变*，只改 launch 的 $N$ 为 $N_q$ / $N_k$，scores kernel 里 `row < N_q`, `col < N_k`。

#insight[
  面试常把 self / cross 混在一个问题里：*算法相同，shape 不同*。Cross 没有 causal（除非自定义 mask）；内存瓶颈仍是 $S$ 的 $N_q times N_k$。
]

=== 生产栈对照

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*组件*], [*Naive（本章）*], [*生产（PyTorch / FA2）*],
    [$Q K^T$], [scores kernel / 朴素 GEMM], [cuBLAS / CUTLASS / Triton tile],
    [Softmax], [独立 kernel + 整表读写], [online + fuse 或 FA 内嵌],
    [$P V$], [value kernel], [GEMM 或 FA 内累积 $O$],
    [$S$ 缓冲], [$B H N^2$ FP32], [不分配或 checkpoint],
    [典型 latency], [$O(N^2)$ HBM 主导], [$O(N^2)$ FLOPs 但 $O(N)$ IO],
  ),
  caption: [*Table:* naive attention 与生产栈（PyTorch / FlashAttention-2）在各计算组件上的实现对照。Naive 列 materialize $S/P$；生产路径用 cuBLAS/tensor core GEMM 并将 softmax 与 $P V$ fuse 进 FA kernel，$S$ 缓冲不分配或 checkpoint。],
  kind: table,
)

*Observation*：生产路径在 $Q K^T$ 与 $P V$ 上走 cuBLAS/tensor core，但 FA 的真正优势是最后一行——典型 latency 从 $O(N^2)$ HBM 主导变为 $O(N^2)$ FLOPs 配 $O(N)$ IO；面试答「FA 快在哪」应先说 IO，再说 launch 与 SRAM 复用，online softmax 是正确性工具而非唯一原因。

#insight[
  面试答「FlashAttention 快在哪」：*第一* IO——$O(N^2) arrow.r O(N)$ HBM traffic；*第二* 融合减少 launch；*第三* 更好的 SRAM 复用。Online softmax 是*正确性*前提，不是唯一原因。
]

== 实测

$N = 8, d = 8$（源码 `kSeqLen` / `kHeadDim`；$Q, K, V$ 各 256 B，`device_scores` 256 B，整例 < 2 KB），A100 80GB SXM4，`ncu --set full` 抓取每个 kernel 的一次 launch。表中 TC % = `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`；warp % = `sm__warps_active.avg.pct_of_peak_sustained_elapsed`。

Launch 配置决定 warp lane 利用率——grid 只有 8 block，远填不满 108 个 SM：

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (left, left, left, right, left),
    [*version*], [*grid*], [*block*], [*threads*], [*并行轴*],
    [scores / softmax], [(8, 1, 1)], [(8, 1, 1)], [64], [`blockIdx.x` = query 行，`threadIdx.x` = key 列],
    [value], [(8, 1, 1)], [(8, 1, 1)], [64], [`threadIdx.x` = head 维 $d$],
    [tiled], [(8, 1, 1)], [(8, 1, 1)], [64], [同 value；smem tile + online softmax],
  ),
  caption: [*Table:* ch7 attention 各 kernel 的 launch 配置（`launch__grid_size` / `launch__block_size`）与并行轴划分。scores/softmax 用 `<<<kSeqLen, kSeqLen>>>`（blockDim = $N = 8$）；value/tiled 用 `<<<kSeqLen, kHeadDim>>>`（blockDim = $d = 8$）。],
  kind: table,
)

*Observation*：三 pass 与 tiled 的 block 都只有 8 thread——`issued/32 = 8.0` 是 blockDim 小于 warp 32 的结构性浪费，不是 divergence。grid 仅 8 block 远填不满 108 SM，diag 表 `warp % approx 0.1%` 是 teaching 规模伪影；生产 $N >= 4096$ 时 grid $(N, N)$ 才涨，但 blockDim 仍应 $>= 128$ 以喂饱 warp。

scores / softmax 用 `#raw("<<<kSeqLen, kSeqLen>>>")`（blockDim = $N = 8$）；value / tiled 用 `#raw("<<<kSeqLen, kHeadDim>>>")`（blockDim = $d = 8$）。$N, d < 32$ 时每个 block 凑不满一个 warp——*这是 teaching default 的结构性特征，不是 kernel bug*。

#include "../bench/07_attention.typ"

#warn[
  这一章的问题规模是教学 default（B×S×H ~ 数千个 float），kernel 单次运行只有 3–20 μs。ncu 的定性指标（`issued/32`、`bank conflicts`、`barrier stall`）仍能反映 kernel 结构，但*绝对数字对生产规模不完全可信*：
  - HBM % 会偏低（分母 elapsed time 含冷启动窗口）
  - dram_bytes 可能被 L2 消化，`GB/s (实测/逻辑)` 两列差距明显
  想拿到生产规模的数字，把主参数（rows/cols/hidden dim）加到让工作集远超 L2 (40 MB)。
]

*perf 表读三件事：*

+ *三 pass 各 3.17–3.55 μs，加总 9.92 μs——几乎全是 launch 固定成本*。scores 3.20 + softmax 3.55 + value 3.17 = 9.92 μs；每步只做 $8 times 8$ 量级 FFMA / exp，算力与 HBM 都未参与竞争。

+ *tiled 5.47 μs vs 三 pass 9.92 μs（~1.8×）*。收益拆开：少 2 次 launch（在本规模上 ~4–6 μs 量级）；不写 global $S$（naive 仍 materialize $8 times 8$ scores 并 in-place softmax）。fused 单次仍比*单步* scores（3.20 μs）慢——smem + online 循环有 intrinsic 成本；只有端到端三 pass 累加才输。

+ *HBM % / TC % / warp % 全表 0.0–0.1%*。TC % = 0：全 FP32 scalar FFMA，无 WMMA。warp % = 0.1%：8 block × 8 thread，108 SM 上几乎空转。*不能*用 HBM % 判断 attention 是否 memory-bound——前文 $N = 4096$ 手算的 68 GB traffic 在本 micro-benchmark 上测不出来。

#figure(
  hbar-chart(
    (
      ("value", 3.17),
      ("scores", 3.20),
      ("softmax", 3.55),
      ("tiled (fused)", 5.47),
      ("naive 三 pass Σ", 9.92),
    ),
    unit: "μs",
  ),
  caption: [`time (μs)`：三 pass 加总 ~2× 于 fused——主因 launch 累加；单 kernel 时间都在 ~3–5 μs noise 带内。],
)

*diag 表读关键教学点：*

*a) 三 pass `issued/32 = 8.0`——blockDim = 8 造成的结构性 lane 浪费*

scores / softmax / value 的 block 都只有 8 个 thread。硬件按 warp（32 lane）粒度发射指令，*24 个 lane 每拍空转*。`issued/32 = 8.0` 正是 $8/32 times 32$ 的定量读数——*不是* warp divergence（不同 lane 走不同 basic block），而是 launch 配置让 warp 从未填满。

#figure(
  warp-lanes(active: range(8), cell: 0.34,
             title: "scores / softmax / value：blockDim = 8，仅 lane 0–7 有活干"),
  caption: [绿色 = 负责一个 key 列或 head 维的 thread。灰色 = 占 issue slot 但无对应输出——*结构性 idle*，不是 predication 能藏起来的浪费。]
)

`pred_on/32` = 7.3–7.9，issued − pred_on ≈ 0.1–0.7：causal mask 的 `if (!causal || col <= row)` 和边界 guard 编译成 predicated instruction，predicated-off lane 藏在 gap 里——*仍不是 divergence*，因为 `issued/32` 稳定在 8.0 而非「有时 32 有时 8」。

*b) tiled `issued/32 = 2.8`，`pred_on/32 = 2.7`——比三 pass 更低，仍是 teaching 规模 artifact*

tiled block 同样只有 8 thread，但 kernel 内 `kTileTokens = 4` 的 tile 循环、causal 分支、online softmax 重标定让*同一 warp 里不同 phase 只有部分 lane 同时有活*——平均参与 issued 指令的 lane 数降到 2.8。*这不是实现写错了*：$N = 8, d = 8$ 时根本没有足够并行度让 32 lane 饱和；生产 FA 用 $d >= 64$、tile 32×32、多 warp 协作后 `issued/32` 会回到 28–32。

#figure(
  warp-grid(
    rows: 8, cols: 8,
    cell: 0.30,
    active: (
      (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
      (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
      (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7),
      (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7),
      (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7),
      (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7),
      (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7),
      (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7),
    ),
    row-labels: ("Q0", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"),
    col-labels: ("K0", "K1", "K2", "K3", "K4", "K5", "K6", "K7"),
    title: "scores kernel：$S = Q K^T$ 的 $8 times 8$ tile（每 block 一行）",
  ),
  caption: [
    绿色 cell $(i, j)$ = block $i$ 的 thread $j$ 算 $S[i,j] = q_i dot k_j$。
    一行 8 个 thread 仍只填满 warp 的 1/4——矩阵再小也改不了 blockDim $< 32$ 的 lane 浪费。
  ],
)

*c) `smem conf. = 0`；softmax `barrier stall = 0.12`，tiled `0.23`*。softmax 32 B smem + 一次 `__syncthreads`；tiled 340 B smem + tile 循环内多次 sync——*不能*从源码推断 bank conflict，metric 说没有。`mem stall` 3.97–9.90：kernel 太短，long_scoreboard 读数无 memory-bound 诊断价值。

*d) regs / smem fingerprint*：scores / value 26 regs、0 smem；softmax 28 regs、32 B；tiled 31 regs、340 B——fused 把 $S/P$ 留在 smem，不写 global，代价是更大 smem footprint（生产 FA 会碰 48 KB/SM 限额，第 8 章）。

*无信息或为零的 metric：*

- `TC %`：全表 0.0——无 tensor pipe，符合 scalar FMA 源码。
- `HBM %`：全表 0.1%——< 2 KB 工作集 L2 resident。
- `warp %`：全表 0.1%——8 block 填不满 GPU，*不是 kernel 设计错了*。

#insight[
  低 `issued/32`（8.0 或 2.8）在本章*首先*说明 teaching shape 太小、blockDim $< 32$——不要误读成「attention kernel 有严重 divergence」。Flash-Attention 在生产规模靠大 tile + 多 warp 把 lane 利用率拉满；本章 micro-benchmark 的价值是 launch 累加与 fuse 省 $S$ 的*机制对照*，不是 occupancy 饱和下的绝对加速比。
]

#insight[
  ~1.8× fused 加速*不是* tensor core 或 occupancy 的胜利——warp / TC / HBM % 全部 ~0.1%。它正是 FA 在生产规模能做到 10×+ 的*机制原型*：fuse 消除 launch + 消除 $O(N^2)$ intermediate 的 HBM traffic。$N = 4096$ 时 launch 占比下降，IO 节省按 $N^2$ 放大。
]

#warn[
  不要把本章 ~1.8× 外推到生产 attention。长序列主瓶颈是 $O(N^2)$ HBM（手算节 68 GB），不是 3× launch；decode 小 batch 时 launch 累加仍是真实 latency 来源。低 `issued/32` 在本规模是 teaching 限制——放大 $N, d$ 后先查 grid/block 是否喂饱 SM，再谈算法。
]

== ncu 该看什么

```
ncu --set full --section SpeedOfLight ./build/07_attention
```

关键 metric（对照本章 $N = 8$ 实测）：

- `gpu__time_duration.sum`：scores 3.20 μs / softmax 3.55 μs / value 3.17 μs / tiled 5.47 μs——*端到端 naive 用三数相加*（9.92 μs），不是单 kernel 对比。
- `smsp__thread_inst_executed_per_inst_executed.ratio`（issued/32）：三 pass 8.0（blockDim = 8）；tiled 2.8——*warp lane utilization*，低值在本规模主因 block 未满 warp，不是 divergence。
- `smsp__average_thread_inst_executed_pred_on_per_inst_executed.ratio`（pred_on/32）：7.3–7.9（三 pass），2.7（tiled）——issued − pred_on 是 predicated-off lane，不是分支 divergence。
- `sm__warps_active.avg.pct_of_peak_sustained_elapsed`（表中 warp %）：全部 0.1%——*并行度不足*，和 softmax 章 naive 单 block 同一类问题；$N$ 大时 grid $(N, N)$ 才会涨。
- `sm__pipe_tensor_cycles_active`（表中 TC %）：0——FP32 scalar，无 WMMA。
- `dram__bytes.sum.pct_of_peak_sustained_elapsed`：0.1%——L2 全命中；*不能*用它判断 attention 是否 memory-bound。增大 $N$ 到远超 L2 后再看 block vs tiled 的 DRAM 差。
- `launch__registers_per_thread` / `launch__shared_mem_per_block_static`：tiled 31 regs + 340 B vs softmax 28 + 32 B——验证 fused 路径的 smem footprint。
- `smsp__sass_thread_inst_executed_op_ffma_pred_on.sum`：scores/value ~512（$8 times 8 times d/2$ 量级）；softmax ~2368（exp 循环主导）；tiled ~2064（三阶段合一）。

=== 分 pass 的预期 metric（$N=4096$, $d=64$, A100——理论，非本章实测）

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*Kernel*], [*dram 主导*], [*其他信号*],
    [scores], [$2 B H N^2 d$ 读 QK + 写 $S$], [FFMA 低 util；grid 极大],
    [softmax], [$2 B H N^2$ 读+写 $S/P$], [MUFU (exp) 占比高],
    [value], [$B H N^2 d$ 读 $P,V$ + 写 $O$], [同 scores 类 memory-bound],
  ),
  caption: [*Table:* $N=4096$, $d=64$, $B=8$, $H=32$ 时 naive attention 三 pass 各 kernel 的预期 DRAM 主导项与其他 ncu 信号（理论估算，非本章 $N=8$ 实测）。dram 列给出 traffic 量级公式；softmax 步 MUFU（exp）占比高。],
  kind: table,
)

*Observation*：三 pass 合计 dram 访问量级 $O(B H N^2)$——与手算节 $approx 68$ GB（$B=8,H=32,N=4096$）一致；scores 与 value 的 traffic 含 $N^2 d$ 因子（读 QK / 读 $P,V$），softmax 纯 $N^2$ 读写且 exp 主导 MUFU。FA 目标是把同一公式降到 $O(B H N d^2)$，$N/d = 64$ 时差约 64×——本章 $N=8$ 实测验证 fuse 机制，不体现这一 IO 差距。

== 面试白板 code

面试官说"手写 scaled-dot-product attention"——先给最直白的三步版本，讲清楚接口和瓶颈，再用一句话过渡到 flash-attention（下一章）。

```cpp
// Single-head causal attention. Q, K, V: [N, D], out: [N, D]. Row-major.
// 三 kernel 版本：QK^T → softmax → PV. 中间存 S 和 P 是 O(N^2) 内存.
// 面试白板到这一步就够——然后马上说"这就是为什么要 flash-attention".

// Kernel 1: S = Q @ K^T * scale. S: [N, N].
__global__ void attn_scores(const float* Q, const float* K, float* S,
                            int N, int D, float scale, bool causal) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;  // query row
  int j = blockIdx.x * blockDim.x + threadIdx.x;  // key row
  if (i >= N || j >= N) return;

  if (causal && j > i) { S[i * N + j] = -INFINITY; return; }

  float acc = 0.f;
  for (int d = 0; d < D; ++d) acc += Q[i * D + d] * K[j * D + d];  // <q_i, k_j>
  S[i * N + j] = acc * scale;  // scale = 1 / sqrt(D)
}

// Kernel 2: P = softmax(S), row-wise. 用 ch3 online softmax 骨架, 每 block 一 row.
// __global__ void softmax_online(const float* S, float* P, int N)  ← 见 ch3.

// Kernel 3: O = P @ V. 普通 GEMM. [N, N] @ [N, D] → [N, D].
__global__ void attn_out(const float* P, const float* V, float* O,
                         int N, int D) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;  // out row
  int d = blockIdx.x * blockDim.x + threadIdx.x;  // out col
  if (i >= N || d >= D) return;
  float acc = 0.f;
  for (int j = 0; j < N; ++j) acc += P[i * N + j] * V[j * D + d];
  O[i * D + d] = acc;
}

// ==== Launch config ====
// Kernel 1 (attn_scores): 每 thread 算 S 里的一个元素 S[i, j].
//   blockDim = (32, 8) = 256 threads——tx 沿 j (key 维) 保证 K^T 读 coalesced;
//   gridDim  = ((N+31)/32, (N+7)/8).  S 是 N×N 网格.
dim3 block1(32, 8);
dim3 grid1 ((N + 31) / 32, (N + 7) / 8);
attn_scores<<<grid1, block1>>>(Q, K, S, N, D, 1.f / sqrtf((float)D), causal);

// Kernel 2 (softmax_online): 每 block 一 row，同 ch3.
//   gridDim = N (行数); blockDim = 256 (N 中等时).
softmax_online<<<N, 256>>>(S, P, N);

// Kernel 3 (attn_out): 每 thread 算 O 里的一个元素 O[i, d].
//   blockDim = (D, 8) 或 (32, 8)，D=64/128 时 D 维正好一 warp;
//   gridDim  = ((D+bx-1)/bx, (N+7)/8).
dim3 block3(min(D, 32), 8);
dim3 grid3 ((D + block3.x - 1) / block3.x, (N + 7) / 8);
attn_out<<<grid3, block3>>>(P, V, O, N, D);

// Multi-head 时: gridDim.z = num_heads * batch, 每个 head/batch 独立走这三 kernel.
// 生产实现会把这三 kernel 都换成 tiled GEMM (K 维 tile) 或直接跳到 flash-attention.
```

*核心考点*（追问顺序，最后一定引到 flash-attention）：

- *"内存复杂度？"* → $S$ 和 $P$ 都是 $N times N$，$N = 8192$ 时约 256 MB（fp32）。这就是 naive attention 显存爆炸的原因。
- *"为什么 memory-bound？"* → 三 kernel 之间要把 $O(N^2)$ 的中间张量写回 HBM 又读回来。AI = $O(N D)$ FLOP / $O(N^2)$ bytes = $O(D/N)$——$N > D$ 时越来越差。
- *"causal mask 怎么实现？"* → 在 QK^T 时 `if (j > i) S[i,j] = -inf`，softmax 后自动为 0。生产实现里 mask 融合到 softmax 一起写、避免物化 mask 矩阵。
- *"multi-head 怎么加？"* → 加一维 `head`，$Q, K, V$ 形状变 $[N, H, D_h]$，把 head 当外层循环 / grid.z 维；每 head 独立算。
- *"如何 fuse 成一个 kernel？"* → 关键是把 $S$ 和 $P$ 都不写回 gmem——但 softmax 需要看完整一 row 才能算 max/sum、$V$ 的 GEMM 又要看完整 $P$。解决办法：*tile K/V，online softmax rescale*——这就是 flash-attention 的核心 idea，下一章展开。
- *"三个 kernel 的 launch config 有什么共性？"* → 都用 2D block $(x, y)$：`threadIdx.x` 映射到 stride-1 内存维度（$K^T$ 的 key 索引、$V$ 的 head_dim）保证 coalesced；`threadIdx.y` 是 batch/row 维、独立无冲突。这个"2D block、tx 沿最内维"是所有 tile-based CUDA kernel 的通用套路，跟 ch4 matmul 一致。

== 面试考点

#interview[
  *Q1*: Scaled dot-product attention 公式？张量 shape？

  A: $ "softmax"(Q K^T / sqrt(d)) V$。$Q,K,V$ 为 $(B, H, N, d)$；$S,P$ 为 $(B, H, N, N)$；$O$ 为 $(B, H, N, d)$。FLOPs $O(B H N^2 d)$，中间矩阵内存 $O(B H N^2)$。
]

#interview[
  *Q2*: Naive attention 为什么长序列不可行？

  A: Materialize $S/P$ 需 $O(N^2)$ HBM；$N=32K$ 单 head 数 GB。且 softmax 与两次 matmul 对 $S/P$ 多次读写，带宽 $O(N^2)$。FLOPs 可接受，内存和 IO 不可接受。
]

#interview[
  *Q3*: Causal mask 怎么实现？

  A: Scores 阶段对 `col > row` 写 $-oo$（或 softmax 只处理 `col <= row`）。不能用 0 代替 $-oo$。与 PyTorch `masked_fill(-inf)` 等价。
]

#interview[
  *Q4*: Multi-head 怎么并行？

  A: Head（及 batch）无 cross-head 依赖；grid 扩 `blockIdx.y = h`, `blockIdx.z = b`，每 head 独立 $S$ slice。注意 Q/K/V 的 memory layout 算 offset。
]

#interview[
  *Q5*: Scale $1/sqrt(d)$ 放哪？数值考量？

  A: QK 乘后、softmax 前、或融进 $Q$ 数学等价。工程常 fuse 进 GEMM epilogue 或 softmax 入口。目的是控制 logits 方差，避免 softmax 饱和；与 mask 的 $-oo$ 不冲突。
]

#interview[
  *Q6*: Attention 算术强度？memory-bound 还是 compute-bound？

  A: 按 $N^2$ 中间态摊销 AI $approx d/4$ FLOP/B；$d=64$ 接近 ridge 但 naive 实际偏 memory-bound（$S/P$ 读写、exp、三次 launch）。*瓶颈首先是 $O(N^2)$ 内存与带宽*，其次才是 matmul AI。
]

#interview[
  *Q7*: Naive 三步分别对应什么 GEMM / softmax？

  A: (1) $S = Q K^T$ batched GEMM $(N,d) times (d,N)$；(2) 行 softmax $(B H N, N)$；(3) $O = P V$ batched GEMM $(N,N) times (N,d)$。Shape 对齐：$P$ 列 index = $V$ 行 index。
]

#interview[
  *Q8*: Fused / Flash attention 的动机？

  A: 不 materialize $S$——$O(N^2) arrow.r O(N)$ IO；online softmax（第 3 章）支持 tile 合并；单 kernel 减 launch；smem 复用 $Q/K/V$ tile。Naive 三 pass 是正确性 baseline，不是生产路径。
]

#interview[
  *Q9*: Padding mask 与 causal mask 能同时用吗？

  A: 能。无效 token 与 future 位置都应在 softmax 支持集外（$-oo$ 或 skip）。实现可在 scores 相加 bias，或在 fused online 循环里分支。
]

#interview[
  *Q10*: 为什么 softmax 在 attention 里必须 subtract-max？

  A: Logits 幅度随 $d$ 和输入尺度变大，`exp` 溢出。减行 max 后等价且稳定（第 3 章证明）。Causal 行若全 $-oo$ 需特判防除零。
]

#interview[
  *Q11*: Inference KV cache 下 attention 内存还是 $O(N^2)$ 吗？

  A: 单步 decode 只算 1 个 query，score 是长度 $N_k$ 的*向量*而非方阵——中间态 $O(N_k)$。但每层仍要读全长 $K,V$ cache（$O(N_k d)$ per layer per step），长上下文瓶颈变成 cache 带宽。Prefill 阶段仍是 $O(N^2)$ materialize $S$ 的问题。
]

#interview[
  *Q12*: Grouped-query attention (GQA) 改变 naive 三 pass 吗？

  A: 算法不变；$K,V$ 的 head 数 $H_"kv" < H$ 时，$Q K^T$ 里 $K$ 需 broadcast/repeat 到 query head。内存从 $O(B H N^2)$ 略降（$K,V$ 变小），但 $S/P$ 仍是 $O(B H N^2)$——*不解决 $N^2$ 瓶颈*，只减 $K,V$ 参数与 cache。
]
