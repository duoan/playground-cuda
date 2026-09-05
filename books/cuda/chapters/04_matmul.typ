#import "../template.typ": *

= Matmul (GEMM)

matrix multiply 是 GPU kernel 优化的*终极试金石*。vector add 的 naive 版本已经能打满带宽；reduction 靠 shuffle 和 warp 协作就能逼近峰值；但 GEMM 从 naive 到 cuBLAS / CUTLASS 级别，中间隔着整整一条优化 ladder——shared memory tiling、register tile、向量化 load、warp 分工、bank conflict swizzle、`cp.async` pipeline、tensor core `mma.sync`……每一层都对应一个可量化的瓶颈和一套可面试的追问。

这一章是全书中*最长、最难*的一章。我们要把它讲透：

- GEMM 的数学定义、存储布局、以及为什么它是 compute-bound 的经典例子。
- Roofline 分析：AI 如何从 naive 的 $O(1)$ 随 tile 增大而爬升。
- 从 cuBLAS baseline → naive miscoalesced → coalesced → SMEM tile → 1D/2D register tile → float4 vectorized → MMA PTX → WMMA 的 9 级 ladder（对应源码 K0–K8）。
- 源码里没写全但面试必问的：`cp.async` 多 stage pipeline、bank conflict swizzle、CUTLASS 4 层 tile 分解、split-K、Hopper WGMMA/TMA。
- 怎么选 tile 大小、怎么用 ncu 诊断、cuBLAS 为什么快。

对应源码：`src/cuda/04_matmul/{00_cublas,01_naive_miscoalesced,02_naive_coalesced,03_smem_tiled,04_block_tile_1d,05_block_tile_2d,06_vectorized,07_mma_ptx,08_wmma}.cu`。

== 问题定义

给定两个矩阵 $A in RR^(M times K)$ 和 $B in RR^(K times N)$，计算：

$ C[i, j] = sum_(ell=0)^(K-1) A[i, ell] dot B[ell, j], quad 0 <= i < M, quad 0 <= j < N $

等价地，$C = A B$。这是 BLAS Level-3 里的 *GEMM*（General Matrix Multiply），深度学习里 90% 以上的算力最终都落在这个操作上。

=== 存储布局

本书和源码都用 *row-major* 布局：

```cpp
// A: M×K,  A[row * k + col]
// B: K×N,  B[row * n + col]
// C: M×N,  C[row * n + col]
```

#note[
  cuBLAS 默认 *column-major*（Fortran 传统）。调用 `cublasSgemm` 时参数顺序和转置标志要对应好。面试里常问 "你的 kernel 是 row-major，cuBLAS 怎么对齐"——答案通常是：要么在 host 侧交换 $A/B$ 并加 transpose flag，要么写 kernel 时就按 column-major 思维组织 tile。
]

=== 计算量

输出 $M times N$ 个元素，每个元素 $K$ 次乘加 = $2K$ FLOP（一次乘法 + 一次加法各算 1 FLOP）。

$ "Total FLOPs" = 2 M N K $

=== 内存访问量（naive 视角）

每个输出 $C[i,j]$ 需要读 $A$ 的一行（$K$ 个 float）和 $B$ 的一列（$K$ 个 float），写 1 个 float。

如果不做任何复用，总读取量 $approx 2 M N K times 4 "B"$，写入 $M N times 4 "B"$。

#insight[
  GEMM 的核心优化目标：*让同一份 $A/B$ 数据被尽可能多的 thread 复用*，从而把有效 AI 从 $O(1)$ 抬到 $O("tile size")$。所有后续技巧——shared memory tile、register tile、tensor core——都是这条思路的延伸。
]

== Roofline：从 memory-bound 到 compute-bound

=== Naive 的 AI

每算 1 个输出元素（1 次完整点积）：

- 读 $K$ 个 $A$ + $K$ 个 $B$ = $2K times 4 = 8K$ 字节
- 写 1 个 $C$ = 4 字节
- 计算：$2K$ FLOP

$ "AI"_"naive" = frac(2K "FLOP", 8K + 4 "B") approx frac(1 "FLOP", 4 "B") = 0.25 "FLOP/B" $

A100 FP32 ridge point $approx 13 "FLOP/B"$。0.25 比 ridge 低两个数量级——*极度 memory-bound*。

=== Tiled 的 AI

一个 block 用 $B_M times B_K$ 的 $A$ tile 和 $B_K times B_N$ 的 $B$ tile，协作算出 $B_M times B_N$ 个输出。

- 读 $A$ tile：$B_M B_K times 4$ B（从 global，每个 K-slab 读一次）
- 读 $B$ tile：$B_K B_N times 4$ B
- 写 $C$ tile：$B_M B_N times 4$ B（只写一次）
- 计算：$2 B_M B_N B_K$ FLOP（沿 $K$ 方向累加 $B_K$ 次，重复 $K / B_K$ 个 slab）

沿整个 $K$ 维度累加后，每个输出元素的*摊销*读取量从 $8K$ B 降到 $approx 4(B_M + B_N)$ B（当 $B_K$ 在 $K$ 方向滑动时，每个 slab 的数据被整个 block 复用）。

有效 AI（忽略写回）：

$ "AI"_"tiled" approx frac(2 B_M B_N B_K, 4(B_M B_K + B_K B_N)) = frac(B_M B_N, 2(B_M + B_N)) "FLOP/B" $

例：$B_M = B_N = B_K = 128$ → AI $approx 32 "FLOP/B"$——*已经超过 ridge point*，进入 compute-bound 区间。

#insight[
  Tile 越大，AI 越高——但 tile 越大，shared memory / register 占用越多，occupancy 越低。GEMM 优化的本质是在 *AI ↔ occupancy* 之间找 sweet spot。这就是为什么 cuBLAS / CUTLASS 要花大量篇幅 auto-tuning tile 参数。
]

=== 性能 ladder 概览

本章 ladder 有 9 个版本 (K0–K8)，都在 A100 SXM4-80GB 上、$M = N = K = 2048$ 实测：

#ladder(
  ("K0 cuBLAS",             "vendor lib, TF32 tensor core",       "83.2 TFLOPS  (100%)"),
  ("K1 naive miscoalesced", "row/col 映射反了",                    "0.47 TFLOPS  (0.6%)"),
  ("K2 naive coalesced",    "swap tid.x/tid.y",                   "2.58 TFLOPS  (3.1%)"),
  ("K3 SMEM tiled",         "16×16 smem tile",                    "3.85 TFLOPS  (4.6%)"),
  ("K4 block tile 1D",      "每 thread 8×1 register 列",           "7.84 TFLOPS  (9.4%)"),
  ("K5 block tile 2D",      "每 thread 2×2 register 块",           "7.67 TFLOPS  (9.2%)"),
  ("K6 vectorized",         "float4 load + 4×4 register tile",    "11.7 TFLOPS  (14%)"),
  ("K7 MMA PTX",            "mma.sync m16n8k16, FP16→FP32",       "14.6 TFLOPS  (18%)"),
  ("K8 WMMA",               "nvcuda::wmma fragment, FP16→FP32",   "19.1 TFLOPS  (23%)"),
)

#warn[
  这是*真·实测*，不是估算。K1→K2 78×、K2→K3 1.7×、K3→K4 2.0×、K6→K7 1.24×、K7→K8 1.31×——每一级都能在 ncu 里看到直接原因。K8 的 19 TFLOPS 打满了 A100 的 FP32 CUDA-core 峰值 (19.5 TFLOPS)，但离 TF32 tensor core (156 TFLOPS) 还差 8×、离 FP16 tensor core (312 TFLOPS) 差 16×——这个 gap 来自本章*没写*的东西：更大的 warp tile、`cp.async` pipeline、swizzled smem、CUTLASS-式 warp specialisation。这些在 Hopper 上被 WGMMA + TMA 打包成 K9–K12（章末讨论，本章 A100 编不了）。
]

#note[
  K0 cuBLAS 是一个特殊的 baseline：它不是"下一版"，而是*天花板*。K1 起是我们从最坏的写法开始爬。K4/K5 的 TFLOPS 几乎相同（一维 vs 二维 register tile）——这说明*ladder 上不是每一级都必然提速*，把两者都放进来是为了让你在 ncu 里看清 "为什么 2D 不再赢"。
]

=== 手算一遍：tile 如何把 AI 从 0.25 拉到 32

以 $M = N = K = 4096$，$B_M = B_N = B_K = 128$ 为例。

*Naive*：每个 $C[i,j]$ 独立读 $2K = 8192$ B，AI $approx 0.25 "FLOP/B"$。

*Tiled*：一个 block 128×128 thread（或等价的 thread 映射）协作：

- 沿 $K$ 方向走 $4096 / 128 = 32$ 个 slab。
- 每个 slab：从 global 读 $128 times 128 times 4 times 2 = 128 "KB"$ 的 $A/B$ tile（各 64 KB）。
- 每个 slab 在 smem 里产出 $128 times 128$ 个部分和，每个元素做 $128 times 2 = 256$ FLOP。
- 32 个 slab 合计 FLOP：$2 times 128^3 times 32 = 2^{24}$ FLOP（和 naive 一样，数学不变）。
- 但从 global 读的数据量：$32 times 128 "KB" = 4 "MB"$（每个 $A/B$ 元素只从 global 读一次），摊到 $128^2$ 个输出上，每输出 $approx 256$ B 读取 → AI $approx 128^2 / (2 times 128) = 32 "FLOP/B"$。

#note[
  上面忽略了写回 $C$、忽略了 smem 和 register 的读写——那些通常比 global 便宜一个数量级。面试手算时，用这个近似公式 $ "AI" approx B_M B_N / (2(B_M + B_N)) $ 足够说明 tile 的价值。
]

=== GEMM 在深度学习里的位置

Linear layer：`Y = X @ W^T + b`，本质是 GEMM。Attention 里的 $Q K^T$ 和 $ "softmax" @ V $ 也是 GEMM（或 batched GEMM）。Optimizer 里的 weight update 同样。可以说：*把 GEMM 优化到极致，就覆盖了 LLM 推理/训练的大头算力*。FlashAttention 的 clever 之处在于它*不是*标准 GEMM——通过 online softmax 避免了 materialize 完整的 $S times S$ 矩阵——但 $Q K^T$ 和 $P V$ 的子问题仍然是 GEMM 结构。

== K0: cuBLAS baseline

先看天花板。cuBLAS 在 A100 上跑 FP32 GEMM 会自动 dispatch 到基于 CUTLASS 的 TF32 tensor core kernel——即使你传的是 FP32 输入，只要 math mode 打开就走 tensor core。

```cpp
static cublasHandle_t handle = nullptr;
if (!handle) {
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
}
const float alpha = 1.f, beta = 0.f;
// 关键：cuBLAS 是 column-major。行主序的 A @ B 用 (A B)^T = B^T A^T 变换成
//       cublasSgemm(handle, N, N, n, m, k, α, B, n, A, k, β, C, n)
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            n, m, k,
            &alpha,
            b, n,   // 行主序 B 在 column-major 里的 leading dim
            a, k,
            &beta,
            c, n);
```

*不要试着从这份代码里学 GEMM 结构*——cuBLAS 内部一个 kernel 可能有几百 KB PTX、按 shape 动态选算法（persistent、split-K、stream-K）、warp specialise、TMA 分层……我们只把它当成"实测天花板"来看。

#ncu-snapshot(
  version: "K0 cuBLAS",
  size: [$M = N = K = 2048$],
  rows: (
    ("Duration",             "206 µs",  "17.2 GFLOP / 0.207 ms = 83.2 TFLOPS"),
    ("Compute (SM) SOL",     "65.9 %",  "TF32 tensor pipe 忙碌"),
    ("Memory SOL",           "46.9 %",  "HBM 也在跑，但不占主导"),
    ("L2 Hit Rate",          "81.1 %",  "cuBLAS 会重排 grid 让相邻 CTA 共享 L2 line"),
    ("Registers / thread",   "230",     "巨大——production kernel 疯狂占寄存器换 ILP"),
    ("Achieved Occupancy",   "6.2 %",   "低到反直觉：230 reg × 128 thread ≈ 30k reg / SM，一个 block 就吃掉大半 SM"),
  ),
)

*两个反直觉的读数*：

- *Occupancy 只有 6%——却是全表最快*。这打破了"高 occupancy = 高性能"的直觉。cuBLAS 用大量寄存器保存 warp 内的 accumulator tile（可能是 $128 times 128$ / warp 级），代价是每 SM 只能塞 1–2 个 block。这些少数 warp 通过 register-level 的 ILP + tensor core 打满 SM 的算力——不需要多 warp 靠切换掩盖 latency。
- *Memory SOL 47% + Compute SOL 66%*：production GEMM 的典型形状——同时在算和搬，没有一头空。tile 够大让 HBM 传的每个 byte 都被复用几百次；同时 warp 数够多让 tensor pipe 不饿。

#verdict(
  problem: [这是天花板，不是"要修的问题"。它的存在只为回答一个问题：*我们手写的能到 cuBLAS 的多少 %？*],
  evidence: [83.2 TFLOPS / 156 TFLOPS TF32 peak $approx$ 53%。cuBLAS 内部也没打满硬件，因为 GEMM 在 shape=2048 上还没到"大到 grid 完全稳态"的规模（108 SM × 82 waves 才能"warm up"）。],
  next: [从 ladder 的最底部开始爬。K1 故意写错，让你直接看到 coalescing 的 78×代价。]
)

== K1: naive miscoalesced（故意错的起点）

```cpp
// row/col 的映射反了：threadIdx.x 沿行走，threadIdx.y 沿列走
__global__ void matmul_kernel(const float* a, const float* b, float* c,
                              int m, int n, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;   // <-- 反了
    const int col = blockIdx.y * blockDim.y + threadIdx.y;   // <-- 反了
    if (row >= m || col >= n) return;
    float acc = 0.f;
    for (int inner = 0; inner < k; ++inner)
        acc += a[row * k + inner] * b[inner * n + col];
    c[row * n + col] = acc;
}
```

*看似正确*——每个 thread 算一个 $C[i, j]$，数学没错。但 warp 内 `threadIdx.x = 0..31` 的 32 个 lane，`col` 是同一个值（因为 col 来自 threadIdx.y），`row` 才不同。它们同时访问 `b[k, col]`——32 个 lane 读*同一个* col 的 32 个不同行。

*后果*：一个 warp 对 B 的读变成 32 次独立 transaction（stride $N$ 访问），而不是 1 次 128 B coalesced。写回 `c[row, col]` 同理。所有 lane 都会串行走 32 个 memory cycle。

#ncu-snapshot(
  version: "K1 miscoalesced",
  size: [$M = N = K = 2048$],
  rows: (
    ("Duration",             "36.9 ms",  "比 K0 慢 179×"),
    ("Memory SOL",           "99.2 %",   "HBM 打满了——但传的都是 *strided 垃圾*"),
    ("Compute (SM) SOL",     "11.7 %",   "SM 大多数时间在等 memory"),
    ("L1/TEX Hit Rate",      "97.1 %",   "反直觉：hit rate 极高！"),
    ("Memory Throughput",    "1.52 GB/s","HBM 峰值 2 TB/s，实际吞吐差 1300×"),
    ("Achieved Occupancy",   "97.8 %",   "warp 是满的，但都在等 load 完成"),
  ),
)

*关键读数解释*：

- *Memory SOL 99% 但吞吐 1.5 GB/s*：SOL 是"发出去的 transaction 占硬件峰值的比例"，跟"传了多少 byte"是两码事。miscoalesced 访问让每个 128 B transaction 里只有 4 B 是"想要的"（1/32 效率），剩下 124 B 被 GPU 传上来又扔掉。
- *L1 hit 97%*：这是最反直觉的一条——多个连续的 warp 对 B 同一列的重叠访问，让 L1 cache 意外 hit 得很好，但 hit 之后每次还是要发一次 request 才能拿到需要的 4 B。cache 帮了忙，但没解决"发太多 request"这个根本问题。

#verdict(
  problem: [坐标映射把*快变化 dim*（threadIdx.x）放到了*慢变化维度*（row），warp 内每个 lane 访问不同 128 B cache line。],
  evidence: [Memory SOL 99% + 实际吞吐 1.52 GB/s（HBM 峰值 2 TB/s 的 0.07%）——SM 忙于发 32× 冗余 transaction。],
  next: [swap `threadIdx.x` ↔ `threadIdx.y`，让 lane 沿 col 走。一行代码。]
)

== K2: naive coalesced

```cpp
__global__ void matmul_kernel(const float* a, const float* b, float* c,
                              int m, int n, int k) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;   // 换回来
    const int col = blockIdx.x * blockDim.x + threadIdx.x;   // 换回来
    if (row >= m || col >= n) return;
    float acc = 0.f;
    for (int inner = 0; inner < k; ++inner)
        acc += a[row * k + inner] * b[inner * n + col];
    c[row * n + col] = acc;
}
```

*一行改动*（其实是两行的 swap）。现在 warp 内 32 lane 的 `col` 依次递增、`row` 相同。它们访问 `b[k, col..col+31]`—— 32 个连续 float = 128 B，一次 coalesced transaction。

#ncu-snapshot(
  version: "K2 coalesced",
  size: [$M = N = K = 2048$],
  rows: (
    ("Duration",             "6.66 ms",  "比 K1 快 5.5×"),
    ("Memory SOL",           "97.6 %",   "HBM 还是打满，但这次每 byte 都有用"),
    ("Compute (SM) SOL",     "65.0 %",   "从 11% 跳到 65%——SM 真在算了"),
    ("Memory Throughput",    "8.24 GB/s","仍然远低于 HBM 峰值——见下 verdict"),
    ("L1/TEX Hit Rate",      "87.5 %",   "L1 帮着 dedupe 相邻 warp 的重复 A 读"),
    ("Achieved Occupancy",   "98.3 %",   ""),
  ),
)

*一个访问模式改动 5.5×*。但注意 Duration 还是 6.66 ms、TFLOPS 只有 2.58——离 K0 的 83 TFLOPS 还差 32 倍。原因见 verdict：coalescing 是"访问模式正确"，但仍然*没有复用*。

#verdict(
  problem: [每个 $C[i,j]$ 需要读 $2K = 4096$ 个 float 才算一次结果，*每个 $A$/$B$ 元素只服务 1 个输出*——AI $approx$ 0.25，重度 memory-bound。],
  evidence: [Memory SOL 97.6% 且实际吞吐 8.24 GB/s（比 K1 高 5.4×）——HBM 用足了，但传输量本身仍然是"每 FMA 8 B"。TFLOPS 只有 2.58 / 19.5（CUDA-core FP32 peak）$approx$ 13%。],
  next: [引入 shared memory：一个 block 里 256 个 thread 协作读一次 tile，然后每个 tile 被复用 tile-size 次。AI 从 0.25 → tile-size / 2。]
)

== K3: SMEM tiled

```cpp
constexpr int kTile = 16;
__global__ void matmul_kernel(const float* a, const float* b, float* c,
                              int m, int n, int k) {
    __shared__ float a_tile[kTile][kTile];
    __shared__ float b_tile[kTile][kTile];
    const int row = blockIdx.y * kTile + threadIdx.y;
    const int col = blockIdx.x * kTile + threadIdx.x;
    float acc = 0.f;
    for (int tile_k = 0; tile_k < k; tile_k += kTile) {
        a_tile[threadIdx.y][threadIdx.x] = a[row * k + tile_k + threadIdx.x];
        b_tile[threadIdx.y][threadIdx.x] = b[(tile_k + threadIdx.y) * n + col];
        __syncthreads();
#pragma unroll
        for (int inner = 0; inner < kTile; ++inner)
            acc += a_tile[threadIdx.y][inner] * b_tile[inner][threadIdx.x];
        __syncthreads();
    }
    c[row * n + col] = acc;
}
```

现在每个 block 有 $16 times 16 = 256$ 个 thread，协作把 $A$ 的一行 tile（$16 times 16 = 256$ float）和 $B$ 的一列 tile 搬进 smem。然后每个 thread 用 smem 里的数据做 16 次乘加。K 维度分 $K / 16 = 128$ 个 slab。

*AI 分析*：一个 tile 从 HBM 读 $2 times 16 times 16 = 512$ float = 2048 B，产出 $16 times 16 = 256$ 个中间结果 $times 16$ 次 FMA/结果 = 4096 FMA = 8192 FLOP。$"AI" = 8192 / 2048 = 4$ FLOP/B。*理论上应该跳出 memory-bound*（A100 ridge $approx$ 13 FLOP/B，还差一点，但比 K2 的 0.25 强 16×）。

#ncu-snapshot(
  version: "K3 SMEM tiled",
  size: [$M = N = K = 2048$],
  rows: (
    ("Duration",             "4.46 ms",  "比 K2 快 1.5×"),
    ("Memory SOL",           "93.6 %",   "HBM 还在打——tile 只有 16，复用不够"),
    ("Compute (SM) SOL",     "72.8 %",   "SM 更忙了"),
    ("L1/TEX Hit Rate",      "2.5 %",    "为什么这么低？见下"),
    ("L2 Hit Rate",          "98.1 %",   "所有相邻 block 都在读相同 tile"),
    ("Registers / thread",   "32",       "还是很轻——每 thread 只有 1 个 acc"),
  ),
)

*两个问题需要解释*：

- *L1 hit rate 从 K2 的 87.5% 掉到 2.5%*：现在 A 和 B 的 tile 大部分时间是在 shared memory 里读的，走 SMEM 路径*不经过 L1*，所以 L1 counter 分母变了。这不是坏事——真正的问题是 HBM 用量还高。
- *HBM SOL 还有 93.6%*：tile-16 太小。每个 block 从 HBM 读 $2 times 16^2 times 128 "slabs" = 65 "KB"$，只算出 $16 times 16 = 256$ 个输出。摊到每输出 260 B 读——比 K2 的 8 KB 少多了，但离"每输出 20 B 以内"（TF-32 tensor core 需要）还差一大截。

#verdict(
  problem: [Tile 只有 16 × 16，*每个 SMEM 值只被复用 16 次*。有效 AI 只有 4 FLOP/B，仍在 roofline 的 memory-bound 区。],
  evidence: [HBM SOL 93.6% + 每输出 260 B HBM 读取。TFLOPS 3.85 / 19.5 peak = 20%。],
  next: [把 tile 做大——但 tile 越大占的 SMEM 越多。真正的解药是 *register tiling*：让每个 thread 算多个输出，让同一个 SMEM 值服务多个 acc。]
)

== K4: 1D block tile（每 thread 一列 8 输出）

第一次 register-blocking。每个 block 有 $64 times 8 = 512$ 个 thread，覆盖一个 $64 times 64$ 输出 tile；每个 thread 拥有*一列 8 个输出*的 register 累加器。

```cpp
constexpr int kBlockTileM = 64, kBlockTileN = 64, kBlockTileK = 8;
constexpr int kThreadTileM = 8;                        // 每 thread 拥有 8 行
constexpr int kThreadsX = 64;                          // 沿 N 方向 64
constexpr int kThreadsY = kBlockTileM / kThreadTileM;  // 沿 M 方向 8

__global__ void matmul_kernel(const float* a, const float* b, float* c,
                              int m, int n, int k) {
    __shared__ float a_tile[kBlockTileM][kBlockTileK];
    __shared__ float b_tile[kBlockTileK][kBlockTileN];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int col = blockIdx.x * kBlockTileN + tx;
    const int row_base = blockIdx.y * kBlockTileM + ty * kThreadTileM;
    float acc[kThreadTileM] = {};

    for (int tile_k = 0; tile_k < k; tile_k += kBlockTileK) {
        /* cooperative load: 512 thread 各搬一个 float 填 A(64×8) 和 B(8×64) */
        __syncthreads();

#pragma unroll
        for (int inner = 0; inner < kBlockTileK; ++inner) {
            // *关键*：b_frag 从 SMEM 加载 1 次，被 8 个 acc 复用！
            const float b_frag = b_tile[inner][tx];
#pragma unroll
            for (int i = 0; i < kThreadTileM; ++i) {
                acc[i] += a_tile[ty * kThreadTileM + i][inner] * b_frag;
            }
        }
        __syncthreads();
    }
    /* 写回 8 个 acc 到 c[row_base..row_base+7][col] */
}
```

*为什么快*：内层 K 循环里，`b_frag = b_tile[inner][tx]` 从 SMEM 读一次（共 `kBlockTileK = 8` 次 per K-slab），然后被 8 个 FMA 用掉。SMEM load 数量减少 8×，同时 acc 一直待在寄存器里。

#ncu-snapshot(
  version: "K4 block tile 1D",
  size: [$M = N = K = 2048$],
  rows: (
    ("Duration",             "2.19 ms",   "比 K3 快 2.0×，比 K2 快 3.0×"),
    ("Memory SOL",           "73.2 %",    "HBM 使用率*下降*——AI 上来了"),
    ("Compute (SM) SOL",     "50.9 %",    ""),
    ("Registers / thread",   "58",        "从 32 涨到 58——8 个 acc + 索引变量"),
    ("Achieved Occupancy",   "47.7 %",    "reg 用得多，warp 数减半——但更快"),
    ("Memory Throughput",    "34.8 GB/s", "从 K3 的 12 GB/s 涨到 35——tile 更大，需要的 HBM 也更多"),
  ),
)

*两个重要转折*：

- *Occupancy 从 98% 跌到 48%——但更快*。这是 K0 那个反直觉观察的第一次微缩预告：register pressure 换 ILP。8 个 register acc 让编译器在内层 8-FMA 循环里做深度指令调度、pipeline，实际 IPC 比双倍 warp 还高。
- *HBM SOL 从 K3 的 94% 掉到 73%*——不是因为 HBM 变慢，而是 SM 花更多时间在算 FMA、等 HBM 不再是瓶颈。这是"往 compute-bound 移动"的第一次视觉证据。

#verdict(
  problem: [1D register tile 只在 M 方向复用（`b_frag` 服务 8 个 a_frag），*N 方向仍是每 acc 独立 SMEM read*。],
  evidence: [Compute SOL 50.9% 说明 SM 有一半时间在等——K 循环里 SMEM load 还是 8 个 a_frag + 1 个 b_frag = 9 loads per 8 FMA，SMEM bandwidth 卡住。],
  next: [做 2D tile：每 thread 拥有 $2 times 2$ 输出，让 SMEM 值在 M 和 N 两个方向都复用。]
)

== K5: 2D block tile（每 thread 2×2 输出）

```cpp
constexpr int kBlockTileM = 32, kBlockTileN = 32, kBlockTileK = 16;
constexpr int kThreadTileM = 2, kThreadTileN = 2;

for (int inner = 0; inner < kBlockTileK; ++inner) {
    const float a_f0 = a_tile[ty * 2 + 0][inner];
    const float a_f1 = a_tile[ty * 2 + 1][inner];   // A frag: 2 loads
    const float b_f0 = b_tile[inner][tx * 2 + 0];
    const float b_f1 = b_tile[inner][tx * 2 + 1];   // B frag: 2 loads
    acc[0][0] += a_f0 * b_f0;   acc[0][1] += a_f0 * b_f1;
    acc[1][0] += a_f1 * b_f0;   acc[1][1] += a_f1 * b_f1;
}
```

现在 SMEM 每 iteration 读 4 个值（2 A + 2 B），做 4 次 FMA——每 SMEM load 对应 1 个 FMA。相比 K4 的 9 loads / 8 FMA（≈ 1.1 load / FMA），K5 是 1.0——理论上*不该更快*。ncu 数据也确认了：

#ncu-snapshot(
  version: "K5 block tile 2D",
  size: [$M = N = K = 2048$],
  rows: (
    ("Duration",             "2.24 ms",   "*和 K4 几乎相同* (2.19 vs 2.24)"),
    ("Memory SOL",           "95.2 %",    "HBM 又打满了——tile 缩到 32²"),
    ("Compute (SM) SOL",     "49.9 %",    ""),
    ("Registers / thread",   "40",        "比 K4 少（4 acc vs 8 acc）"),
    ("Achieved Occupancy",   "71.5 %",    "占用率更高，但 tile 也小了"),
  ),
)

*一个诚实的教训*：ladder 上"更好"不等于"更快"。K4 和 K5 数字几乎重合，因为：

- K4 tile 64² + 每 thread 8 acc → tile 大、reg 多；
- K5 tile 32² + 每 thread 4 acc → tile 小、reg 少；

两者对 SMEM 传输和寄存器复用的总量*近似相等*，只是分布不同。真正的下一步不是继续拉 register tile，而是*让 SMEM/HBM 每次传更多 byte*——vectorization。

#verdict(
  problem: [2D register tile 结构对了，但 tile 只有 $32 times 32$、K 只有 16——SMEM tile fill 阶段（每次搬 $32 times 16 times 2 = 1024$ float）用的还是 scalar 4-byte load。],
  evidence: [K5 vs K4：Duration 2.24 vs 2.19，占用率 71% vs 48%——结构变得"更健康"了，但吞吐没变。HBM SOL 又回到 95%，说明 tile 太小、reuse 不足。],
  next: [用 `float4`（128 bit LDG）把 HBM→SMEM 和 SMEM→register 的 traffic 都压成 1/4 条指令；同时把 tile 撑到 $64 times 64$、thread tile 撑到 $4 times 4$。]
)

== K6: vectorized loads（float4）

关键改动有三个，*同时*发生：

1. tile 从 $32 times 32$ 撑到 $64 times 64$；
2. 每 thread 从 $2 times 2$ 撑到 $4 times 4$ = 16 个 acc；
3. HBM→SMEM 的 load 用 `float4`（128 bit LDG），SMEM→register 的 frag load 一次读 4 个 float 存 register。

```cpp
constexpr int kBlockTileM = 64, kBlockTileN = 64, kBlockTileK = 8;
constexpr int kThreadTileM = 4, kThreadTileN = 4;
constexpr int kThreadsX = 16, kThreadsY = 16;   // 256 thread / block

// HBM → SMEM，一半 thread 搬 A，另一半搬 B，各发一条 LDG.E.128
if (tid < 128) {
    // 32 lane × 128 bit = 4096 bit = 512 B per warp = 4 个 128 B transaction
    float4 v = *reinterpret_cast<const float4*>(&a[g_row * k + g_col]);
    *reinterpret_cast<float4*>(&a_tile[a_r][a_c]) = v;
} else {
    float4 v = *reinterpret_cast<const float4*>(&b[g_row * n + g_col]);
    *reinterpret_cast<float4*>(&b_tile[b_r][b_c]) = v;
}

// register tile: 4×4 outer product
#pragma unroll
for (int inner = 0; inner < kBlockTileK; ++inner) {
    float a_frag[4], b_frag[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) a_frag[i] = a_tile[ty * 4 + i][inner];
#pragma unroll
    for (int j = 0; j < 4; ++j) b_frag[j] = b_tile[inner][tx * 4 + j];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            acc[i][j] += a_frag[i] * b_frag[j];   // 16 FMA / 8 loads
}
```

现在内层 K 循环里，8 个 SMEM load 服务 16 个 FMA——ratio 2.0（K4 是 0.9，K5 是 1.0）。同时 tile 大到 $64 times 64$，从 HBM 每读入 1 KB 的 A/B tile 能算 $64^2 = 4096$ 个部分和。

#ncu-snapshot(
  version: "K6 vectorized",
  size: [$M = N = K = 2048$],
  rows: (
    ("Duration",             "1.47 ms",   "比 K5 快 1.5×，比 K0 慢 7.1×"),
    ("Compute (SM) SOL",     "74.2 %",    "*第一次 compute SOL 高于 memory SOL*"),
    ("Memory SOL",           "72.2 %",    "HBM 使用率降回 70——AI 明显上来了"),
    ("Registers / thread",   "59",        "16 acc + fragment 变量"),
    ("Achieved Occupancy",   "43.7 %",    "又跌一档——但更快"),
    ("Memory Throughput",    "47.3 GB/s", ""),
  ),
)

*重要读数*：Compute SOL 74% > Memory SOL 72%——GEMM *终于跨过 roofline 的 ridge*，进入 compute-bound 区。这是"scalar CUDA-core GEMM"能达到的顶点：CUDA core 忙于发 FMA，HBM 不再是瓶颈。

*剩下的 gap*：11.7 TFLOPS / 19.5 TFLOPS CUDA-core peak = 60%。剩下的 40% 主要是：
- SMEM bank conflict（$4 times 4$ register tile 的 A frag load 有 stride-1 pattern，容易 conflict）；
- `__syncthreads` 阻塞；
- 没有 `cp.async` async load，SMEM fill 期间 SM 空等。

#verdict(
  problem: [已经把 CUDA-core FP32 GEMM 压到接近极限。要再快，必须绕开 CUDA core 走 tensor core。],
  evidence: [Compute SOL 74%（$approx$ CUDA-core FP32 peak 的 60%），HBM 用量降到 72%——不再是 memory-bound。],
  next: [切换到 FP16 输入 + tensor core（`mma.sync`）。同一个 K 循环从"每 warp 32 个 FFMA"变成"每 warp 1 条 HMMA = 4096 个 FMA"。硬件算力峰值从 19.5 TFLOPS 跳到 312 TFLOPS。]
)

== K7: MMA PTX（tensor core，raw PTX）

*从这里开始，dtype 换到 FP16 输入 / FP32 accumulate*。CUDA core 的 FFMA 换成 tensor core 的 HMMA。

A100 上支持的 FP16→FP32 MMA 指令有多种 shape，我们用 `mma.sync.aligned.m16n8k16`：一个 warp（32 lane）协作，做一次 $16 times 8$ 输出、$K = 16$ 的外积累加。*一个 warp 一条指令 = 16 × 8 × 16 × 2 = 4096 FLOP。*

```cpp
__global__ void matmul_kernel(const __half* a, const __half* b, float* c,
                              int m, int n, int k) {
    // block: 4 warp = 128 thread。2×2 warp 分工，覆盖 32×16 的 C tile。
    __shared__ __half a_tile[32][16];
    __shared__ __half b_tile[16][16];
    /* warp_m = warp_id / 2, warp_n = warp_id % 2 */
    float acc[4] = {};

    for (int tile_k = 0; tile_k < k; tile_k += 16) {
        /* cooperative load __half tiles */
        __syncthreads();

        // ldmatrix：一条指令让 32 lane 协作把 SMEM 里 4 个 8×8 __half 块
        // 加载到每 lane 的 4 个 .b32 寄存器里，layout 正好符合 mma.sync 期望
        uint32_t a_reg[4], b_reg[2];
        asm("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(a_reg[0]), ... : "r"(smem_addr_a));
        asm("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
            : "=r"(b_reg[0]), "=r"(b_reg[1]) : "r"(smem_addr_b));

        // 一条 HMMA = 4096 FLOP
        asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
            : "r"(a_reg[0]), ..., "r"(b_reg[0]), "r"(b_reg[1]));

        __syncthreads();
    }
    /* 按 mma 的 output layout 存回 c——每 lane 拥有 (group, tig)：
       group = laneid/4 拥有第 group 行；tig = laneid%4 拥有第 tig*2 和 tig*2+1 列 */
}
```

#warn[
  `ldmatrix` 和 `mma.sync` 都是 warp-scope 指令——32 lane 必须*同时执行*且*收敛*。任何 divergence（`if`）都会让它 undefined。这是 tensor core 编程和普通 CUDA 最大的心智负担：从"per-thread scalar 思维"切到"per-warp collective 思维"。
]

#ncu-snapshot(
  version: "K7 MMA PTX",
  size: [$M = N = K = 2048$],
  rows: (
    ("Duration",             "1.18 ms",   "比 K6 快 1.24×——但*远* 未达 tensor core 峰值"),
    ("Compute (SM) SOL",     "44.3 %",    "*反而降低了*！tensor pipe 只占用了 44%"),
    ("Memory SOL",           "91.7 %",    "HBM 又打满——tile 太小、reuse 不足"),
    ("Registers / thread",   "48",        ""),
    ("L2 Hit Rate",          "98.7 %",    "cuBLAS 式的 L2 命中"),
    ("Achieved Occupancy",   "60.1 %",    ""),
  ),
)

*核心问题*：14.6 TFLOPS 只是 A100 FP16 tensor core 峰值 (312 TFLOPS) 的 *4.7%*——比 K6 (CUDA core, 11.7 TFLOPS) 只快 25%。原因在 block 结构：

- 每 block 4 warp，输出 tile 只有 $32 times 16$；一个 K-slab 只发 4 条 HMMA 指令，然后就要 `__syncthreads` 等下一批 SMEM 数据。
- 4 条 HMMA / K-slab、每条 100+ 周期——总共 K 循环有 $2048 / 16 = 128$ 个 slab、每 slab 5–6 μs 空转，绝大部分时间在等 SMEM fill。

*这不是 tensor core 的错——是我们用得太"薄"了*。真正的 tensor core kernel 应该让每 warp 拥有 $64 times 64$ 或 $128 times 128$ 的输出 tile，把几十条 HMMA *串起来* 从而摊薄 SMEM fill 开销。这需要*寄存器爆表*和*多 stage pipeline*，超出本章 K7/K8 的示范范围。

#verdict(
  problem: [Per-warp output tile 只有 $16 times 8$，SMEM fill 占了大部分时间。HMMA 快，但 kernel 没让它一直跑。],
  evidence: [Compute SOL 44% + Memory SOL 92%——SM 大多数时间等 HBM。TFLOPS 14.6 / 312 peak = 4.7%。],
  next: [K8 用 nvcuda::wmma API 让每 warp tile 从 $16 times 8$ 升到 $16 times 16$，同时把语法层从 raw PTX 抬到 C++ 模板——但底层其实是同一件事。真正吃掉 tensor core 需要*章末讨论的* CUTLASS 式 stage / warp specialisation。]
)

== K8: WMMA（`nvcuda::wmma` fragment API）

同样的思想，更高抽象层：

```cpp
#include <mma.h>
using namespace nvcuda;

// warp 拥有 16×16 输出（K7 是 16×8），block 2×2 warp = 32×32 C tile
constexpr int kBlockTileM = 32, kBlockTileN = 32, kBlockTileK = 16;

__global__ void matmul_kernel(const __half* a, const __half* b, float* c,
                              int m, int n, int k) {
    __shared__ __half a_tile[kBlockTileM][kBlockTileK];
    __shared__ __half b_tile[kBlockTileK][kBlockTileN];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.f);

    for (int tile_k = 0; tile_k < k; tile_k += kBlockTileK) {
        /* cooperative load __half tiles */
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
        wmma::load_matrix_sync(a_frag, &a_tile[warp_m * 16][0], kBlockTileK);
        wmma::load_matrix_sync(b_frag, &b_tile[0][warp_n * 16], kBlockTileN);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }
    wmma::store_matrix_sync(&c[row_base * n + col_base], c_frag, n,
                            wmma::mem_row_major);
}
```

*和 K7 的区别*：C++ 模板包住了 `ldmatrix` + `mma.sync` + 输出 layout，*底层 PTX 一模一样*。多出来的价值：
- 编译器帮你选 `ldmatrix` 变体、处理 layout，代码可读性高一个量级；
- 编译器还能在 fragment 之间做 register 分配优化；
- fragment $M times N = 16 times 16$，而 K7 的 $16 times 8$——per warp 输出翻倍，占同样 SMEM tile 时 arithmetic 密度也翻倍。

#ncu-snapshot(
  version: "K8 WMMA",
  size: [$M = N = K = 2048$],
  rows: (
    ("Duration",             "901 µs",    "比 K7 快 1.31×"),
    ("Compute (SM) SOL",     "37.1 %",    "还是低——但比 K7 更好用满 tensor pipe"),
    ("Memory SOL",           "86.4 %",    ""),
    ("Registers / thread",   "56",        ""),
    ("L1/TEX Hit Rate",      "14.7 %",    "wmma 的 SMEM staging 用不同的路径"),
    ("Achieved Occupancy",   "52.4 %",    ""),
  ),
)

*19.1 TFLOPS——正好打满 A100 的 CUDA-core FP32 峰值*（19.5 TFLOPS）。但注意：这是"tensor core 干活但性能上限被 CUDA-core-peak 限制"的巧合。真正的天花板：FP16 tensor core = 312 TFLOPS，我们只用了 6.1%。K0 cuBLAS 83 TFLOPS = 27%，也远未打满。

#final-verdict(
  status: [Ladder K0–K8 在 A100 上到此为止。K8 已经用到了 tensor core，语法层清爽，但*没有*：(1) 多 stage `cp.async` pipeline 掩盖 HBM latency；(2) warp specialisation 让 producer/consumer 各司其职；(3) swizzled SMEM 消除 bank conflict；(4) persistent + split-K/stream-K grid 保持 108 SM 稳态。],
  note: [K9–K12 就是把这四件事在 Hopper (H100+) 上用 WGMMA + TMA 硬件包装起来——细节在下一节讨论。A100 上继续爬的正统路线是读 CUTLASS 3.x 源码：https://github.com/NVIDIA/cutlass。]
)

== 综合实测：跨规模 TFLOPS 表

$M = N = K$ 从 128 到 2048 扫一圈：

#figure(
  table(
    columns: 6,
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (left, right, right, right, right, right),
    [*version*], [*n=128*], [*n=256*], [*n=512*], [*n=1024*], [*n=2048*],
    [K0 cuBLAS],              [0.56], [4.46],  [19.9],  [52.5],  [*83.2*],
    [K1 miscoalesced],        [0.20], [0.34],  [0.43],  [0.46],  [0.47],
    [K2 coalesced],           [0.31], [1.26],  [2.03],  [2.50],  [2.58],
    [K3 SMEM tiled],          [0.45], [1.82],  [3.12],  [3.72],  [3.85],
    [K4 block tile 1D],       [0.17], [0.69],  [2.82],  [5.79],  [7.84],
    [K5 block tile 2D],       [0.35], [1.57],  [4.82],  [6.89],  [7.67],
    [K6 vectorized],          [0.18], [0.76],  [3.17],  [7.99],  [11.7],
    [K7 MMA PTX],             [0.49], [2.09],  [8.21],  [13.1],  [14.6],
    [K8 WMMA],                [0.38], [1.64],  [6.71],  [13.7],  [*19.1*],
  ),
  caption: [各版本在不同规模下的 effective TFLOPS。粗体是各版本最高值。],
)

*读三件事*：

+ *小规模 (n=128, 256) 上高 K 反而慢*——K7/K8 tensor core 版本在 n=128 时（1.5 million FLOP）只有几百 GFLOPS，被 kernel launch overhead 淹没；K0 cuBLAS 甚至更差，因为它内部选算法的 heuristics 在小 shape 上做不好选择。*Small-shape GEMM 是 LLM inference 的常见 case，是 cuBLAS 的传统弱项*——所以生产用 CUTLASS / cuBLASLt 更细的 dispatch。

+ *K1 → K2*：不管什么规模，都是 5×–6× 加速。coalescing 是"永远该做"的一步。

+ *K7 → K8 加速比随规模变小*：n=512 时 8.2 → 6.7（K8 甚至更慢），n=2048 时 14.6 → 19.1（K8 更快）。原因：K7 的 tile 更小、grid 更多、L2 命中率高；K8 tile 大但 warp/block 少，只有 n 大到能 warm up L2 时才发挥优势。

== K9–K12: Hopper WGMMA/TMA（概览，非本章代码）

以下内容只讲思想，没有本章代码——A100 (sm_80) 不支持 WGMMA 和 TMA，编都编不了。想跑 K9+ 需要 H100/H200 (sm_90+) 或 Blackwell (sm_100+)。所有描述基于公开的 NVIDIA / CUTLASS 3.x 文档。

=== K9: 基础 WGMMA（$64 times 64 times 64$ tile，128 thread）

Hopper 引入的核心新指令是 *WGMMA* (*Warpgroup* Matrix Multiply-Accumulate)。关键区别：

- WMMA/MMA 是 *warp*-collective（32 lane），一条 `mma.sync` 一个 warp 做 $16 times 8 times 16$ = 4096 FLOP。
- WGMMA 是 *warpgroup*-collective（*4 warp = 128 lane*），一条 `wgmma.mma_async` 128 lane 做 $64 times 64 times 16$ = 128 K FLOP——单指令算力提升 32×。

更重要的是：*wgmma 是 async 的*。发指令后立即返回，硬件排队执行；`wgmma.wait_group` 才等结果。这样一个 warpgroup 可以：
1. 发 wgmma；
2. 立刻发下一批 SMEM load（cp.async 或 TMA）；
3. wgmma 结果需要用时才 wait。

*K9 tile*: $64 times 64 times 64$。Block 只有 128 thread（一个 warpgroup），SMEM 用 32 KB，占用率低但 wgmma 密度极高。

=== K10: 更大 tile（$128 times 128 times 64$）

同样 128 thread，但 warpgroup 里的每 warp 各拥有更多输出 → 每 warp 累加寄存器变成 $32 times 32$。SMEM 用 128 KB（H100 每 SM 有 228 KB），一个 block 独占大半 SM 但 wgmma 数量翻 4 倍。

K10 vs K9：更大 tile → 更高 AI → 更少 HBM 传输 → 更接近 tensor core peak。典型收益：从 40% peak 到 55% peak。

=== K11: async TMA + producer/consumer warp 分工

TMA (Tensor Memory Accelerator) 是 Hopper 上专门做 HBM ↔ SMEM 批量传输的硬件单元。相对 `cp.async`：
- 一条 TMA 指令传一整个 rank-N tensor（多维索引硬件级支持）；
- 完全 async，硬件级 DMA，不占 SM 的 issue slot；
- 支持 multicast（一个 tile 广播到多个 SM 的 SMEM）。

K11 的结构变成：
- warpgroup 0 = *producer*，只发 TMA 指令搬 A/B tile 到 SMEM；
- warpgroup 1 = *consumer*，只发 wgmma 消费 SMEM tile。
- 两者靠 hardware `mbarrier` 同步（Hopper 的 mbarrier 硬件加速）。

这就是 CUTLASS 3.x 的 pingpong / cooperative scheduler 的模型。K11 vs K10：从 55% peak → 75% peak。

=== K12: max tile（$128 times 256 times 64$，3 warpgroup）

Full 版：3 个 warpgroup, 一个 producer + 2 个 consumer 或者 1 consumer + 2 producer 之类的组合。每 CTA 384 thread（12 warp），每 SM 只跑 1 个 CTA 但 SMEM 用满 200 KB+。这样每 CTA 可以维持 4–5 stage 的 pipeline，wgmma 几乎不停。

*生产系数*：CUTLASS 3.x 上 FP16 GEMM 达到 H100 830 TFLOPS （HW peak ~1000 TFLOPS 的 83%）。

#insight[
  从 K9 到 K12 看似 4 级 ladder，其实是*同一个 idea 的 4 个规模档*：让 warp 从 32 lane 抬到 128 lane 后，硬件愿意帮你 async、tile 变大、pipeline 变深、producer/consumer 分工——这几件事互相强化。想读懂需要*把 A100 时代的所有 CUTLASS 概念内化*，再理解 Hopper 硬件怎么把它们变成一条指令。这是我推荐 A100 上练完 K0–K8 之后专门开一章"CUTLASS 3.x 精读"的原因。
]

#warn[
  以上 K9–K12 数字来自 NVIDIA 公开 blog / CUTLASS 3.x example benchmark，没有在本仓库里跑通。想真跑，签 H100 机器，clone CUTLASS 3.x，选 `example/48_hopper_warp_specialized_gemm/` 之类的 target。
]

== Bank conflict 与 swizzle

=== Shared memory bank 结构

A100 的 shared memory 有 32 个 bank，每个 bank 宽度 4 B。地址 `addr` 映射到 `bank = (addr / 4) % 32`。

一个 warp 32 个 lane *同时*访问 smem 时，如果多个 lane 落到同一 bank 的不同 word → *bank conflict* → 硬件串行化，有效带宽 / conflict 数。

=== 经典冲突场景

Tiled kernel 内层循环：

```cpp
for (int inner = 0; inner < 16; ++inner) {
  acc += a_tile[threadIdx.y][inner] * b_tile[inner][threadIdx.x];
}
```

- 读 `a_tile[ty][inner]`：固定 `ty`，`inner` 递增 → 连续地址，无 conflict ✓。
- 读 `b_tile[inner][tx]`：固定 `inner` 时，lane 0 读 `b_tile[inner][0]`，lane 1 读 `b_tile[inner][1]`，... → 32 个 lane 读 32 个连续 word → 32 个不同 bank → 无 conflict ✓。

但如果布局是 `a_tile[inner][ty]`（$K$ 维在行方向）：

- 读 `a_tile[inner][ty]`：固定 `inner` 时，32 个 lane 的 `ty` 从 0 到 31 → 地址间隔 = 行宽 × 4B。若行宽 = 32 → 所有 lane 命中*同一个 bank* → 32-way conflict → smem 带宽降到 1/32。

=== Swizzle 实现

目标：变换 smem 索引，让 warp 的并发访问均匀分布到 32 个 bank。

*XOR swizzle*（CUTLASS 常用）：

```cpp
// 原始列索引 col，swizzle 后:
int swizzled_col = col ^ (row & 7);  // 对 8 的模做 XOR
a_tile[row][swizzled_col] = value;
```

原理：相邻行的同一列偏移不同，打破 stride-32 的 bank 对齐。

*Permute / 128B swizzle*（更通用）：

```cpp
// 把 smem 看成 128B 的 "atom"
// atom 内偏移 XOR row 的低位
int offset = row * stride + col;
int bank_offset = (offset / 128) * 128 + ((offset % 128) ^ ((row & 7) * 16));
```

#insight[
  Swizzle 是 GEMM 面试的高频题：*不是可选优化，是到达 peak 的必要条件*。ncu 里看 `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`——非零就说明有问题。
]

=== 怎么检测

```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./build/04_matmul
```

或用 CUDA shared memory calculator 工具模拟访问模式。

== Async copy：`cp.async`（SM80+）

Ampere 起引入 `cp.async`：thread 发起异步 copy（global → shared），不等完成就继续执行后续指令。配合 `cp.async.wait_group` 和 pipeline 实现真正的 load-compute overlap。

=== 基本用法

```cpp
#include <cuda_pipeline.h>  // 或 inline asm

__device__ void cp_async4(void* smem_ptr, const void* gmem_ptr) {
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "cp.async.cg.shared.global [%0], [%1], 16;"
    :: "r"(smem), "l"(gmem_ptr));
}

// pipeline 双缓冲
constexpr int kStages = 2;
__shared__ float a_smem[kStages][BM][BK];

// prologue
cp_async4(&a_smem[0][ty][tx], &a[global_addr]);
cp_async_commit_group();

for (int tile_k = 0; tile_k < k; tile_k += BK) {
  const int compute_stage = tile_k / BK % kStages;
  const int load_stage = (tile_k / BK + 1) % kStages;

  cp_async_wait_group(kStages - 1);  // 等 compute_stage 的数据就绪
  __syncthreads();

  // compute on a_smem[compute_stage]
  ...

  if (tile_k + BK < k) {
    cp_async4(&a_smem[load_stage][ty][tx], &a[next_global_addr]);
    cp_async_commit_group();
  }
}
```

=== 正确 pipeline 的要点

1. *Prologue*：先发起 `kStages-1` 个 async copy，再进入主循环。
2. *Wait before compute*：`cp.async.wait_group(N)` 保证当前 stage 数据到达 smem。
3. *Sync before overwrite*：`__syncthreads()` 保证所有 thread 算完当前 stage，才能发起覆写。
4. *Epilogue*：最后一轮只 compute，不 load。

#warn[
  `cp.async` 的 smem 地址必须是 16B 对齐。`cp.async.cg` 绕过 L1 cache 直达 smem——GEMM 通常想要这个行为（数据不复用 L1）。如果用 `ca` 变体，数据会缓存在 L1，可能和后续访问冲突。
]

== Tensor Core 深入：shape 语义与精度

K7/K8 已经落地了 `mma.sync.aligned.m16n8k16` 和 `nvcuda::wmma`。这一节补 shape 表和精度选项——面试常问、代码里没写全的部分。

=== Ampere (sm_80) 支持的 MMA shape

不同 shape 硬件效率不同——`m16n8k16` 是常用的 FP16 shape，但每个 dtype 都有一套自己的 shape 集：

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (left, left, left, left),
    [*shape*], [*dtype (A, B → D)*], [*一条指令 FLOP*], [*用途*],
    [m16n8k8],  [f16, f16 → f32],   [2048],   [小 tile, warp specialisation],
    [m16n8k16], [f16, f16 → f32],   [4096],   [标准 (K7/K8)],
    [m16n8k16], [bf16, bf16 → f32], [4096],   [训练场景，动态范围大],
    [m16n8k8],  [tf32, tf32 → f32], [1024],   [FP32-flavored 训练],
    [m16n8k32], [s8, s8 → s32],     [8192],   [INT8 推理],
    [m16n8k256],[u1, u1 → s32],     [65536],  [BNN 极限——一般用不到],
  ),
  caption: [Ampere MMA shape × dtype 表（部分）。完整表见 PTX ISA §mma.]
)

*选 shape 的思路*：K 越大，单指令 FLOP 越多，摊薄 issue overhead 越好；但太大的 K 也意味着更多 register pressure。K7 选 `m16n8k16` 是 FP16 里能拿到的最大 K（$K = 8$ 版本用于更小的 warp specialisation kernel）。

=== FP32 路径

A100 上"FP32 GEMM"有三条路：

+ *TF32*（`mma...f32.tf32.tf32.f32`）：TF32 是"截断到 10 mantissa bit 的 FP32"，硬件峰值 156 TFLOPS（vs FP16 的 312）。相对 FP32 精度损失 $approx 5 times 10^(-4)$，训练里通常够用。K0 cuBLAS 走的就是这条。
+ *FP16 / BF16 tensor core + FP32 accum*（混合精度）：训练里最常用。BF16 有和 FP32 同样的指数位，动态范围好；FP16 mantissa 精度稍高但范围小、易溢出。
+ *纯 FP32 CUDA-core FFMA*：K1–K6 走的路径。峰值 19.5 TFLOPS，完全没有 tensor core 加持。

#insight[
  面试说 "我用 tensor core 加速了 GEMM"，要能马上讲清：*数据类型*（FP16 / BF16 / TF32 / INT8）、*shape*（m16n8k16 之类）、*accum 精度*（FP16 累加会精度爆炸，训练 GEMM 必须 FP32 accum）。
]

== CUTLASS 的分层思路

NVIDIA CUTLASS 把 GEMM 拆成层次化的 tile 结构——和本章 ladder 一一对应：

```
GemmShape<128, 256, 16>   // CTA tile: 128×256 output, K=16 per step
  └── WarpShape<64, 64, 16>   // 4 warps, 每个 64×64
        └── InstructionShape<16, 8, 16>  // mma.sync m16n8k16
              └── ThreadShape  // 每个 thread 的 fragment 份额
```

*各层职责*：

#figure(
  table(
    columns: (auto, 1fr, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, left),
    [*层级*], [*做什么*], [*对应本章*],
    [CTA (block) tile], [决定 smem 大小、grid 划分],       [K3 SMEM tiled],
    [Warp tile],         [warp 间分工、warp-level mma],    [K8 WMMA 里的 `warp_m/warp_n`],
    [Thread tile],       [register acc、数据复用],        [K4 / K5 / K6],
    [Instruction],       [`mma.sync` / `ldmatrix`],       [K7 (PTX) / K8 (wmma API)],
    [Epilogue],          [alpha/beta scaling、bias、activation], [未覆盖（见 MLP 章）],
  ),
  caption: [*Table:* CUTLASS hierarchical GEMM 的五层分解，及本书 ladder 各版本对应的层级。从上到下粒度递减：CTA 决定"这个 block 输出哪一片"，warp 决定"warp 内 32 lane 如何分工做小 mma"，thread 决定寄存器如何缓存数据以复用，instruction 是硬件级 `mma.sync` / `wgmma.mma_async` 的粒度，epilogue 是 accum → output 的写回阶段（activation、scale、bias 融合都在这里）。],
  kind: table,
)

*Observation*：本书 K0–K8 沿"CTA → thread → instruction"这条主路径爬完，但*没走 warp specialisation*——CUTLASS 3.x 的 pingpong / cooperative scheduler 让 warpgroup 一部分做 producer、一部分做 consumer，是 A100/H100 拿到 tensor-core 峰值 60–80% 的关键（章末 K11 讲）。手写还想继续追，需要 (1) 更大 tile（$128 times 128$ 及以上）+ (2) `cp.async` 多 stage pipeline + (3) swizzled SMEM layout。这三件事互相耦合，一起改。

CUTLASS 还处理了：auto-tuning tile 参数、split-K、Stream-K、TMA async copy（SM90）、FP8/Block-scaled 等。本书 K0–K8 ladder 是 CUTLASS 的简化手工版；想读懂 CUTLASS 3.x，先把本章 K3–K8 完全内化。

== Split-K：什么时候用

标准 GEMM：每个 block 沿 $K$ 完整累加，写最终 $C$。

*Split-K*：$K$ 太大或 CTA tile 太大导致 grid 太小（GPU 填不满）时，把 $K$ 切给多个 block 并行累加，最后 reduce：

```
Block (0,0,k_slice=0): C_partial[0] += A[:, 0:K/2]   × B[0:K/2, :]
Block (0,0,k_slice=1): C_partial[1] += A[:, K/2:K] × B[K/2:K, :]
Final: C = C_partial[0] + C_partial[1]
```

*适用场景*：

- $M, N$ 小、$K$ 大 → grid 太小，SM 利用率低。
- 想要更大 CTA tile（提高 AI）但 $K$ 方向 smem 放不下 → 拆 $K$。

*代价*：需要 atomic add 或 secondary reduction kernel，引入同步开销。cuBLAS 内部 heuristics 自动决定是否 split-K。

#interview[
  Split-K 和 K-dimension tiling（本章 K3+ 的 `tile_k` 循环）是不同的：后者是在*同一个 block 内*沿 $K$ 累加，最后一次性写 $C$；前者是*多个 block 分工* $K$ 的不同段，需要 atomic add 或二级 reduction merge。
]

== cuBLAS 为什么快

把本章 ladder 和 cuBLAS 对比，差距来自：

1. *Tensor core*：硬件算力 10×+ 于 CUDA core FMA。
2. *Years of auto-tuning*：每种 $(M,N,K)$ 组合、每种 GPU 架构都有 tuned tile config。
3. *完整 pipeline*：`cp.async` 多 stage + `ldmatrix` + `mma` 深度 overlap。
4. *Swizzle + layout*：零 bank conflict 的 smem layout。
5. *Split-K / Stream-K*：动态 grid 利用率优化。
6. *Epilogue fusion*：bias + ReLU 不额外读写 $C$。
7. *Workspace algorithms*：某些 shape 用 non-standard 算法（Strassen 不会，但 Winograd 在某些 conv 里类似）。

#note[
  手写 GEMM 追 cuBLAS 不现实——面试期望的是：你能讲清楚 ladder 每一层的原理，知道 cuBLAS 多了什么，能用 ncu 定位自己的瓶颈在哪一层。
]

== Tile 大小怎么选

没有万能公式，但有一套系统方法：

*Step 1：算 smem 用量*

$ "smem" = (B_M times B_K + B_K times B_N) times 4 "B" $

A100 每 SM shared memory 164 KB（可配），但和 L1 共享。超过 48 KB 可能限制每 SM 的 max blocks。

*Step 2：算 register 用量*

$ "regs" approx "TM" times "TN" + "fragments" + "indices" $

用 `--ptxas-options=-v` 实测。目标：256 threads/block 时 regs ≤ 64 以保持 50%+ occupancy。

*Step 3：算 AI*

$ "AI" approx frac(B_M B_N, 2(B_M + B_N)) "FLOP/B" $

要超过 ridge point（A100 FP32: 13，TF32 tensor: ~20+）。

*Step 4：查 occupancy*

CUDA Occupancy Calculator 或 `cudaOccupancyMaxActiveBlocksPerMultiprocessor`。

*Step 5：benchmark 几个候选*

$(64,128,16)$, $(128,128,16)$, $(128,256,16)$ 等。peak 往往在 AI 和 occupancy 的交点。

#insight[
  面试手写 tiled GEMM 时，选 $16 times 16 times 16$ 或 $32 times 32 times 16$ 就好——重点是结构正确，不是 tune 到 peak。
]

=== Occupancy 手算示例

CTA tile $128 times 128 times 16$，FP32，2×2 thread tile，256 threads/block：

- Smem：$(128 times 16 + 16 times 128) times 4 = 16 "KB"$ → 每 SM 可驻 2~3 个 block（48 KB 限额内）。
- Registers：假设 64 regs/thread → $256 times 64 = 16384$ regs/block。A100 每 SM 65536 regs → 最多 4 block/SM（寄存器维度）。
- 实际 bottleneck 往往是 regs 或 smem，取 min → 2 block/SM × 256 threads = 512 threads/SM → occupancy 25%（2048 max）。

*这够吗？* 对 compute-bound GEMM，25% occupancy 有时反而更快——每个 block 更大、AI 更高、每个 SM 上 active warps 虽少但每个 warp 的 instruction issue 更满。这就是为什么 GEMM tuning 不能只看 occupancy 数字，必须 benchmark。

#note[
  Hopper (SM90) 的 TMA 可以把 smem load 从 thread 职责里剥离，进一步改变 occupancy tradeoff。本书聚焦 Ampere 路径，SM90 的 TMA 可以看作 `cp.async` 的"硬件加速升级版"。
]

== ncu 该看什么

```bash
ncu --set full --section SpeedOfLight ./build/04_matmul
```

GEMM 关键 metric：

- `smsp__sass_thread_inst_executed_op_ffma_pred_on.sum` / peak：FMA 利用率。
- `sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active`：tensor core 利用率（WMMA 版）。
- `dram__bytes.sum.per_second`：是否还在 memory-bound。
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`：bank conflict 计数。
- `sm__warps_active.avg.pct_of_peak_sustained_active`：occupancy。
- `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`：compute vs memory 哪个是瓶颈。

健康的 tensor core GEMM：`pipe_tensor` > 80%，bank conflicts = 0，occupancy > 50%。

== 面试白板 code

面试官说"手写一个 matmul"——不要开口就 tensor core（除非要求）。先写 shared-mem tiled 版本，这是所有 GEMM 优化的基础骨架：

```cpp
// C = A * B, A: [M, K], B: [K, N], C: [M, N]. 行主序.
constexpr int TILE = 32;

__global__ void gemm_tiled(const float* A, const float* B, float* C,
                           int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;  // C 的行
  int col = blockIdx.x * TILE + threadIdx.x;  // C 的列
  float acc = 0.f;

  // K 维度 tile 循环
  for (int kt = 0; kt < K; kt += TILE) {
    // Cooperative load: 每 thread 搬一个元素. 边界条件用 0 padding.
    As[threadIdx.y][threadIdx.x] =
        (row < M && kt + threadIdx.x < K) ? A[row * K + kt + threadIdx.x] : 0.f;
    Bs[threadIdx.y][threadIdx.x] =
        (kt + threadIdx.y < K && col < N) ? B[(kt + threadIdx.y) * N + col] : 0.f;
    __syncthreads();  // 等所有 lane 把 tile 填好

    // 每 thread 做 TILE 次 FMA，B 广播、A 沿列.
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
      acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    __syncthreads();  // 等所有 lane 用完 tile 再进下一轮 load
  }

  if (row < M && col < N) C[row * N + col] = acc;
}

// ==== Launch config ====
// blockDim = (32, 32) = 1024 threads (32 warp)：
//   * 恰好覆盖一个 32×32 输出 tile，1 thread 算 1 个 C[i, j]；
//   * 32 是 warp 宽度——threadIdx.x 沿列走保证 gmem/smem 访问 coalesced；
//   * 1024 是 A100 每 block threads 上限，占用 32 warp / SM，寄存器压力见下.
// gridDim  = ((N+31)/32, (M+31)/32)：每 block 一个输出 tile.
//   * 每 block smem = 2 * 32 * 32 * 4B = 8 KB, A100 每 SM 有 164 KB smem, 够开 20 个 block/SM;
//   * 每 thread 约 30 register, 1024 * 30 = 30720 reg > SM 上限 64 K？其实 30 reg 时正好.
//     实际由 nvcc `-maxrregcount` 控制, 面试时说 "会看 occupancy calculator".
dim3 block(TILE, TILE);
dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
gemm_tiled<<<grid, block>>>(A, B, C, M, N, K);
```

*核心考点*（追问顺序）：

- *"为什么 tiled 快？算算 AI。"* → naive 每 FMA 读 8B、AI = 0.125 FLOP/B；tiled 每个 tile 从 gmem 读 $2 T^2$ 个元素、做 $2 T^3$ FLOP，AI = $T/2$。$T = 32$ 时 AI = 16 > A100 ridge point 13 → 变 compute-bound。
- *"两个 `__syncthreads` 分别在同步什么？"* → 第一个：等 load 完成才能开始 compute（RAW）；第二个：等 compute 用完 smem 才能被下一轮 load 覆写（WAR）。
- *"bank conflict 在哪？"* → `As[ty][k]` 沿 `k`：warp 内 32 lane 的 `ty` 相同、`k` 由 unroll 决定同值 → broadcast、无冲突。`Bs[k][tx]` 沿 `tx`：`k` 相同、`tx` 全不同 → 32 不同 bank、无冲突。所以这份骨架天然无 bank conflict——面试官如果问一定要答出来。
- *"下一步优化？"* → (1) thread tiling：每 thread 算 4×4 或 8×8 输出，重用寄存器 & 减 smem load；(2) 双 buffer + async copy 让 load 和 compute 重叠；(3) 换 tensor core。三级 ladder 每级 2-3× 加速。
- *"tensor core 怎么用？"* → `wmma::mma_sync(D, A, B, C)`，$M=N=K=16$、fp16 输入 fp32 accum，一条指令代替 4096 次 FMA。SM80 上还有 `mma.sync.m16n8k16`（更灵活的 tile 形状）。
- *"blockDim 为什么选 32×32？"* → `threadIdx.x` 沿列走：warp 内 32 lane 的 `tx` 全不同，读 `Bs[k][tx]` 沿列 → 32 不同 bank / gmem 32 连续 word，两侧都 coalesced。如果反过来 `threadIdx.y` 沿列，warp 内 `tx` 相同 → 所有 lane 挤 1 个 bank，32-way conflict。这是 CUDA 里"threadIdx.x 一定要映射到 stride-1 维度"的经典应用。

== 面试考点

#interview[
  *Q1*: 为什么 naive GEMM 是 memory-bound？AI 大概多少？

  A: 每个输出读 $2K$ 个 float、写 1 个 float、做 $2K$ FLOP。AI $approx 1/4 "FLOP/B"$。A100 ridge $approx 13 "FLOP/B"$，差 50 倍。且 $B$ 的列访问 stride-$N$ 不合并，实际更差。
]

#interview[
  *Q2*: Shared memory tiling 怎么提升 AI？公式是什么？

  A: Block 协作读 $B_M times B_K$ 的 $A$ 和 $B_K times B_N$ 的 $B$，算 $B_M times B_N$ 个输出。每个 $A/B$ 元素被复用 $B_N / B_K$ 或 $B_M / B_K$ 次。有效 AI $approx B_M B_N / (2(B_M + B_N))$ FLOP/B，随 tile 线性增长。
]

#interview[
  *Q3*: Tiled GEMM 里两个 `__syncthreads` 分别保护什么？

  A: 第一个：所有 thread 完成 smem 写入后再读取（load → compute）。第二个：所有 thread 完成 smem 读取后再覆写（compute → next load）。缺任何一个都会 data race。
]

#interview[
  *Q4*: 什么是 bank conflict？怎么解决？

  A: 32 个 bank 各 4B 宽，warp 并发访问同一 bank 不同 word 时串行化。GEMM 中常见原因是 smem 行宽为 32 的倍数导致 stride-32 访问。解决：swizzle（XOR row 低位到 col 索引）、padding（行宽 +1）、或 permute layout。
]

#interview[
  *Q5*: Thread tile (register tiling) 的好处是什么？

  A: 每个 thread 维护 TM×TN 个 accumulators，同一个 $A/B$ smem 值被复用 TM 或 TN 次，提高 register 计算密度、减少 smem 读次数。代价是 register 占用增加，可能降低 occupancy。
]

#interview[
  *Q6*: `cp.async` 双缓冲 pipeline 怎么写？

  A: 两个 smem stage。Prologue 预 load stage 0。主循环：wait stage *i* ready → sync → compute on stage *i* → async load stage *i+1* → commit。Epilogue 算最后一个 stage。关键：`wait_group` + `__syncthreads` 配合，不能 overwrite 正在 compute 的 stage。
]

#interview[
  *Q7*: Tensor core m16n8k16 是什么意思？一个 warp 做什么？

  A: 一个 warp 协作执行一条 mma 指令，完成 $16 times 8$ 输出、$K = 16$ 的外积累加。每个 thread 持有 matrix fragment 的一部分（registers），通过 `ldmatrix` 从 swizzled smem 加载，再 `mma.sync`。不是每个 thread 独立算一个输出。
]

#interview[
  *Q8*: CUTLASS 的四层 tile 结构是什么？

  A: CTA tile（block 级 smem tile）→ Warp tile（warp 级分工 + warp mma）→ Thread tile（register accumulators）→ Instruction tile（`mma.sync` m16n8k16）。每层让上层数据在下层复用更多次。
]

#interview[
  *Q9*: Split-K 什么时候用？和 K-dimension tiling 有什么区别？

  A: 当 grid 太小（$M, N$ 小）或需要更大 CTA tile 但 smem 不够时，多个 block 分工 $K$ 的不同段，最后 reduce。K-dimension tiling 是 block 内沿 $K$ 循环累加，不需要跨 block reduce。Split-K 有 atomic/reduction 开销。
]

#interview[
  *Q10*: cuBLAS 为什么比手写 GEMM 快很多？

  A: Tensor core 硬件算力、多年 auto-tuned tile config、`cp.async` 多 stage pipeline、零 conflict swizzle layout、split-K/Stream-K grid 优化、epilogue fusion。手写版本很难在每条上都到位。
]

#interview[
  *Q11*: 怎么选 GEMM 的 tile 大小？

  A: 约束三方平衡：(1) smem 不超过硬件限制且 occupancy 可接受；(2) register 用量不导致 occupancy 暴跌；(3) AI 超过 ridge point。用 occupancy calculator + ncu benchmark 几个候选 config，选实测最快的。
]

#interview[
  *Q12*: GEMM 中 $B$ 矩阵的访问模式为什么比 $A$ 更棘手？

  A: `A[row, :]` 连续访问（合并）。`B[:, col]` stride-$N$ 访问（不合并）。Tiling 后 $B$ 的 smem tile 是连续块，解决了 global 访问；但 smem 读 `b_tile[inner][tx]` 需要正确的 layout/swizzle 避免 bank conflict。
]
