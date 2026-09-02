#import "../template.typ": *

= Matmul (GEMM)

matrix multiply 是 GPU kernel 优化的*终极试金石*。vector add 的 naive 版本已经能打满带宽；reduction 靠 shuffle 和 warp 协作就能逼近峰值；但 GEMM 从 naive 到 cuBLAS / CUTLASS 级别，中间隔着整整一条优化 ladder——shared memory tiling、register tile、向量化 load、warp 分工、bank conflict swizzle、`cp.async` pipeline、tensor core `mma.sync`……每一层都对应一个可量化的瓶颈和一套可面试的追问。

这一章是全书中*最长、最难*的一章。我们要把它讲透：

- GEMM 的数学定义、存储布局、以及为什么它是 compute-bound 的经典例子。
- Roofline 分析：AI 如何从 naive 的 $O(1)$ 随 tile 增大而爬升。
- 从 naive → shared memory tile → warp tile → register tile → pipeline 的完整 ladder（对应源码五个 kernel）。
- 源码里没有单独写、但面试必问的：`float4` 向量化、`cp.async` 双缓冲、tensor core fragment 布局、bank conflict swizzle、CUTLASS 分层、split-K。
- 怎么选 tile 大小、怎么用 ncu 诊断、cuBLAS 为什么快。

对应源码：`src/cuda/04_matmul.cu`。

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

$M = N = K = 4096$，A100 FP32，相对 cuBLAS 的粗略比例（教学实测，具体数字因 GPU/driver 版本略有出入）：

#ladder(
  ("naive",              "1 thread / output",                    "~1%"),
  ("shared-memory tile", "16×16 smem tile",                      "~5%"),
  ("warp tile",          "8 warps × 2 outputs/lane",             "~15%"),
  ("register tile",      "2×2 thread tile",                      "~25%"),
  ("+ vectorized load",  "float4 global→smem",                   "~35%"),
  ("+ swizzle + cp.async","bank-free + pipeline",                "~55%"),
  ("tensor core WMMA",   "m16n8k16 mma.sync",                    "~85%"),
  ("cuBLAS",             "years of tuning + split-K + ...",      "100%"),
)

#warn[
  上表是*量级参考*，不是精确 benchmark。本书源码的 ladder 停在 pipeline teaching 版（scalar FMA），目的是读懂结构，不是和 cuBLAS 赛跑。真正追 peak 需要 tensor core + 完整 pipeline + 专业 tuning。
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

== v1: naive

```cpp
__global__ void matmul_naive_kernel(
    const float* a, const float* b, float* c,
    int m, int n, int k) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= m || col >= n) return;

  float acc = 0.0f;
  for (int inner = 0; inner < k; ++inner) {
    acc += a[row * k + inner] * b[inner * n + col];
  }
  c[row * n + col] = acc;
}
```

Launch：`block(16, 16)`，`grid(ceil(n/16), ceil(m/16))`。一个 thread 负责 `C[row, col]` 一个输出。

=== 三个致命问题

*1. 零数据复用*

同一个 block 里，相邻 thread 读 $A$ 的*同一行*但 $B$ 的*不同列*。`A[row, :]` 被 16 个 thread 各读一遍（本来可以 1 遍）。$B$ 同理——`B[:, col]` 被 16 个 thread 各读一遍。

一个 $16 times 16$ block 本可以协作读 $16 times 16$ 的 $A/B$ 子块各一次，naive 却读了 $16 times (16 + 16) = 512$ 次（每个元素被 16 个 thread 重复读）。

*2. $B$ 的访问不合并*

`A[row * k + inner]`：固定 `row`，`inner` 递增——连续访问，合并 ✓。

`B[inner * n + col]`：固定 `col`，`inner` 递增——stride = $N times 4$ B。一个 warp 里 32 个 thread 的 `col` 连续，但 `inner` 相同时访问地址间隔 $N times 4$ B。当 $N >= 32$ 时，32 个 lane 的地址跨整个矩阵宽度——*完全不合并*，32 个独立 transaction。

*3. 无 latency hiding*

每个 thread 独立做 $K$ 次串行 load + FMA。一个 block 只有 256 个 thread，对于 $K = 4096$ 来说，memory latency 完全暴露。

#note[
  naive GEMM 的价值是*建立正确性直觉*：输出矩阵每个元素 = $A$ 的一行 · $B$ 的一列。优化 ladder 的每一步都是在不改变这个数学的前提下，减少冗余读取、改善访问模式、提高计算密度。
]

== v2: shared memory tiling

```cpp
constexpr int kTile = 16;

__global__ void matmul_tiled_kernel(
    const float* a, const float* b, float* c,
    int m, int n, int k) {
  __shared__ float a_tile[kTile][kTile];
  __shared__ float b_tile[kTile][kTile];

  const int row = blockIdx.y * kTile + threadIdx.y;
  const int col = blockIdx.x * kTile + threadIdx.x;
  float acc = 0.0f;

  for (int tile_k = 0; tile_k < k; tile_k += kTile) {
    const int a_col = tile_k + threadIdx.x;
    const int b_row = tile_k + threadIdx.y;

    a_tile[threadIdx.y][threadIdx.x] =
        (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
    b_tile[threadIdx.y][threadIdx.x] =
        (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;

    __syncthreads();  // ① load 完成，才能读 smem

    #pragma unroll
    for (int inner = 0; inner < kTile; ++inner) {
      acc += a_tile[threadIdx.y][inner] * b_tile[inner][threadIdx.x];
    }

    __syncthreads();  // ② compute 完成，才能覆写 smem
  }

  if (row < m && col < n) {
    c[row * n + col] = acc;
  }
}
```

=== 核心思想

沿 $K$ 维度分块。每次循环：

1. 协作把 `A[row:row+16, tile_k:tile_k+16]` 和 `B[tile_k:tile_k+16, col:col+16]` 搬进 shared memory。
2. 256 个 thread 在 smem 里做 $16 times 16$ 的小矩阵乘——每个 $A/B$ 元素被 16 个 thread 复用。
3. `tile_k += 16`，推进到下一个 $K$ slab。

=== BM×BK tile 与 K 维度分块

记 $B_M = B_N = B_K = 16$（本书用方阵 tile 简化讲解；生产代码 $B_M, B_N, B_K$ 通常不相等，比如 $128 times 256 times 16$）。

- `a_tile[ty][tx]`：thread `(ty, tx)` 搬 `A[row, tile_k + tx]`。
- `b_tile[ty][tx]`：thread `(ty, tx)` 搬 `B[tile_k + ty, col]`。

内层循环 `inner = 0..15`：thread `(ty, tx)` 读 `a_tile[ty][inner]`（$A$ 的第 `ty` 行）和 `b_tile[inner][tx]`（$B$ 的第 `tx` 列），做外积累加。

=== `__syncthreads` 的两个位置

*同步 ①*（load 后、compute 前）：所有 thread 必须完成 smem 写入，才能开始读取。缺少它 → 读到未初始化数据 → 结果错误。

*同步 ②*（compute 后、下一轮 load 前）：所有 thread 必须用完当前 smem 数据，才能覆写。缺少它 → 慢 thread 还在读 `a_tile[ty][inner]`，快 thread 已经开始写下一轮 → 数据竞争。

#warn[
  两个 `__syncthreads` 缺一不可，且不能合并成一个。这是 tiled GEMM 最容易写错的地方。面试手写 tiled matmul 时，先画 timeline：load → sync → compute → sync → load → ...
]

=== 双缓冲（ping-pong）思路

当前版本：load → sync → compute → sync → load → ... 完全串行，SM 在 load 阶段 idle，在 compute 阶段 memory pipe idle。

双缓冲：准备两块 smem `a_tiles[2][...]`, `b_tiles[2][...]`。

```
stage 0: [load tile 0]
stage 1: [compute tile 0] || [load tile 1]   ← 理想情况下重叠
stage 2: [compute tile 1] || [load tile 2]
...
```

用 `cp.async`（下一节详讲）可以让 load 和 compute *真正* overlap。本书 v5 pipeline teaching kernel 用 `stage = tile_index & 1` 实现了 ping-pong 骨架（仍用 scalar load，结构先行）。

=== 这版还剩什么瓶颈

1. 每个 thread 只算 1 个输出 → register 利用率低。
2. Global load 仍是 scalar `float` → 带宽浪费。
3. `a_tile[ty][inner]` 如果布局不当会有 bank conflict（后面专讲）。
4. $B_M = B_N = B_K = 16$ 太小 → AI 只有 $approx 4 "FLOP/B"$，还在 memory-bound 区间。

== v3: warp tile

```cpp
// block: (32, 8) = 256 threads = 8 warps
// block tile: 8×64 output
__global__ void matmul_warp_tiled_kernel(...) {
  __shared__ float a_tile[kBlockTileM][kBlockTileK];  // 8×16
  __shared__ float b_tile[kBlockTileK][kBlockTileN];  // 16×64

  const int lane = threadIdx.x;       // 0..31, warp 内 lane id
  const int warp_id = threadIdx.y;    // 0..7, 第几个 warp
  const int row = blockIdx.y * kBlockTileM + warp_id;
  const int col0 = blockIdx.x * kBlockTileN + lane;
  const int col1 = col0 + 32;

  float acc0 = 0.0f, acc1 = 0.0f;

  for (int tile_k = 0; tile_k < k; tile_k += kBlockTileK) {
    // 协作 load A tile (128 elements) 和 B tile (1024 elements)
    ...
    __syncthreads();

    if (row < m) {
      #pragma unroll
      for (int inner = 0; inner < kBlockTileK; ++inner) {
        const float a_value = a_tile[warp_id][inner];
        acc0 += a_value * b_tile[inner][lane];
        acc1 += a_value * b_tile[inner][lane + 32];
      }
    }
    __syncthreads();
  }
  // 写回 acc0 → C[row, col0], acc1 → C[row, col1]
}
```

=== 分工结构

```
block tile (8×64)
├── warp 0 → row 0, cols [0..63]
├── warp 1 → row 1, cols [0..63]
├── ...
└── warp 7 → row 7, cols [0..63]

每个 warp 内:
  lane 0  → col 0, col 32
  lane 1  → col 1, col 33
  ...
  lane 31 → col 31, col 63
```

*关键变化*：

1. *Warp 级分工*：`threadIdx.y` = warp id，`threadIdx.x` = lane id。一个 warp 负责 block tile 的一整行。
2. *Register 复用*：每个 lane 算 2 个输出（`col0` 和 `col1`）。同一个 `a_value = a_tile[warp_id][inner]` 乘两个不同的 $B$ 值——$A$ 数据在 register 里复用 2 次。
3. *Broadcast*：`a_tile[warp_id][inner]` 对一个 warp 内所有 lane 相同——编译器/NVIDIA 硬件会把它优化成 warp-wide broadcast（从 smem 读一次，广播给 32 lane）。

#insight[
  这就是 CUTLASS 分层的第一 glimpse：*CTA tile* (8×64) → *warp tile* (1×64) → *thread tile* (1×2)。每一层让上一级数据在下一级被复用更多次。
]

=== 为什么 block 是 32×8

`threadIdx.x = 32` 刚好一个 warp 宽度——warp 内没有 partial warp 浪费。`threadIdx.y = 8` 个 warp 覆盖 8 行。

Load 阶段用 `linear_tid = ty * 32 + tx` 做 grid-stride 协作搬运（B tile 有 1024 元素，256 thread 每人搬 4 个）——这和 vector add 的 grid-stride 是同一个模式。

== v4: register tile (thread tile)

```cpp
constexpr int kThreadTileM = 2;
constexpr int kThreadTileN = 2;
// block tile: 32×32, 每个 thread 算 2×2 = 4 个输出

__global__ void matmul_register_blocked_kernel(...) {
  __shared__ float a_tile[32][16];
  __shared__ float b_tile[16][32];

  const int row_base = blockIdx.y * 32 + ty * 2;
  const int col_base = blockIdx.x * 32 + tx * 2;
  float acc[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

  for (int tile_k = 0; tile_k < k; tile_k += 16) {
    // 每个 thread 搬 A 的 2 行 × 1 列, B 的 1 行 × 2 列
    a_tile[ty][tx] = ...;           // row ty
    a_tile[ty + 16][tx] = ...;      // row ty+16
    b_tile[ty][tx] = ...;           // col tx
    b_tile[ty][tx + 16] = ...;      // col tx+16

    __syncthreads();

    #pragma unroll
    for (int inner = 0; inner < 16; ++inner) {
      const float a_frag0 = a_tile[ty * 2 + 0][inner];
      const float a_frag1 = a_tile[ty * 2 + 1][inner];
      const float b_frag0 = b_tile[inner][tx * 2 + 0];
      const float b_frag1 = b_tile[inner][tx * 2 + 1];

      acc[0][0] += a_frag0 * b_frag0;
      acc[0][1] += a_frag0 * b_frag1;
      acc[1][0] += a_frag1 * b_frag0;
      acc[1][1] += a_frag1 * b_frag1;
    }
    __syncthreads();
  }
  // 写回 2×2 累加器
}
```

=== TM×TN thread tile

每个 thread 维护 `acc[TM][TN]` = `acc[2][2]`，一次 inner 迭代：

- 读 2 个 $A$ 值 + 2 个 $B$ 值（共 4 个 smem load）
- 做 4 次 FMA（更新 4 个输出）

*数据复用比*：每个 $A/B$ smem 元素被复用 `TN`/`TM` 次。2×2 tile → 每个 load 服务 2 个 FMA。

生产 GEMM 常见 4×8、8×8 等更大 thread tile——register 占用和 AI 的 tradeoff。

=== 寄存器压力

`acc[2][2]` + 4 个 fragment + loop 变量 ≈ 10+ registers/thread。如果 thread tile 太大（比如 8×8 = 64 个 acc），256 thread/block 可能需要 200+ registers/thread → occupancy 暴跌。

#note[
  选 thread tile 大小时，先用 `--ptxas-options=-v` 看 register 用量，再查 occupancy calculator。A100 一个 SM 最多 65536 registers，2048 threads → 平均 32 registers/thread 时 occupancy 100%。
]

=== 计算顺序：inner product vs outer product

当前实现是 *inner product* 风格：固定 `inner`，遍历 $A$ 的一列片段和 $B$ 的一行片段。

高性能 GEMM 也常用 *outer product* 风格：固定 $A$ 的一小列 + $B$ 的一小行，外积累加到整个 acc 矩阵。Tensor core 版本必须是 outer product（`mma.sync` 的语义）。

== v5: pipeline + ping-pong staging

源码的 `matmul_pipeline_teaching_kernel` 在 v4 基础上加了双缓冲：

```cpp
__shared__ float a_tiles[2][32][16];
__shared__ float b_tiles[2][16][32];

// prologue: 搬 tile 0 到 stage 0
__syncthreads();

for (int tile_index = 0; tile_index < num_k_tiles; ++tile_index) {
  const int stage = tile_index & 1;

  // 用 stage 的数据计算
  #pragma unroll
  for (int inner = 0; inner < 16; ++inner) {
    acc[0][0] += a_tiles[stage][ty*2+0][inner] * b_tiles[stage][inner][tx*2+0];
    // ... acc[0][1], acc[1][0], acc[1][1]
  }

  // 预加载下一块到 stage ^ 1
  if (tile_index + 1 < num_k_tiles) {
    __syncthreads();
    // load into a_tiles[next_stage], b_tiles[next_stage]
    __syncthreads();
  }
}
```

#insight[
  这个 kernel 仍然是 scalar FMA——*不是*真正的 tensor core 实现。它的教学价值是让你读懂 modern GEMM 的 pipeline 骨架：prologue → (compute stage *i* || load stage *i+1*) → epilogue。真正的 overlap 需要 `cp.async` 或 TMA（SM90+）。
]

=== Pipeline 三阶段时间线

```
时间 →
Prologue:  |-- load K-slab 0 → smem stage 0 --|
Loop i=0:  |-- compute on stage 0 --|-- load K-slab 1 → stage 1 --|
Loop i=1:  |-- compute on stage 1 --|-- load K-slab 2 → stage 0 --|  ← 覆写 stage 0 前必须等 i=0 的 compute 全部完成
...
Epilogue:  |-- compute last stage --|
```

Teaching kernel 用两个 `__syncthreads()` 把 load 和 compute *分开*——清楚但无 overlap。`cp.async` 版在 compute 循环体内插入 async copy，由 hardware 保证 stage 就绪，compute 和 load 并行。

=== Register tile 的 inner vs outer product

Teaching kernel 和 register-blocked 版都用 *inner product* 风格：固定 `inner`，取 $A$ 的列片段和 $B$ 的行片段做点积。

Tensor core 路径必须走 *outer product*：每次 `mma.sync` 完成 $K = 16$ 的外积累加，$A$ 的 $16 times 16$ 列块和 $B$ 的 $16 times 8$ 行块 outer-product 到 $16 times 8$ 的 accumulators。CUTLASS 的主循环结构是 outer-product over $K$，和本章 scalar 版的 inner loop 方向相反——读 smem 的 pattern 也因此不同。

== 实测

$M = N = K = 64$（$A + B + C approx 48 "KB"$，整工作集落在 L2 内），A100 80GB SXM4，`ncu --set full` 抓取每个 kernel 的一次 launch。本章是 *compute 章*：perf 表主列是 `TC %` 和 `warp %`，不是 HBM %——但 $64^3$ 规模下两者都极低，定性结构仍看 diag 表。

Launch 配置和 grid 规模如下——grid 极小，远填不满 108 个 SM：

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt + gray,
    inset: 5pt,
    align: (left, left, left, right),
    [*version*], [*grid*], [*block*], [*输出覆盖*],
    [naive / tiled], [(4, 4, 1)], [(16, 16, 1)], [$64 times 64$（$16 times 16$ tile × 16 block）],
    [warp-tiled], [(1, 8, 1)], [(32, 8, 1)], [$64 times 64$（$8 times 64$ tile × 8 block）],
    [register-blocked / pipeline], [(2, 2, 1)], [(16, 16, 1)], [$64 times 64$（$32 times 32$ tile × 4 block）],
  ),
  caption: [*Table:* ch4 matmul 各 ladder 版本的 launch 配置。*grid* / *block* 直接对应 `<<<grid, block>>>` 的 CUDA launch 参数（`launch__grid_size` / `launch__block_size`）。*输出覆盖* 说明该 grid 每 block 负责的输出 tile 大小 × block 数量 = 完整的 $64 times 64$ 输出。三种配置对同一输出规模选择了不同的 tile / block / thread-work 权衡：naive 一 thread 一 output（256 thread × 16 block）；warp-tiled 一 warp 一 row（1 warp × 8 block × 8 row/warp）；register-blocked 一 thread 一 4×4 output tile（256 thread × 4 block × 16 output/thread）。],
  kind: table,
)

*Observation*：三档 grid 大小都远小于 A100 的 108 个 SM——4×4=16、1×8=8、2×2=4 个 block，一次 wave 就跑完，*根本没进入稳态*。这解释了为什么 `sm__cycles_active` 只有 1% 左右：SM 大部分时间在等 launch overhead 摊薄。所以下面 diag 表的 `warp %`（约 1%）不是 kernel 效率差，而是问题规模太小的伪影。

#include "../bench/04_matmul.typ"

*TC %* = `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`；*warp %* = `sm__warps_active.avg.pct_of_peak_sustained_elapsed`；*HBM %* = `dram__bytes.sum.pct_of_peak_sustained_elapsed`。详见附录 B。

#warn[
  这一章的问题规模是教学 default（$M = N = K = 64$，三个矩阵共约 48 KB），kernel 单次运行只有 6–14 μs。ncu 的定性指标（`issued/32`、`smem conf.`、`barrier stall`）仍能反映 kernel 结构，但*绝对数字对生产规模不完全可信*：
  - TC % / warp % 会极低（108 个 SM 填不满、无 tensor pipe 活动）
  - HBM % 会偏低（工作集在 L2 内，分母 elapsed time 含冷启动窗口）
  想拿到生产规模的数字，把 $M, N, K$ 拉到 4096+，让工作集远超 L2 (40 MB)。
]

*perf 表读三件事：*

+ *TC % 全表 = 0.0——五个 kernel 没有一个走 tensor pipe*。源码全部 `float` + 手写 FFMA，无 `wmma::`、`mma.sync`、`ldmatrix`。`pipeline-teaching` 名字带 "tensor-core"，注释写得很清楚：*仍是 scalar FMA，只展示 ping-pong 骨架*。这是本章最重要的 honesty 点——后半段 WMMA / fragment layout 是*面试和读 CUTLASS 需要的概念*，不是 `./build/04_matmul` 已实现的路径。要测 TC 收益，跑 cuBLAS `cublasGemmEx` 或 CUTLASS example。

+ *tiled 最快（6.24 μs），warp-tiled 最慢（13.76 μs）*。tiled 比 naive 快 26%（6.24 vs 8.48 μs）；warp-tiled 比 naive *慢* 62%——ladder 位置不等于快慢。`warp %` 全表 0.3–1.4%：最多 16 block × 256 thread = 4096 thread，108 SM × 2048 thread/SM 的理论上限差三个数量级——*测到的是结构差异，不是 occupancy 饱和下的 GEMM 峰值*。

+ *register-blocked / pipeline-teaching 介于中间*（7.52 / 7.10 μs）。register 复用把 `mem stall` 从 naive 的 13.07 降到 4.08–4.23，但 $K = 64$ 只有 4 个 K-slab，FMA 密度提升不够抵消 sync 和 smem 开销；pipeline 版双缓冲 8 KB smem 无 `cp.async` overlap，只比 register-blocked 快 6%。

#figure(
  hbar-chart(
    (
      ("tiled", 6.24),
      ("pipeline-teaching", 7.10),
      ("register-blocked", 7.52),
      ("naive", 8.48),
      ("warp-tiled", 13.76),
    ),
    unit: "μs",
  ),
  caption: [`time (μs)` 排序：tiled smem 复用 win；warp-tiled 结构复杂度在小 $N$ 下反噬。],
)

*diag 表读关键教学点：*

*a) 全表 `issued/32 = 32.0`——没有 warp divergence*。边界 `if (row >= m || col >= n)` 和 warp-tiled 的 `if (row < m)` 编译成 predicated FFMA，不是不同 basic block 的分支 divergence。`pred_on/32` 31.2–31.8，issued − pred_on ≈ 0.2–0.8 来自 grid 边缘 thread——*用 "warp lane utilization" 描述 gap，不要说 "divergence"*。

*b) register-blocked / pipeline-teaching：`smem conf. = 256`——全书第一次非零 bank conflict*。naive / tiled / warp-tiled 都是 0——*不能从源码推断 conflict，只有 metric 说了算*。

内层循环读 `a_tile[ty * 2 + 0][inner]` 和 `a_tile[ty * 2 + 1][inner]`：`a_tile` 行宽 16 float = 64 B，行 stride 16 mod 32 = 16 bank → 行 0 与行 2、行 4… 映射到同一 bank 组。一个 warp 里 `ty = 0, 1` 的 thread 同时读 row 0/1/2/3 → 固定 `inner` 时 2-way bank conflict。这是 *register blocking + 固定行 stride* 的代价；生产 GEMM 用 XOR swizzle 或 padding 把 `smem conf.` 压回 0（见下文 bank conflict 节）。

*c) tiled 引入 sync 成本，warp-tiled 更重*。naive `barrier stall = 0.00`（无 smem）；tiled `0.82`；warp-tiled `1.26`——每 K-slab 两次 `__syncthreads`，$K = 64$、$B_K = 16$ 时重复 4 次，sync 占比在小 kernel 里被放大。warp-tiled 额外有 grid-stride load B tile（1024 元素 / 256 thread）的 loop 控制开销，grid 只有 8 block（vs naive/tiled 16 block），`mem stall = 10.47` 仍高于 tiled 的 7.61——latency hiding 更差。

*d) register 复用降低 mem stall，但 smem conflict 抵消部分收益*。register-blocked `mem stall = 4.23`（naive 13.07 的 1/3），每个 $A/B$ smem 值服务 2×2 FMA；同时 `smem conf. = 256` 让 smem load 串行化。pipeline-teaching 与 register-blocked 共享同一 compute 路径，diag 几乎相同——双缓冲没带来结构性的 stall 下降。

#figure(
  warp-grid(
    rows: 8, cols: 32,
    active: (
      (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
      (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (0, 13), (0, 14), (0, 15),
      (0, 16), (0, 17), (0, 18), (0, 19), (0, 20), (0, 21), (0, 22), (0, 23),
      (0, 24), (0, 25), (0, 26), (0, 27), (0, 28), (0, 29), (0, 30), (0, 31),
      (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
      (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (1, 14), (1, 15),
      (1, 16), (1, 17), (1, 18), (1, 19), (1, 20), (1, 21), (1, 22), (1, 23),
      (1, 24), (1, 25), (1, 26), (1, 27), (1, 28), (1, 29), (1, 30), (1, 31),
      (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7),
      (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (2, 15),
      (2, 16), (2, 17), (2, 18), (2, 19), (2, 20), (2, 21), (2, 22), (2, 23),
      (2, 24), (2, 25), (2, 26), (2, 27), (2, 28), (2, 29), (2, 30), (2, 31),
      (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7),
      (3, 8), (3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (3, 14), (3, 15),
      (3, 16), (3, 17), (3, 18), (3, 19), (3, 20), (3, 21), (3, 22), (3, 23),
      (3, 24), (3, 25), (3, 26), (3, 27), (3, 28), (3, 29), (3, 30), (3, 31),
      (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7),
      (4, 8), (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (4, 15),
      (4, 16), (4, 17), (4, 18), (4, 19), (4, 20), (4, 21), (4, 22), (4, 23),
      (4, 24), (4, 25), (4, 26), (4, 27), (4, 28), (4, 29), (4, 30), (4, 31),
      (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7),
      (5, 8), (5, 9), (5, 10), (5, 11), (5, 12), (5, 13), (5, 14), (5, 15),
      (5, 16), (5, 17), (5, 18), (5, 19), (5, 20), (5, 21), (5, 22), (5, 23),
      (5, 24), (5, 25), (5, 26), (5, 27), (5, 28), (5, 29), (5, 30), (5, 31),
      (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7),
      (6, 8), (6, 9), (6, 10), (6, 11), (6, 12), (6, 13), (6, 14), (6, 15),
      (6, 16), (6, 17), (6, 18), (6, 19), (6, 20), (6, 21), (6, 22), (6, 23),
      (6, 24), (6, 25), (6, 26), (6, 27), (6, 28), (6, 29), (6, 30), (6, 31),
      (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7),
      (7, 8), (7, 9), (7, 10), (7, 11), (7, 12), (7, 13), (7, 14), (7, 15),
      (7, 16), (7, 17), (7, 18), (7, 19), (7, 20), (7, 21), (7, 22), (7, 23),
      (7, 24), (7, 25), (7, 26), (7, 27), (7, 28), (7, 29), (7, 30), (7, 31),
    ),
    row-labels: ("W0", "W1", "W2", "W3", "W4", "W5", "W6", "W7"),
    title: "warp-tiled block tile（8×64）：每行一个 warp，列方向 32 lane",
  ),
  caption: [
    绿色 = 该 warp 负责的输出列（lane $i$ 算 `col0 = i` 和 `col1 = i + 32`）。
    `threadIdx.y` = warp id，`threadIdx.x` = lane id——CUTLASS 分层的第一 glimpse。
  ],
)

*无信息或为零的 metric：*

- `TC %`：全表 0.0——无 tensor pipe 活动，*符合 scalar FMA 源码*。
- `HBM %`：0.1–0.3%——48 KB 工作集 L2 resident，*不能用来判断 memory-bound vs compute-bound*。
- `warp %`：0.3–1.4%——grid 太小，*不是 kernel 设计错了*。

#insight[
  GEMM ladder 的第一步永远是 *smem tile 让 $A/B$ 在 block 内复用*（tiled 6.24 μs vs naive 8.48 μs）。在 grid 能喂饱 GPU、且走 tensor pipe 之前，讨论 register tile / pipeline 的绝对加速几乎没有意义——warp-tiled 13.76 μs 比 naive 还慢就是证据。
]

#insight[
  *Tile 骨架的收益*和*tensor core 的收益*是两码事。cuBLAS FP16 GEMM 200+ TFLOPS 来自 (a) `mma.sync` tensor pipe、(b) $4096^3$ 填满 108 SM、(c) `cp.async` + swizzle。本章 ladder 教 (a) 之前的结构层；TC % = 0 正是设计如此，不是 benchmark 失败。
]

#warn[
  warp-tiled 比 naive 慢*不说明 warp 分工思路错了*——说明在 $64^3$ 微型问题上 load/sync 开销超过了 register 复用收益。永远用 ncu 验证，不要凭 ladder 位置推断快慢。
]

粗算最快版本（tiled, 6.24 μs）effective TFLOPS：$frac(2 times 64^3, 6.24 times 10^(-6)) approx 53 "GFLOPS"$——A100 FP32 CUDA core 峰值 ~19.5 TFLOPS，利用率 < 0.3%。TC % = 0、warp % < 2%、HBM % < 1% 三个数字一起出现，就是在说：硬件几乎没干活，测到的是 kernel 启动 + L2 命中 + 同步开销，不是生产 GEMM 的性能画像。

== 向量化 load：`float4`

Tiled kernel 的 global → shared 阶段是 bandwidth 瓶颈。Scalar load 每次 4 B，`float4` 每次 16 B：

```cpp
// 假设 a_tile 按 float4 对齐布局
const int4* a4 = reinterpret_cast<const int4*>(a + global_row * k + tile_k);
int4 val = a4[threadIdx.x];  // 一次搬 4 个 float 到 register
// 再拆分到 smem 或用 int4 直接写 smem
```

=== 收益来源

1. *指令数减半*：`LDG.E.128` vs 4× `LDG.E.32`。
2. *Transaction 效率*：一个 warp 32×16B = 512B = 4 个 128B transaction，和 32×4B 一样——但 MSHR 条目更少，latency 更好掩盖。
3. *Smem 写入*：如果 smem layout 允许，`int4` 写 smem 也更快。

=== 约束

- Global 地址 16B 对齐。
- Smem 布局必须配合——如果 `a_tile[ty][tx]` 的 `[ty]` 行不是连续 16B，就不能直接 `float4` 写。
- 尾块处理：$K$ 不是 4 的倍数时需要 scalar epilogue（和 vector add v5 的 tail 同理）。

#warn[
  向量化 load 必须和 smem *layout* 一起设计。先确定 swizzle 后的布局，再决定怎么用 `float4` 搬——顺序反了会反复改。
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

== Tensor Core：`wmma` 与 `mma.sync`

Volta 起 NVIDIA 引入 tensor core：专用硬件做 small tile 矩阵乘加，单条指令完成 $16 times 8 times 16$（FP16→FP32 accumulate）的运算。

=== Shape 语义：m16n8k16

`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` 含义：

- $M = 16, N = 8, K = 16$：一个 warp 一次算 $16 times 8$ 的输出 tile，沿 $K = 16$ 累加。
- `.row.col`：$A$ row-major fragment，$B$ column-major fragment。
- 输入 FP16，累加 FP32。

一个 warp（32 thread）协作填 fragment 并执行一条 `mma.sync`——*每个 thread 不是独立算一个输出*，而是合起来算 16×8 块。

=== WMMA API（高层）

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

__global__ void wmma_gemm(half* a, half* b, float* c, ...) {
  // 一个 warp 一个 wmma tile
  fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
  fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
  fragment<accumulator, 16, 16, 16, float> c_frag;

  load_matrix_sync(a_frag, a_smem_ptr, lda);
  load_matrix_sync(b_frag, b_smem_ptr, ldb);
  fill_fragment(c_frag, 0.0f);

  mma_sync(c_frag, a_frag, b_frag, c_frag);  // C += A×B

  store_matrix_sync(c_smem_ptr, c_frag, ldc, mem_row_major);
}
```

=== MMA PTX（底层，CUTLASS 风格）

```cpp
// 每个 thread 持有 fragment 的一部分（registers）
uint32_t a_frag[4];  // 分布因 layout 而异
uint32_t b_frag[2];
float    c_frag[4];

// load fragment from smem (custom layout)
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {...};

// mma
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
  {c0,c1,c2,c3}, {a0,a1,a2,a3}, {b0,b1}, {c0,c1,c2,c3};
```

#note[
  Fragment layout 是 tensor core 最难的部分——每个 thread 拿哪些元素、smem 怎么 swizzle 才能 `ldmatrix` 无 conflict，需要查 PTX ISA 的 matrix fragment 表。CUTLASS 的 `Layout` 模板就是干这个的。
]

=== FP32 路径

A100 没有原生 FP32 tensor core。FP32 GEMM 要么：

1. 用 TF32（`mma...f32.tf32.tf32.f32`，精度略降）。
2. 用 FP16/BF16 tensor core + FP32 accumulate（混合精度）。
3. 纯 FP32 靠 CUDA core FMA（cuBLAS 的 `Sgemm` 在 Ampere 上仍大量用 CUDA core + 极致 tuning）。

#insight[
  面试说 "我用 tensor core 加速了 GEMM" 时，要说清楚：数据类型（FP16/BF16/TF32）、shape（m16n8k16）、fragment layout、以及 accum 精度（FP32 vs FP16）。
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
    [CTA (block) tile], [决定 smem 大小、grid 划分], [v2 tiled],
    [Warp tile], [warp 间分工、warp-level mma], [v3 warp tile],
    [Thread tile], [register acc、数据复用], [v4 register blocked],
    [Instruction], [`mma.sync` / `ldmatrix`], [tensor core 节],
    [Epilogue], [alpha/beta scaling、bias、activation], [未覆盖（见 MLP 章）],
  ),
  caption: [*Table:* CUTLASS hierarchical GEMM 的五层分解，及本书 ladder 各版本对应的层级。从上到下粒度递减：CTA 决定"这个 block 输出哪一片"，warp 决定"warp 内 32 lane 如何分工做小 mma"，thread 决定寄存器如何缓存数据以复用，instruction 是硬件级 `mma.sync` / `wgmma.mma_async_sync` 的粒度，epilogue 是 accum → output 的写回阶段（活化、scale、bias 融合都在这里）。],
  kind: table,
)

*Observation*：本书 ladder 沿"CTA → warp → thread"这条主路径爬到 v4，*没走 instruction 层*（未用 tensor core PTX）——这正是本章 diag 表 `TC % = 0` 的根源。生产 kernel（cuBLASLt、CUTLASS）几乎所有优化都发生在 instruction 层之下（`ldmatrix.x4`、`mma.sync` shape 选择、swizzle 消除 bank conflict、`cp.async` 双缓冲），本书跳过这层。想真上手 tensor core 请从 CUTLASS 官方 tutorial 起步。

CUTLASS 还处理了：auto-tuning tile 参数、split-K、Stream-K、TMA async copy（SM90）、FP8/Block-scaled 等。本书 ladder 是 CUTLASS 的简化手工版。

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
  Split-K 和 K-dimension tiling（本章 v2 的 `tile_k` 循环）是不同的：后者是在*同一个 block 内*沿 $K$ 累加；前者是*多个 block 分工* $K$ 的不同段，需要 merge。
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
