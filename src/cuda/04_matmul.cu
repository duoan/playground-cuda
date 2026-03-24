#include "common/cuda_utils.cuh"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

// This file follows one teaching rule:
// keep the whole optimization ladder in one place.
//
// Current ladder:
// 1. naive
// 2. shared-memory tiled
// 3. warp-aware / register tiled
// 4. register-blocked
// 5. tensor-core / pipeline-oriented teaching version
//
// The goal is to learn how a GEMM grows from the simplest correct version
// toward a modern high-performance structure. We should only call a version
// "library-level" after measuring it against real reference libraries.

// 这里故意选一个小一点、好理解的 tile。
// 16x16 的 thread block 意味着:
// - 一个 block 有 256 个 thread
// - 一个 block 负责输出矩阵 C 的一个 16x16 小块
constexpr int kTile = 16;

// 第三版会开始显式引入 warp 思维。
// 我们用一个 32x8 的 block:
// - x 方向 32 个 thread，刚好是一整个 warp 的宽度
// - y 方向 8 个 warp
//
// 这样做的好处是:
// - `threadIdx.y` 可以直接理解成 "第几个 warp"
// - `threadIdx.x` 可以直接理解成 warp 内的 lane id
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kWarpTileN = 64;
constexpr int kBlockTileM = kWarpsPerBlock;
constexpr int kBlockTileN = kWarpTileN;
constexpr int kBlockTileK = 16;

// 第四版再往前走一步:
// 不只是 warp 分工，还让每个 thread 在寄存器里计算一个 2x2 小块。
//
// 这就开始接近现代高性能 GEMM 的核心习惯了:
// - block tile
// - shared-memory tile
// - thread/register tile
//
// 这里 block 仍然是 16x16 = 256 个 thread，
// 但每个 thread 算 2x2 个输出，所以整个 block 会覆盖 32x32 输出 tile。
constexpr int kRegBlockThreadsX = 16;
constexpr int kRegBlockThreadsY = 16;
constexpr int kThreadTileM = 2;
constexpr int kThreadTileN = 2;
constexpr int kRegBlockTileM = kRegBlockThreadsY * kThreadTileM;
constexpr int kRegBlockTileN = kRegBlockThreadsX * kThreadTileN;
constexpr int kRegBlockTileK = 16;

// naive matmul kernel:
// 一个 thread 负责输出矩阵里的一个元素 C[row, col]。
// 计算这个元素时，它会把 A 的一整行和 B 的一整列做点积。
// 这是最容易建立直觉的版本，但缺点也很明显:
// 相邻 thread 会反复从 global memory 读取很多重复数据。
__global__ void matmul_naive_kernel(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= m || col >= n) {
    return;
  }

  float acc = 0.0f;
  for (int inner = 0; inner < k; ++inner) {
    acc += a[row * k + inner] * b[inner * n + col];
  }
  c[row * n + col] = acc;
}

// tiled matmul kernel:
// 思想变化只有一句话:
//   先把当前 block 需要的一小块 A 和 B 搬到 shared memory，
//   再让 block 里的 thread 反复复用这些数据。
//
// 对于 matmul 来说，这就是 tile 真正有价值的地方。
__global__ void matmul_tiled_kernel(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k) {
  __shared__ float a_tile[kTile][kTile];
  __shared__ float b_tile[kTile][kTile];

  // 当前 thread 最终要计算的输出坐标。
  const int row = blockIdx.y * kTile + threadIdx.y;
  const int col = blockIdx.x * kTile + threadIdx.x;

  float acc = 0.0f;

  // 沿着 K 维一块一块地推进。
  // 每次循环，当前 block 都会:
  // 1. 读入 A 的一个 tile
  // 2. 读入 B 的一个 tile
  // 3. 在 shared memory 里做一小段乘加
  for (int tile_k = 0; tile_k < k; tile_k += kTile) {
    const int a_col = tile_k + threadIdx.x;
    const int b_row = tile_k + threadIdx.y;

    // 每个 thread 搬一个 A 元素和一个 B 元素进 shared memory。
    // 如果越界，就补 0，这样尾块也能统一处理。
    a_tile[threadIdx.y][threadIdx.x] =
        (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
    b_tile[threadIdx.y][threadIdx.x] =
        (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;

    // 这里必须同步。
    // 因为有些 thread 还在搬数据，不能让别的 thread 提前开始读取 tile。
    __syncthreads();

    // 现在 shared memory 里的 tile 已经准备好了。
    // 当前 thread 取 A 的一行、B 的一列，在 tile 内做一个小点积。
    #pragma unroll
    for (int inner = 0; inner < kTile; ++inner) {
      acc += a_tile[threadIdx.y][inner] * b_tile[inner][threadIdx.x];
    }

    // 这次同步是为了防止有 thread 已经开始写下一轮 tile，
    // 但别的 thread 还没把当前 tile 用完。
    __syncthreads();
  }

  if (row < m && col < n) {
    c[row * n + col] = acc;
  }
}

// warp-aware / register-tiled matmul kernel:
//
// 这个版本比 shared-memory tiled 又往前走了一步。
// 核心变化有两个:
//
// 1. block 内的工作不再只是“256 个 thread 平铺干活”，
//    而是开始显式按 warp 分工
// 2. 每个 thread 不再只算一个输出，而是在寄存器里积累多个输出
//
// 这里的分工方式是:
// - 一个 block 负责 C 的一个 8x64 tile
// - block 里有 8 个 warp
// - 每个 warp 负责这一块里的 1 行 x 64 列
// - warp 里的每个 lane 计算两个输出值:
//   一个是自己的列，一个是自己的列 + 32
//
// 所以这版已经开始有一点 CUTLASS / Triton 常见的味道了:
// block tile -> warp tile -> thread/register tile
__global__ void matmul_warp_tiled_kernel(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k) {
  __shared__ float a_tile[kBlockTileM][kBlockTileK];
  __shared__ float b_tile[kBlockTileK][kBlockTileN];

  const int lane = threadIdx.x;
  const int warp_id = threadIdx.y;
  const int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;

  const int row = blockIdx.y * kBlockTileM + warp_id;
  const int col0 = blockIdx.x * kBlockTileN + lane;
  const int col1 = col0 + kWarpSize;

  // 每个 thread 在寄存器里积累两个输出元素。
  float acc0 = 0.0f;
  float acc1 = 0.0f;

  for (int tile_k = 0; tile_k < k; tile_k += kBlockTileK) {
    // 先协作搬 A 的 tile。
    // A tile 只有 8x16 = 128 个元素，所以前 128 个 thread 各搬一个。
    if (linear_tid < kBlockTileM * kBlockTileK) {
      const int tile_row = linear_tid / kBlockTileK;
      const int tile_col = linear_tid % kBlockTileK;
      const int global_row = blockIdx.y * kBlockTileM + tile_row;
      const int global_col = tile_k + tile_col;

      a_tile[tile_row][tile_col] =
          (global_row < m && global_col < k) ? a[global_row * k + global_col] : 0.0f;
    }

    // 再协作搬 B 的 tile。
    // B tile 是 16x64 = 1024 个元素，256 个 thread 每人搬 4 个。
    for (int idx = linear_tid; idx < kBlockTileK * kBlockTileN; idx += blockDim.x * blockDim.y) {
      const int tile_row = idx / kBlockTileN;
      const int tile_col = idx % kBlockTileN;
      const int global_row = tile_k + tile_row;
      const int global_col = blockIdx.x * kBlockTileN + tile_col;

      b_tile[tile_row][tile_col] =
          (global_row < k && global_col < n) ? b[global_row * n + global_col] : 0.0f;
    }

    __syncthreads();

    // 当前 warp 负责当前 block tile 里的一整行。
    // 这个 warp 里的 lane 会一起扫过 K 方向的小 tile，
    // 并在寄存器里累积两个输出列。
    if (row < m) {
      #pragma unroll
      for (int inner = 0; inner < kBlockTileK; ++inner) {
        const float a_value = a_tile[warp_id][inner];
        acc0 += a_value * b_tile[inner][lane];
        acc1 += a_value * b_tile[inner][lane + kWarpSize];
      }
    }

    __syncthreads();
  }

  if (row < m && col0 < n) {
    c[row * n + col0] = acc0;
  }
  if (row < m && col1 < n) {
    c[row * n + col1] = acc1;
  }
}

// register-blocked matmul kernel:
//
// 这一版的关键变化是:
// - 每个 thread 不再只管 1 个或 2 个输出
// - 每个 thread 会在寄存器里维护一个 2x2 的小累加器
//
// 直觉上你可以这样理解:
// shared memory 是“整个 block 共用的小仓库”
// register 是“每个 thread 自己手里攥着的最快小本子”
//
// 当一个 thread 一次算多个输出时，
// 它能把同一个 A/B 数据片在寄存器里复用更多次。
__global__ void matmul_register_blocked_kernel(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k) {
  __shared__ float a_tile[kRegBlockTileM][kRegBlockTileK];
  __shared__ float b_tile[kRegBlockTileK][kRegBlockTileN];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int row_base = blockIdx.y * kRegBlockTileM + ty * kThreadTileM;
  const int col_base = blockIdx.x * kRegBlockTileN + tx * kThreadTileN;

  float acc[kThreadTileM][kThreadTileN] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

  for (int tile_k = 0; tile_k < k; tile_k += kRegBlockTileK) {
    // 协作搬 A tile:
    // 一个 16x16 thread block 负责把 32x16 的 tile 搬进来。
    // 所以每个 thread 搬两行。
    const int a_col = tile_k + tx;
    const int a_row0 = blockIdx.y * kRegBlockTileM + ty;
    const int a_row1 = a_row0 + kRegBlockThreadsY;

    a_tile[ty][tx] =
        (a_row0 < m && a_col < k) ? a[a_row0 * k + a_col] : 0.0f;
    a_tile[ty + kRegBlockThreadsY][tx] =
        (a_row1 < m && a_col < k) ? a[a_row1 * k + a_col] : 0.0f;

    // 协作搬 B tile:
    // 一个 thread 搬同一行里的两个列位置。
    const int b_row = tile_k + ty;
    const int b_col0 = blockIdx.x * kRegBlockTileN + tx;
    const int b_col1 = b_col0 + kRegBlockThreadsX;

    b_tile[ty][tx] =
        (b_row < k && b_col0 < n) ? b[b_row * n + b_col0] : 0.0f;
    b_tile[ty][tx + kRegBlockThreadsX] =
        (b_row < k && b_col1 < n) ? b[b_row * n + b_col1] : 0.0f;

    __syncthreads();

    // 在当前 K tile 内做乘加。
    // 一个 thread 先从 shared memory 里读出:
    // - 2 个 A 值
    // - 2 个 B 值
    // 然后更新自己手里的 2x2 寄存器小块。
    #pragma unroll
    for (int inner = 0; inner < kRegBlockTileK; ++inner) {
      const float a_frag0 = a_tile[ty * kThreadTileM + 0][inner];
      const float a_frag1 = a_tile[ty * kThreadTileM + 1][inner];
      const float b_frag0 = b_tile[inner][tx * kThreadTileN + 0];
      const float b_frag1 = b_tile[inner][tx * kThreadTileN + 1];

      acc[0][0] += a_frag0 * b_frag0;
      acc[0][1] += a_frag0 * b_frag1;
      acc[1][0] += a_frag1 * b_frag0;
      acc[1][1] += a_frag1 * b_frag1;
    }

    __syncthreads();
  }

  // 最后把 2x2 累加结果写回全局内存。
  for (int i = 0; i < kThreadTileM; ++i) {
    for (int j = 0; j < kThreadTileN; ++j) {
      const int row = row_base + i;
      const int col = col_base + j;
      if (row < m && col < n) {
        c[row * n + col] = acc[i][j];
      }
    }
  }
}

// tensor-core / pipeline-oriented teaching kernel:
//
// 这不是一个真正使用 WMMA / MMA 指令的生产 kernel。
// 为了保证代码对初学者可读、而且不强绑某一代 GPU，这里仍然使用普通 FMA。
//
// 但它会把“下一层结构”明确摆出来:
// - 仍然有 block tile
// - 仍然有 thread/register tile
// - 额外引入 ping-pong shared-memory staging
//
// 真实的 tensor-core 高性能 kernel 往往会更进一步:
// - 使用 tensor core 指令族
// - 使用 async copy / cp.async
// - 让“搬下一块数据”和“算当前块”更深地重叠
//
// 这里的教学目标是:
// 先把 pipeline 的骨架读懂，再去看更硬核的硬件专用实现。
__global__ void matmul_pipeline_teaching_kernel(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k) {
  __shared__ float a_tiles[2][kRegBlockTileM][kRegBlockTileK];
  __shared__ float b_tiles[2][kRegBlockTileK][kRegBlockTileN];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int row_base = blockIdx.y * kRegBlockTileM + ty * kThreadTileM;
  const int col_base = blockIdx.x * kRegBlockTileN + tx * kThreadTileN;

  float acc[kThreadTileM][kThreadTileN] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

  const int num_k_tiles = (k + kRegBlockTileK - 1) / kRegBlockTileK;

  // 先把第 0 个 K tile 搬进 stage 0。
  {
    const int tile_k = 0;
    const int a_col = tile_k + tx;
    const int a_row0 = blockIdx.y * kRegBlockTileM + ty;
    const int a_row1 = a_row0 + kRegBlockThreadsY;

    a_tiles[0][ty][tx] =
        (a_row0 < m && a_col < k) ? a[a_row0 * k + a_col] : 0.0f;
    a_tiles[0][ty + kRegBlockThreadsY][tx] =
        (a_row1 < m && a_col < k) ? a[a_row1 * k + a_col] : 0.0f;

    const int b_row = tile_k + ty;
    const int b_col0 = blockIdx.x * kRegBlockTileN + tx;
    const int b_col1 = b_col0 + kRegBlockThreadsX;

    b_tiles[0][ty][tx] =
        (b_row < k && b_col0 < n) ? b[b_row * n + b_col0] : 0.0f;
    b_tiles[0][ty][tx + kRegBlockThreadsX] =
        (b_row < k && b_col1 < n) ? b[b_row * n + b_col1] : 0.0f;
  }
  __syncthreads();

  for (int tile_index = 0; tile_index < num_k_tiles; ++tile_index) {
    const int stage = tile_index & 1;

    // 用当前 stage 的数据做计算。
    #pragma unroll
    for (int inner = 0; inner < kRegBlockTileK; ++inner) {
      const float a_frag0 = a_tiles[stage][ty * kThreadTileM + 0][inner];
      const float a_frag1 = a_tiles[stage][ty * kThreadTileM + 1][inner];
      const float b_frag0 = b_tiles[stage][inner][tx * kThreadTileN + 0];
      const float b_frag1 = b_tiles[stage][inner][tx * kThreadTileN + 1];

      acc[0][0] += a_frag0 * b_frag0;
      acc[0][1] += a_frag0 * b_frag1;
      acc[1][0] += a_frag1 * b_frag0;
      acc[1][1] += a_frag1 * b_frag1;
    }

    // 预加载下一块到另一个 stage。
    // 真实高性能版本会尽量把这一步和上面的计算重叠。
    // 这里为了教学清楚，还是用同步边界把流程写明白。
    if (tile_index + 1 < num_k_tiles) {
      __syncthreads();
      {
        const int next_stage = stage ^ 1;
        const int tile_k = (tile_index + 1) * kRegBlockTileK;
        const int a_col = tile_k + tx;
        const int a_row0 = blockIdx.y * kRegBlockTileM + ty;
        const int a_row1 = a_row0 + kRegBlockThreadsY;

        a_tiles[next_stage][ty][tx] =
            (a_row0 < m && a_col < k) ? a[a_row0 * k + a_col] : 0.0f;
        a_tiles[next_stage][ty + kRegBlockThreadsY][tx] =
            (a_row1 < m && a_col < k) ? a[a_row1 * k + a_col] : 0.0f;

        const int b_row = tile_k + ty;
        const int b_col0 = blockIdx.x * kRegBlockTileN + tx;
        const int b_col1 = b_col0 + kRegBlockThreadsX;

        b_tiles[next_stage][ty][tx] =
            (b_row < k && b_col0 < n) ? b[b_row * n + b_col0] : 0.0f;
        b_tiles[next_stage][ty][tx + kRegBlockThreadsX] =
            (b_row < k && b_col1 < n) ? b[b_row * n + b_col1] : 0.0f;
      }
      __syncthreads();
    }
  }

  for (int i = 0; i < kThreadTileM; ++i) {
    for (int j = 0; j < kThreadTileN; ++j) {
      const int row = row_base + i;
      const int col = col_base + j;
      if (row < m && col < n) {
        c[row * n + col] = acc[i][j];
      }
    }
  }
}

void fill_inputs(std::vector<float>& a, std::vector<float>& b, int m, int n, int k) {
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < k; ++col) {
      a[row * k + col] = static_cast<float>((row + col) % 7);
    }
  }

  for (int row = 0; row < k; ++row) {
    for (int col = 0; col < n; ++col) {
      b[row * n + col] = static_cast<float>((row * 2 + col) % 5);
    }
  }
}

void matmul_cpu(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::vector<float>& c,
    int m,
    int n,
    int k) {
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float acc = 0.0f;
      for (int inner = 0; inner < k; ++inner) {
        acc += a[row * k + inner] * b[inner * n + col];
      }
      c[row * n + col] = acc;
    }
  }
}

bool check_output(const std::vector<float>& got, const std::vector<float>& expected) {
  for (size_t i = 0; i < got.size(); ++i) {
    if (std::fabs(got[i] - expected[i]) > 1e-4f) {
      std::cerr << "Mismatch at " << i
                << ": got " << got[i]
                << ", expected " << expected[i] << '\n';
      return false;
    }
  }
  return true;
}

void run_naive_matmul(
    const std::vector<float>& host_a,
    const std::vector<float>& host_b,
    std::vector<float>& host_c,
    int m,
    int n,
    int k) {
  float* device_a = nullptr;
  float* device_b = nullptr;
  float* device_c = nullptr;

  const size_t a_bytes = host_a.size() * sizeof(float);
  const size_t b_bytes = host_b.size() * sizeof(float);
  const size_t c_bytes = host_c.size() * sizeof(float);

  CHECK_CUDA(cudaMalloc(&device_a, a_bytes));
  CHECK_CUDA(cudaMalloc(&device_b, b_bytes));
  CHECK_CUDA(cudaMalloc(&device_c, c_bytes));

  CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), a_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), b_bytes, cudaMemcpyHostToDevice));

  // naive 版本的 grid 很直观:
  // blockDim 对应输出矩阵里一个小矩形，
  // gridDim 负责把整个输出矩阵覆盖掉。
  const dim3 block(kTile, kTile);
  const dim3 grid(cuda_utils::ceil_div(n, kTile), cuda_utils::ceil_div(m, kTile));
  matmul_naive_kernel<<<grid, block>>>(device_a, device_b, device_c, m, n, k);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(host_c.data(), device_c, c_bytes, cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(device_a));
  CHECK_CUDA(cudaFree(device_b));
  CHECK_CUDA(cudaFree(device_c));
}

void run_tiled_matmul(
    const std::vector<float>& host_a,
    const std::vector<float>& host_b,
    std::vector<float>& host_c,
    int m,
    int n,
    int k) {
  float* device_a = nullptr;
  float* device_b = nullptr;
  float* device_c = nullptr;

  const size_t a_bytes = host_a.size() * sizeof(float);
  const size_t b_bytes = host_b.size() * sizeof(float);
  const size_t c_bytes = host_c.size() * sizeof(float);

  CHECK_CUDA(cudaMalloc(&device_a, a_bytes));
  CHECK_CUDA(cudaMalloc(&device_b, b_bytes));
  CHECK_CUDA(cudaMalloc(&device_c, c_bytes));

  CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), a_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), b_bytes, cudaMemcpyHostToDevice));

  // tiled 版本在 grid / block 形状上看起来和 naive 一样，
  // 但 block 内部的工作方式已经完全不同：
  // naive 是直接读 global memory，
  // tiled 是先协作搬 tile 到 shared memory 再复用。
  const dim3 block(kTile, kTile);
  const dim3 grid(cuda_utils::ceil_div(n, kTile), cuda_utils::ceil_div(m, kTile));
  matmul_tiled_kernel<<<grid, block>>>(device_a, device_b, device_c, m, n, k);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(host_c.data(), device_c, c_bytes, cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(device_a));
  CHECK_CUDA(cudaFree(device_b));
  CHECK_CUDA(cudaFree(device_c));
}

void run_warp_tiled_matmul(
    const std::vector<float>& host_a,
    const std::vector<float>& host_b,
    std::vector<float>& host_c,
    int m,
    int n,
    int k) {
  float* device_a = nullptr;
  float* device_b = nullptr;
  float* device_c = nullptr;

  const size_t a_bytes = host_a.size() * sizeof(float);
  const size_t b_bytes = host_b.size() * sizeof(float);
  const size_t c_bytes = host_c.size() * sizeof(float);

  CHECK_CUDA(cudaMalloc(&device_a, a_bytes));
  CHECK_CUDA(cudaMalloc(&device_b, b_bytes));
  CHECK_CUDA(cudaMalloc(&device_c, c_bytes));

  CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), a_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), b_bytes, cudaMemcpyHostToDevice));

  // 这一版开始显式按 warp 分工:
  // - 一个 block 是 (32 lanes) x (8 warps)
  // - grid 的一个 block 覆盖输出矩阵里的 8x64 区域
  const dim3 block(kWarpSize, kWarpsPerBlock);
  const dim3 grid(cuda_utils::ceil_div(n, kBlockTileN), cuda_utils::ceil_div(m, kBlockTileM));
  matmul_warp_tiled_kernel<<<grid, block>>>(device_a, device_b, device_c, m, n, k);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(host_c.data(), device_c, c_bytes, cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(device_a));
  CHECK_CUDA(cudaFree(device_b));
  CHECK_CUDA(cudaFree(device_c));
}

void run_register_blocked_matmul(
    const std::vector<float>& host_a,
    const std::vector<float>& host_b,
    std::vector<float>& host_c,
    int m,
    int n,
    int k) {
  float* device_a = nullptr;
  float* device_b = nullptr;
  float* device_c = nullptr;

  const size_t a_bytes = host_a.size() * sizeof(float);
  const size_t b_bytes = host_b.size() * sizeof(float);
  const size_t c_bytes = host_c.size() * sizeof(float);

  CHECK_CUDA(cudaMalloc(&device_a, a_bytes));
  CHECK_CUDA(cudaMalloc(&device_b, b_bytes));
  CHECK_CUDA(cudaMalloc(&device_c, c_bytes));

  CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), a_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), b_bytes, cudaMemcpyHostToDevice));

  // 这一版开始把 thread tile 也显式拿出来。
  // 一个 block 覆盖 32x32 输出 tile，
  // 每个 thread 负责其中一个 2x2 小块。
  const dim3 block(kRegBlockThreadsX, kRegBlockThreadsY);
  const dim3 grid(
      cuda_utils::ceil_div(n, kRegBlockTileN),
      cuda_utils::ceil_div(m, kRegBlockTileM));
  matmul_register_blocked_kernel<<<grid, block>>>(device_a, device_b, device_c, m, n, k);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(host_c.data(), device_c, c_bytes, cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(device_a));
  CHECK_CUDA(cudaFree(device_b));
  CHECK_CUDA(cudaFree(device_c));
}

void run_pipeline_teaching_matmul(
    const std::vector<float>& host_a,
    const std::vector<float>& host_b,
    std::vector<float>& host_c,
    int m,
    int n,
    int k) {
  float* device_a = nullptr;
  float* device_b = nullptr;
  float* device_c = nullptr;

  const size_t a_bytes = host_a.size() * sizeof(float);
  const size_t b_bytes = host_b.size() * sizeof(float);
  const size_t c_bytes = host_c.size() * sizeof(float);

  CHECK_CUDA(cudaMalloc(&device_a, a_bytes));
  CHECK_CUDA(cudaMalloc(&device_b, b_bytes));
  CHECK_CUDA(cudaMalloc(&device_c, c_bytes));

  CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), a_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), b_bytes, cudaMemcpyHostToDevice));

  // 这一步是“朝 tensor-core 时代靠近”的教学版。
  // 重点不是说它已经和 cuBLAS 一样快，
  // 而是让你先把现代高性能 GEMM 最关键的结构读明白:
  // staging、ping-pong buffer、register tile、pipeline skeleton。
  const dim3 block(kRegBlockThreadsX, kRegBlockThreadsY);
  const dim3 grid(
      cuda_utils::ceil_div(n, kRegBlockTileN),
      cuda_utils::ceil_div(m, kRegBlockTileM));
  matmul_pipeline_teaching_kernel<<<grid, block>>>(device_a, device_b, device_c, m, n, k);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(host_c.data(), device_c, c_bytes, cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(device_a));
  CHECK_CUDA(cudaFree(device_b));
  CHECK_CUDA(cudaFree(device_c));
}

void print_top_left_corner(const std::vector<float>& matrix, int rows, int cols, const char* label) {
  std::cout << label << '\n';
  for (int row = 0; row < rows; ++row) {
    std::cout << "  ";
    for (int col = 0; col < cols; ++col) {
      std::cout << matrix[row * cols + col] << ' ';
    }
    std::cout << '\n';
  }
}

}  // namespace

int main() {
  constexpr int m = 64;
  constexpr int n = 64;
  constexpr int k = 64;

  std::vector<float> host_a(m * k);
  std::vector<float> host_b(k * n);
  std::vector<float> host_c_naive(m * n, 0.0f);
  std::vector<float> host_c_tiled(m * n, 0.0f);
  std::vector<float> host_c_warp_tiled(m * n, 0.0f);
  std::vector<float> host_c_register_blocked(m * n, 0.0f);
  std::vector<float> host_c_pipeline_teaching(m * n, 0.0f);
  std::vector<float> reference(m * n, 0.0f);

  fill_inputs(host_a, host_b, m, n, k);
  matmul_cpu(host_a, host_b, reference, m, n, k);

  // 第一步: 先跑 naive 版本。
  // 你要先建立一个最基本的认知：
  // "输出矩阵的每个元素，本质上就是一行乘一列。"
  run_naive_matmul(host_a, host_b, host_c_naive, m, n, k);

  // 第二步: 再跑 tiled 版本。
  // 这一步你要开始关注数据复用：
  // 为什么同一个 A/B 子块会被 block 里的很多 thread 重复使用？
  run_tiled_matmul(host_a, host_b, host_c_tiled, m, n, k);

  // 第三步: 再往前走一步，显式按 warp 分工。
  // 这里你要开始关注:
  // - warp 在 block 里怎么分角色
  // - 为什么一个 thread 会开始计算多个输出
  // - 为什么寄存器 tiling 往往比“一线程一输出”更接近高性能实现
  run_warp_tiled_matmul(host_a, host_b, host_c_warp_tiled, m, n, k);

  // 第四步: 再往前走一步，引入 thread/register tile。
  // 这一步很关键，因为它更像现代高性能 GEMM 的骨架:
  // - block tile
  // - shared-memory tile
  // - register tile
  run_register_blocked_matmul(host_a, host_b, host_c_register_blocked, m, n, k);

  // 第五步: 把 pipeline 结构显式写出来。
  // 这一步的重点不是“已经用上真正 tensor core”，
  // 而是先把 modern GEMM 最重要的框架读懂。
  run_pipeline_teaching_matmul(host_a, host_b, host_c_pipeline_teaching, m, n, k);

  const bool naive_ok = check_output(host_c_naive, reference);
  const bool tiled_ok = check_output(host_c_tiled, reference);
  const bool warp_tiled_ok = check_output(host_c_warp_tiled, reference);
  const bool register_blocked_ok = check_output(host_c_register_blocked, reference);
  const bool pipeline_teaching_ok = check_output(host_c_pipeline_teaching, reference);

  if (!naive_ok || !tiled_ok || !warp_tiled_ok || !register_blocked_ok || !pipeline_teaching_ok) {
    return EXIT_FAILURE;
  }

  std::cout << "App: matmul" << '\n';
  std::cout << "Shape: (" << m << ", " << k << ") x (" << k << ", " << n << ")" << '\n';
  std::cout << '\n';

  std::cout << "[naive version]" << '\n';
  std::cout << "  one thread computes one output element" << '\n';
  print_top_left_corner(host_c_naive, 4, n, "  top-left 4x4 sample:");
  std::cout << '\n';

  std::cout << "[tiled version]" << '\n';
  std::cout << "  tile size: " << kTile << "x" << kTile << '\n';
  std::cout << "  one block computes one output tile" << '\n';
  print_top_left_corner(host_c_tiled, 4, n, "  top-left 4x4 sample:");
  std::cout << '\n';

  std::cout << "[warp/register tiled version]" << '\n';
  std::cout << "  block shape: " << kWarpSize << "x" << kWarpsPerBlock << '\n';
  std::cout << "  block tile: " << kBlockTileM << "x" << kBlockTileN << '\n';
  std::cout << "  each lane computes 2 output values in registers" << '\n';
  print_top_left_corner(host_c_warp_tiled, 4, n, "  top-left 4x4 sample:");
  std::cout << '\n';

  std::cout << "[register-blocked version]" << '\n';
  std::cout << "  block shape: " << kRegBlockThreadsX << "x" << kRegBlockThreadsY << '\n';
  std::cout << "  block tile: " << kRegBlockTileM << "x" << kRegBlockTileN << '\n';
  std::cout << "  thread tile: " << kThreadTileM << "x" << kThreadTileN << '\n';
  std::cout << "  each thread accumulates a 2x2 output tile in registers" << '\n';
  print_top_left_corner(host_c_register_blocked, 4, n, "  top-left 4x4 sample:");
  std::cout << '\n';

  std::cout << "[tensor-core / pipeline-oriented teaching version]" << '\n';
  std::cout << "  block shape: " << kRegBlockThreadsX << "x" << kRegBlockThreadsY << '\n';
  std::cout << "  block tile: " << kRegBlockTileM << "x" << kRegBlockTileN << '\n';
  std::cout << "  structure: ping-pong staging + register tile + pipeline skeleton" << '\n';
  std::cout << "  note: still scalar FMA for readability, not a real WMMA/cuBLAS claim" << '\n';
  print_top_left_corner(host_c_pipeline_teaching, 4, n, "  top-left 4x4 sample:");
  std::cout << '\n';

  std::cout << "All versions: PASS" << '\n';
  return EXIT_SUCCESS;
}
