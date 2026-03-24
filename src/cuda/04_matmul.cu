#include "common/cuda_utils.cuh"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

// 这里故意选一个小一点、好理解的 tile。
// 16x16 的 thread block 意味着:
// - 一个 block 有 256 个 thread
// - 一个 block 负责输出矩阵 C 的一个 16x16 小块
constexpr int kTile = 16;

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

  const bool naive_ok = check_output(host_c_naive, reference);
  const bool tiled_ok = check_output(host_c_tiled, reference);

  if (!naive_ok || !tiled_ok) {
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

  std::cout << "Both versions: PASS" << '\n';
  return EXIT_SUCCESS;
}
