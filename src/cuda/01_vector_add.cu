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
// 2. block-tiled / multi-items-per-thread
// 3. grid-stride
// 4. vectorized load/store

// 先给一个容易理解的配置。
// naive 版本里，一个 thread 只处理一个元素。
constexpr int kNaiveThreadsPerBlock = 256;

// tile 风格版本里，我们把一个 block 负责的一整段数据叫做一个 tile。
// 这里每个 thread 处理 4 个元素，所以一个 block 处理的 tile 大小是:
//   128 threads * 4 items/thread = 512 items
constexpr int kTiledThreadsPerBlock = 128;
constexpr int kItemsPerThread = 4;
constexpr int kTileSize = kTiledThreadsPerBlock * kItemsPerThread;

// naive kernel:
// 一个 thread 对应一个输出元素。
// 这是学习 CUDA 最常见的第一步。
__global__ void vector_add_naive_kernel(
    const float* a,
    const float* b,
    float* c,
    int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    c[index] = a[index] + b[index];
  }
}

// CPU reference:
// 这个版本最慢，但它是我们最可靠的答案。
// 先把正确性放到 CPU 上确认，再去看 GPU 怎么一步步优化。
void vector_add_cpu(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::vector<float>& c) {
  for (int i = 0; i < static_cast<int>(c.size()); ++i) {
    c[i] = a[i] + b[i];
  }
}

// grid-stride kernel:
// 这里的重点是让一个线程不只负责一个元素，
// 而是沿着一个固定 stride 扫过整个数组。
//
// 这个写法很重要，因为它是 CUDA 里非常常见的基础模板。
__global__ void vector_add_grid_stride_kernel(
    const float* a,
    const float* b,
    float* c,
    int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  for (int i = index; i < count; i += stride) {
    c[i] = a[i] + b[i];
  }
}

// vectorized kernel:
// 这里把连续的 4 个 float 作为一个 float4 来处理。
// 这样可以更自然地练习“更宽的内存访问”。
//
// 注意:
// - 这个 kernel 只处理完整的 float4 前缀
// - 尾巴部分仍然可以交给 scalar kernel 处理
__global__ void vector_add_vectorized_kernel(
    const float4* a4,
    const float4* b4,
    float4* c4,
    int vector_count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < vector_count) {
    const float4 lhs = a4[index];
    const float4 rhs = b4[index];
    c4[index] = make_float4(
        lhs.x + rhs.x,
        lhs.y + rhs.y,
        lhs.z + rhs.z,
        lhs.w + rhs.w);
  }
}

// tiled kernel:
// 这里的 "tile" 不是 shared memory tile，而是 thread block 负责的
// 一段连续数据。vector add 计算非常简单，通常不需要 shared memory，
// 但它很适合拿来练习 "一个 block 怎么分工处理一块数据" 这个思路。
__global__ void vector_add_tiled_kernel(
    const float* a,
    const float* b,
    float* c,
    int count) {
  // blockIdx.x 决定当前 block 处理第几个 tile。
  const int tile_start = blockIdx.x * kTileSize;

  // 在当前 tile 内，threadIdx.x 决定 thread 的起始位置。
  const int thread_base = tile_start + threadIdx.x;

  // 同一个 thread 连续处理多个元素。
  // item=0 时，所有 thread 访问 tile 的前 128 个元素。
  // item=1 时，所有 thread 再访问后 128 个元素。
  // 这样每一轮访问在 warp 内仍然是连续的，内存访问模式比较规整。
  #pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int index = thread_base + item * kTiledThreadsPerBlock;
    if (index < count) {
      c[index] = a[index] + b[index];
    }
  }
}

void fill_inputs(std::vector<float>& a, std::vector<float>& b) {
  for (int i = 0; i < static_cast<int>(a.size()); ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(2 * i);
  }
}

bool check_output(
    const std::vector<float>& got,
    const std::vector<float>& expected) {
  for (int i = 0; i < static_cast<int>(got.size()); ++i) {
    if (std::fabs(got[i] - expected[i]) > 1e-5f) {
      std::cerr << "Mismatch at " << i
                << ": got " << got[i]
                << ", expected " << expected[i] << '\n';
      return false;
    }
  }
  return true;
}

void run_naive_vector_add(
    const std::vector<float>& host_a,
    const std::vector<float>& host_b,
    std::vector<float>& host_c) {
  float* device_a = nullptr;
  float* device_b = nullptr;
  float* device_c = nullptr;

  const int count = static_cast<int>(host_a.size());
  const size_t bytes = host_a.size() * sizeof(float);

  CHECK_CUDA(cudaMalloc(&device_a, bytes));
  CHECK_CUDA(cudaMalloc(&device_b, bytes));
  CHECK_CUDA(cudaMalloc(&device_c, bytes));

  CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), bytes, cudaMemcpyHostToDevice));

  // naive 版本里，grid 的计算逻辑也最直接:
  // 需要多少个元素，就开多少个 thread；
  // 如果一个 block 放不下，就用多个 block。
  const int blocks = cuda_utils::ceil_div(count, kNaiveThreadsPerBlock);
  vector_add_naive_kernel<<<blocks, kNaiveThreadsPerBlock>>>(device_a, device_b, device_c, count);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(host_c.data(), device_c, bytes, cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(device_a));
  CHECK_CUDA(cudaFree(device_b));
  CHECK_CUDA(cudaFree(device_c));
}

void run_tiled_vector_add(
    const std::vector<float>& host_a,
    const std::vector<float>& host_b,
    std::vector<float>& host_c) {
  float* device_a = nullptr;
  float* device_b = nullptr;
  float* device_c = nullptr;

  const int count = static_cast<int>(host_a.size());
  const size_t bytes = host_a.size() * sizeof(float);

  CHECK_CUDA(cudaMalloc(&device_a, bytes));
  CHECK_CUDA(cudaMalloc(&device_b, bytes));
  CHECK_CUDA(cudaMalloc(&device_c, bytes));

  CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), bytes, cudaMemcpyHostToDevice));

  // tiled 版本里，grid 的单位变成 "tile 数量"。
  // 因为一个 block 一次不是只做 128 个元素，而是做 512 个元素。
  const int blocks = cuda_utils::ceil_div(count, kTileSize);
  vector_add_tiled_kernel<<<blocks, kTiledThreadsPerBlock>>>(device_a, device_b, device_c, count);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(host_c.data(), device_c, bytes, cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(device_a));
  CHECK_CUDA(cudaFree(device_b));
  CHECK_CUDA(cudaFree(device_c));
}

void run_grid_stride_vector_add(
    const std::vector<float>& host_a,
    const std::vector<float>& host_b,
    std::vector<float>& host_c) {
  float* device_a = nullptr;
  float* device_b = nullptr;
  float* device_c = nullptr;

  const int count = static_cast<int>(host_a.size());
  const size_t bytes = host_a.size() * sizeof(float);

  CHECK_CUDA(cudaMalloc(&device_a, bytes));
  CHECK_CUDA(cudaMalloc(&device_b, bytes));
  CHECK_CUDA(cudaMalloc(&device_c, bytes));

  CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), bytes, cudaMemcpyHostToDevice));

  // grid-stride 版本里，每个 thread 会沿着固定 stride 扫过整个数组。
  // 这是一种更通用的写法，也更接近工程里常见的 CUDA 模板。
  const int blocks = cuda_utils::ceil_div(count, kNaiveThreadsPerBlock);
  vector_add_grid_stride_kernel<<<blocks, kNaiveThreadsPerBlock>>>(device_a, device_b, device_c, count);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(host_c.data(), device_c, bytes, cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(device_a));
  CHECK_CUDA(cudaFree(device_b));
  CHECK_CUDA(cudaFree(device_c));
}

void run_vectorized_vector_add(
    const std::vector<float>& host_a,
    const std::vector<float>& host_b,
    std::vector<float>& host_c) {
  float* device_a = nullptr;
  float* device_b = nullptr;
  float* device_c = nullptr;

  const int count = static_cast<int>(host_a.size());
  const size_t bytes = host_a.size() * sizeof(float);

  CHECK_CUDA(cudaMalloc(&device_a, bytes));
  CHECK_CUDA(cudaMalloc(&device_b, bytes));
  CHECK_CUDA(cudaMalloc(&device_c, bytes));

  CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), bytes, cudaMemcpyHostToDevice));

  const int vector_count = count / 4;
  const int vectorized_count = vector_count * 4;

  if (vector_count > 0) {
    const int blocks = cuda_utils::ceil_div(vector_count, kNaiveThreadsPerBlock);
    const auto* a4 = reinterpret_cast<const float4*>(device_a);
    const auto* b4 = reinterpret_cast<const float4*>(device_b);
    auto* c4 = reinterpret_cast<float4*>(device_c);
    vector_add_vectorized_kernel<<<blocks, kNaiveThreadsPerBlock>>>(a4, b4, c4, vector_count);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  // 尾巴部分单独处理。
  // 这一步让 vectorized kernel 更容易理解：
  // 主体处理宽访问，剩下的少量元素交给 scalar 路径。
  if (vectorized_count < count) {
    const int tail_count = count - vectorized_count;
    const int tail_blocks = cuda_utils::ceil_div(tail_count, kNaiveThreadsPerBlock);
    vector_add_grid_stride_kernel<<<tail_blocks, kNaiveThreadsPerBlock>>>(
        device_a + vectorized_count,
        device_b + vectorized_count,
        device_c + vectorized_count,
        tail_count);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  CHECK_CUDA(cudaMemcpy(host_c.data(), device_c, bytes, cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(device_a));
  CHECK_CUDA(cudaFree(device_b));
  CHECK_CUDA(cudaFree(device_c));
}

void print_sample(const std::vector<float>& values, const char* label) {
  std::cout << label << ":";
  for (int i = 0; i < 8 && i < static_cast<int>(values.size()); ++i) {
    std::cout << ' ' << values[i];
  }
  std::cout << " ..." << '\n';
}

}  // namespace

int main() {
  constexpr int count = 1 << 20;

  std::vector<float> host_a(count);
  std::vector<float> host_b(count);
  std::vector<float> host_c_naive(count, 0.0f);
  std::vector<float> host_c_tiled(count, 0.0f);
  std::vector<float> host_c_grid_stride(count, 0.0f);
  std::vector<float> host_c_vectorized(count, 0.0f);
  std::vector<float> reference(count, 0.0f);

  fill_inputs(host_a, host_b);
  vector_add_cpu(host_a, host_b, reference);

  // 第一步: 跑最朴素的版本，建立对 CUDA 执行模型的基本感觉。
  run_naive_vector_add(host_a, host_b, host_c_naive);

  // 第二步: 跑 tile 风格版本。
  // 这一步的重点不是 "一定更快"，而是开始建立
  // "一个 block 可以系统性地处理一整块数据" 的思维。
  run_tiled_vector_add(host_a, host_b, host_c_tiled);

  // 第三步: 让每个 thread 沿着 stride 扫过整个数组。
  // 这是一种更常见、更通用的 CUDA 写法。
  run_grid_stride_vector_add(host_a, host_b, host_c_grid_stride);

  // 第四步: 用 float4 做更宽的 load/store。
  // 这一步更偏向“内存访问方式”的优化思路。
  run_vectorized_vector_add(host_a, host_b, host_c_vectorized);

  const bool naive_ok = check_output(host_c_naive, reference);
  const bool tiled_ok = check_output(host_c_tiled, reference);
  const bool grid_stride_ok = check_output(host_c_grid_stride, reference);
  const bool vectorized_ok = check_output(host_c_vectorized, reference);

  if (!naive_ok || !tiled_ok || !grid_stride_ok || !vectorized_ok) {
    return EXIT_FAILURE;
  }

  std::cout << "App: vector_add" << '\n';
  std::cout << "Count: " << count << '\n';
  std::cout << '\n';

  std::cout << "[naive version]" << '\n';
  std::cout << "  threads per block: " << kNaiveThreadsPerBlock << '\n';
  std::cout << "  one thread computes one element" << '\n';
  print_sample(host_c_naive, "  sample output");
  std::cout << '\n';

  std::cout << "[tiled version]" << '\n';
  std::cout << "  threads per block: " << kTiledThreadsPerBlock << '\n';
  std::cout << "  items per thread: " << kItemsPerThread << '\n';
  std::cout << "  tile size: " << kTileSize << '\n';
  print_sample(host_c_tiled, "  sample output");
  std::cout << '\n';

  std::cout << "[grid-stride version]" << '\n';
  std::cout << "  threads per block: " << kNaiveThreadsPerBlock << '\n';
  std::cout << "  one thread strides across the array" << '\n';
  print_sample(host_c_grid_stride, "  sample output");
  std::cout << '\n';

  std::cout << "[vectorized version]" << '\n';
  std::cout << "  vector width: 4 floats" << '\n';
  std::cout << "  body uses float4 load/store" << '\n';
  print_sample(host_c_vectorized, "  sample output");
  std::cout << '\n';

  std::cout << "All versions: PASS" << '\n';
  return EXIT_SUCCESS;
}
