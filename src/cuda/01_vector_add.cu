#include "common/cuda_utils.cuh"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

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
    const std::vector<float>& a,
    const std::vector<float>& b,
    const std::vector<float>& c) {
  for (int i = 0; i < static_cast<int>(c.size()); ++i) {
    const float expected = a[i] + b[i];
    if (std::fabs(c[i] - expected) > 1e-5f) {
      std::cerr << "Mismatch at " << i
                << ": got " << c[i]
                << ", expected " << expected << '\n';
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

  fill_inputs(host_a, host_b);

  // 第一步: 跑最朴素的版本，建立对 CUDA 执行模型的基本感觉。
  run_naive_vector_add(host_a, host_b, host_c_naive);

  // 第二步: 跑 tile 风格版本。
  // 这一步的重点不是 "一定更快"，而是开始建立
  // "一个 block 可以系统性地处理一整块数据" 的思维。
  run_tiled_vector_add(host_a, host_b, host_c_tiled);

  const bool naive_ok = check_output(host_a, host_b, host_c_naive);
  const bool tiled_ok = check_output(host_a, host_b, host_c_tiled);

  if (!naive_ok || !tiled_ok) {
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

  std::cout << "Both versions: PASS" << '\n';
  return EXIT_SUCCESS;
}
