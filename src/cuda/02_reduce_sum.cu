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
// 1. atomic
// 2. shared-memory tree reduction
// 3. warp-aware reduction
// 4. multi-elements-per-thread / hierarchical reduction

constexpr int kThreadsPerBlock = 256;
constexpr int kWarpSize = 32;
constexpr int kChunkItemsPerThread = 4;

struct ReductionRun {
  float value = 0.0f;
  int stages = 0;
};

// CPU 版本是我们的参考答案。
// 它最慢，但最容易读懂，也最适合拿来检查 GPU 结果是否正确。
double reduce_sum_cpu(const std::vector<float>& values) {
  double total = 0.0;
  for (float value : values) {
    total += static_cast<double>(value);
  }
  return total;
}

void fill_input(std::vector<float>& values) {
  // 这里故意用小整数模式，避免因为浮点加法顺序不同带来太多误差，
  // 这样更适合初学时专注理解规约本身。
  for (int i = 0; i < static_cast<int>(values.size()); ++i) {
    values[i] = static_cast<float>(i % 7);
  }
}

bool check_output(float got, double expected, const char* label) {
  const double diff = std::fabs(static_cast<double>(got) - expected);
  if (diff > 1e-3) {
    std::cerr << label << " mismatch: got " << got
              << ", expected " << expected
              << ", diff " << diff << '\n';
    return false;
  }
  return true;
}

// 第一版: atomic 版本。
// 每个 thread 取到一个输入值，然后直接 atomicAdd 到同一个输出地址。
//
// 这个版本的优点:
// - 最容易想到
// - 代码最短
// - 能立刻说明为什么多个 thread 不能随便同时写同一个位置
//
// 缺点:
// - 所有 thread 都在争同一个地址
// - contention 非常重
// - 一般不会是高性能解法
__global__ void reduce_sum_atomic_kernel(
    const float* input,
    float* output,
    int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  for (int i = index; i < count; i += stride) {
    atomicAdd(output, input[i]);
  }
}

// 第二版: shared memory block reduction。
// 思路是:
// 1. 每个 block 先把自己负责的一段数据规约成一个 partial sum
// 2. 每个 block 输出一个 partial sum
// 3. 再对这些 partial sum 继续规约，直到只剩一个值
//
// 这才是更典型的 CUDA reduction 思路:
// 先 block 内协作，再多阶段缩小问题规模。
__global__ void reduce_sum_shared_kernel(
    const float* input,
    float* block_sums,
    int count) {
  extern __shared__ float shared[];

  const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  shared[threadIdx.x] = (global_index < count) ? input[global_index] : 0.0f;
  __syncthreads();

  // 这是最经典的树形规约。
  // 第一轮: 前 128 个 thread 吃掉后 128 个 thread 的值
  // 第二轮: 前 64 个 thread 再吃掉后 64 个 thread 的值
  // ...
  // 最后 shared[0] 就是这个 block 的和。
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      shared[threadIdx.x] += shared[threadIdx.x + offset];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    block_sums[blockIdx.x] = shared[0];
  }
}

// warp 内规约 helper。
// 一个 warp 有 32 个 thread。warp 内的 thread 天然同步执行，
// 所以很多时候可以直接用 shuffle 指令交换寄存器里的值，
// 不一定要先写 shared memory。
__device__ float warp_reduce_sum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

// 第三版: warp-aware block reduction。
// 这个版本还是 block 输出一个 partial sum，
// 但 block 内部不再把所有步骤都放在 shared memory 里做。
//
// 过程是:
// 1. 每个 warp 先在寄存器里完成自己的小规约
// 2. 每个 warp 的 lane 0 把结果写到 shared memory
// 3. 第一个 warp 再把这些 warp partial sums 规约成 block sum
__global__ void reduce_sum_warp_kernel(
    const float* input,
    float* block_sums,
    int count) {
  __shared__ float warp_sums[kThreadsPerBlock / kWarpSize];

  const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  float value = (global_index < count) ? input[global_index] : 0.0f;

  value = warp_reduce_sum(value);

  const int lane = threadIdx.x % kWarpSize;
  const int warp_id = threadIdx.x / kWarpSize;

  if (lane == 0) {
    warp_sums[warp_id] = value;
  }
  __syncthreads();

  if (warp_id == 0) {
    value = (lane < (blockDim.x / kWarpSize)) ? warp_sums[lane] : 0.0f;
    value = warp_reduce_sum(value);

    if (lane == 0) {
      block_sums[blockIdx.x] = value;
    }
  }
}

// 第四版: multi-elements-per-thread / hierarchical reduction。
//
// 这一步和 shared-memory tree reduction 的区别是:
// - 每个 thread 不再只读一个值
// - 而是先在寄存器里累加多个值
// - 再把每个 thread 的局部和交给 block 内规约
//
// 这在真实 kernel 里很常见，因为它能减少“线程管理成本”，
// 让每个 thread 先做更多有用工作再去同步。
__global__ void reduce_sum_chunked_kernel(
    const float* input,
    float* block_sums,
    int count) {
  extern __shared__ float shared[];

  const int tid = threadIdx.x;
  const int block_start = blockIdx.x * blockDim.x * kChunkItemsPerThread;
  const int thread_start = block_start + tid;

  float local_sum = 0.0f;
  #pragma unroll
  for (int item = 0; item < kChunkItemsPerThread; ++item) {
    const int index = thread_start + item * blockDim.x;
    if (index < count) {
      local_sum += input[index];
    }
  }

  shared[tid] = local_sum;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (tid < offset) {
      shared[tid] += shared[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    block_sums[blockIdx.x] = shared[0];
  }
}

ReductionRun run_atomic_reduction(const std::vector<float>& host_input) {
  float* device_input = nullptr;
  float* device_output = nullptr;

  const int count = static_cast<int>(host_input.size());
  const size_t input_bytes = host_input.size() * sizeof(float);

  CHECK_CUDA(cudaMalloc(&device_input, input_bytes));
  CHECK_CUDA(cudaMalloc(&device_output, sizeof(float)));

  CHECK_CUDA(cudaMemcpy(device_input, host_input.data(), input_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(device_output, 0, sizeof(float)));

  // 这里 blocks 的计算还是元素覆盖逻辑。
  // 因为 atomic 版本的本质就是:
  // "尽量让所有 thread 都去扫输入，然后往同一个地方加。"
  const int blocks = cuda_utils::ceil_div(count, kThreadsPerBlock);
  reduce_sum_atomic_kernel<<<blocks, kThreadsPerBlock>>>(device_input, device_output, count);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());

  ReductionRun result;
  result.stages = 1;
  CHECK_CUDA(cudaMemcpy(&result.value, device_output, sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(device_input));
  CHECK_CUDA(cudaFree(device_output));
  return result;
}

template <typename KernelLauncher>
ReductionRun run_multi_stage_reduction(
    const std::vector<float>& host_input,
    KernelLauncher launch_kernel) {
  float* current_input = nullptr;
  const size_t input_bytes = host_input.size() * sizeof(float);
  CHECK_CUDA(cudaMalloc(&current_input, input_bytes));
  CHECK_CUDA(cudaMemcpy(current_input, host_input.data(), input_bytes, cudaMemcpyHostToDevice));

  int current_count = static_cast<int>(host_input.size());
  int stages = 0;

  // 这里是 reduction 很关键的思想:
  // 一次 kernel launch 并不会直接把 N 个元素变成 1 个元素，
  // 它通常只是把 N 个元素缩成 "每个 block 一个 partial sum"。
  //
  // 所以我们要多次 launch:
  // N -> num_blocks
  // num_blocks -> 更小的 num_blocks
  // ...
  // 直到只剩一个值。
  while (current_count > 1) {
    const int blocks = cuda_utils::ceil_div(current_count, kThreadsPerBlock);
    float* next_output = nullptr;
    CHECK_CUDA(cudaMalloc(&next_output, blocks * sizeof(float)));

    launch_kernel(current_input, next_output, current_count, blocks);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaFree(current_input));
    current_input = next_output;
    current_count = blocks;
    ++stages;
  }

  ReductionRun result;
  result.stages = stages;
  CHECK_CUDA(cudaMemcpy(&result.value, current_input, sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(current_input));
  return result;
}

ReductionRun run_shared_reduction(const std::vector<float>& host_input) {
  return run_multi_stage_reduction(
      host_input,
      [](const float* input, float* output, int count, int blocks) {
        reduce_sum_shared_kernel<<<blocks, kThreadsPerBlock, kThreadsPerBlock * sizeof(float)>>>(
            input, output, count);
      });
}

ReductionRun run_warp_reduction(const std::vector<float>& host_input) {
  return run_multi_stage_reduction(
      host_input,
      [](const float* input, float* output, int count, int blocks) {
        reduce_sum_warp_kernel<<<blocks, kThreadsPerBlock>>>(input, output, count);
      });
}

ReductionRun run_chunked_hierarchical_reduction(const std::vector<float>& host_input) {
  float* current_input = nullptr;
  const size_t input_bytes = host_input.size() * sizeof(float);
  CHECK_CUDA(cudaMalloc(&current_input, input_bytes));
  CHECK_CUDA(cudaMemcpy(current_input, host_input.data(), input_bytes, cudaMemcpyHostToDevice));

  int current_count = static_cast<int>(host_input.size());
  int stages = 0;

  while (current_count > 1) {
    const int block_span = kThreadsPerBlock * kChunkItemsPerThread;
    const int blocks = cuda_utils::ceil_div(current_count, block_span);
    float* next_output = nullptr;
    CHECK_CUDA(cudaMalloc(&next_output, blocks * sizeof(float)));

    reduce_sum_chunked_kernel<<<blocks, kThreadsPerBlock, kThreadsPerBlock * sizeof(float)>>>(
        current_input, next_output, current_count);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaFree(current_input));
    current_input = next_output;
    current_count = blocks;
    ++stages;
  }

  ReductionRun result;
  result.stages = stages;
  CHECK_CUDA(cudaMemcpy(&result.value, current_input, sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(current_input));
  return result;
}

}  // namespace

int main() {
  // 故意不选 2 的整次幂，方便你观察:
  // 真正的 kernel 通常要处理各种“尾巴”。
  constexpr int count = (1 << 20) + 37;

  std::vector<float> host_input(count);
  fill_input(host_input);

  const double expected = reduce_sum_cpu(host_input);

  // 先跑最容易懂、但最不高效的 atomic 版本。
  const ReductionRun atomic_result = run_atomic_reduction(host_input);

  // 再跑 shared memory 版本，理解 block 内树形规约。
  const ReductionRun shared_result = run_shared_reduction(host_input);

  // 最后跑 warp-aware 版本，开始接触 warp shuffle 思路。
  const ReductionRun warp_result = run_warp_reduction(host_input);

  // 再往前走一步，让每个 thread 先累加多个元素。
  // 这会更接近真实高性能 reduction 的组织方式。
  const ReductionRun chunked_result = run_chunked_hierarchical_reduction(host_input);

  const bool atomic_ok = check_output(atomic_result.value, expected, "atomic");
  const bool shared_ok = check_output(shared_result.value, expected, "shared");
  const bool warp_ok = check_output(warp_result.value, expected, "warp");
  const bool chunked_ok = check_output(chunked_result.value, expected, "chunked");

  if (!atomic_ok || !shared_ok || !warp_ok || !chunked_ok) {
    return EXIT_FAILURE;
  }

  std::cout << "App: reduce_sum" << '\n';
  std::cout << "Count: " << count << '\n';
  std::cout << "Reference (CPU): " << expected << '\n';
  std::cout << '\n';

  std::cout << "[atomic version]" << '\n';
  std::cout << "  result: " << atomic_result.value << '\n';
  std::cout << "  stages: " << atomic_result.stages << '\n';
  std::cout << "  idea: all threads atomically add into one output" << '\n';
  std::cout << '\n';

  std::cout << "[shared-memory reduction]" << '\n';
  std::cout << "  result: " << shared_result.value << '\n';
  std::cout << "  stages: " << shared_result.stages << '\n';
  std::cout << "  idea: each block reduces to one partial sum, then repeat" << '\n';
  std::cout << '\n';

  std::cout << "[warp-aware reduction]" << '\n';
  std::cout << "  result: " << warp_result.value << '\n';
  std::cout << "  stages: " << warp_result.stages << '\n';
  std::cout << "  idea: reduce inside warps first, then combine warp partial sums" << '\n';
  std::cout << '\n';

  std::cout << "[chunked hierarchical reduction]" << '\n';
  std::cout << "  result: " << chunked_result.value << '\n';
  std::cout << "  stages: " << chunked_result.stages << '\n';
  std::cout << "  idea: each thread accumulates multiple values before block reduction" << '\n';
  std::cout << '\n';

  std::cout << "All versions: PASS" << '\n';
  return EXIT_SUCCESS;
}
