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
// 1. CPU reference
// 2. naive CUDA (one thread per row)
// 3. block/shared-memory row layernorm
// 4. warp-aware / fused layernorm version
//
// RMSNorm is a close sibling, but we keep it out of the main ladder here
// so the file stays focused on the LayerNorm path.

// LayerNorm 是按“行”做的。
// 每一行是一条 token / 一个 feature vector。
constexpr int kThreadsPerBlock = 256;
constexpr int kWarpSize = 32;

struct LayerNormRun {
  int rows = 0;
  int cols = 0;
};

void fill_input(std::vector<float>& input, std::vector<float>& gamma, std::vector<float>& beta,
                int rows, int cols) {
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      input[row * cols + col] = static_cast<float>((row * 11 + col * 5) % 23) * 0.2f - 1.0f;
    }
  }

  for (int col = 0; col < cols; ++col) {
    gamma[col] = 1.0f + 0.01f * static_cast<float>(col % 7);
    beta[col] = 0.05f * static_cast<float>((col % 5) - 2);
  }
}

void layernorm_cpu(
    const std::vector<float>& input,
    const std::vector<float>& gamma,
    const std::vector<float>& beta,
    std::vector<float>& output,
    int rows,
    int cols,
    float eps) {
  for (int row = 0; row < rows; ++row) {
    const float* row_ptr = input.data() + row * cols;
    float* out_ptr = output.data() + row * cols;

    double mean = 0.0;
    for (int col = 0; col < cols; ++col) {
      mean += static_cast<double>(row_ptr[col]);
    }
    mean /= static_cast<double>(cols);

    double var = 0.0;
    for (int col = 0; col < cols; ++col) {
      const double diff = static_cast<double>(row_ptr[col]) - mean;
      var += diff * diff;
    }
    var /= static_cast<double>(cols);

    const double inv_std = 1.0 / std::sqrt(var + static_cast<double>(eps));

    for (int col = 0; col < cols; ++col) {
      const double normalized = (static_cast<double>(row_ptr[col]) - mean) * inv_std;
      out_ptr[col] = static_cast<float>(normalized * gamma[col] + beta[col]);
    }
  }
}

bool check_output(
    const std::vector<float>& got,
    const std::vector<float>& expected,
    int rows,
    int cols) {
  for (int i = 0; i < rows * cols; ++i) {
    if (std::fabs(got[i] - expected[i]) > 1e-4f) {
      std::cerr << "Mismatch at " << i
                << ": got " << got[i]
                << ", expected " << expected[i] << '\n';
      return false;
    }
  }
  return true;
}

// naive layernorm:
// 一个 thread 负责一整行。
// 它先算 mean / var，再把这一行归一化并加上 gamma/beta。
//
// 这个版本最适合先建立直觉:
// LayerNorm 其实就是“对每一行做标准化，再做 affine transform”。
__global__ void layernorm_naive_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int rows,
    int cols,
    float eps) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }

  const float* row_ptr = input + row * cols;
  float* out_ptr = output + row * cols;

  float mean = 0.0f;
  for (int col = 0; col < cols; ++col) {
    mean += row_ptr[col];
  }
  mean /= static_cast<float>(cols);

  float var = 0.0f;
  for (int col = 0; col < cols; ++col) {
    const float diff = row_ptr[col] - mean;
    var += diff * diff;
  }
  var /= static_cast<float>(cols);

  const float inv_std = 1.0f / sqrtf(var + eps);

  for (int col = 0; col < cols; ++col) {
    const float normalized = (row_ptr[col] - mean) * inv_std;
    out_ptr[col] = normalized * gamma[col] + beta[col];
  }
}

// structured layernorm:
// 一个 block 负责一整行，block 里的 thread 协作做 reduction。
//
// 这个版本和 softmax 的 block 版本非常像:
// - 先求 mean
// - 再求 var
// - 最后做 normalization + affine
//
// 对初学者来说，最重要的是把它看成:
// "两个 reduction + 一个逐元素变换"
__global__ void layernorm_block_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int rows,
    int cols,
    float eps) {
  __shared__ float shared_sum[kThreadsPerBlock];
  __shared__ float shared_var[kThreadsPerBlock];

  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const float* row_ptr = input + row * cols;
  float* out_ptr = output + row * cols;

  // 第一步: 算 mean。
  float local_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    local_sum += row_ptr[col];
  }
  shared_sum[threadIdx.x] = local_sum;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + offset];
    }
    __syncthreads();
  }

  const float mean = shared_sum[0] / static_cast<float>(cols);

  // 第二步: 算 variance。
  float local_var = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float diff = row_ptr[col] - mean;
    local_var += diff * diff;
  }
  shared_var[threadIdx.x] = local_var;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      shared_var[threadIdx.x] += shared_var[threadIdx.x + offset];
    }
    __syncthreads();
  }

  const float inv_std = 1.0f / sqrtf(shared_var[0] / static_cast<float>(cols) + eps);

  // 第三步: 写回输出。
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float normalized = (row_ptr[col] - mean) * inv_std;
    out_ptr[col] = normalized * gamma[col] + beta[col];
  }
}

// warp-aware layernorm:
// 这一版把 reduction 的视角再往下收一层。
//
// 做法是:
// 1. 每个 thread 先扫自己负责的列，累积 local sum 和 local sumsq
// 2. 先在 warp 内规约，再把 warp 结果写到 shared memory
// 3. 再让第一个 warp 汇总所有 warp 的结果，得到 mean 和 variance
// 4. 最后所有 thread 重新扫一遍这一行，直接写回归一化后的输出
//
// 相比 block/shared-memory 版本，这里更强调:
// - warp 级别的聚合
// - 一次扫描同时得到 sum 和 sumsq
// - 更像实际高性能 kernel 的组织方式
__device__ float warp_reduce_sum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__global__ void layernorm_warp_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int rows,
    int cols,
    float eps) {
  __shared__ float warp_sums[kThreadsPerBlock / kWarpSize];
  __shared__ float warp_sumsq[kThreadsPerBlock / kWarpSize];
  __shared__ float stats[2];

  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const float* row_ptr = input + row * cols;
  float* out_ptr = output + row * cols;

  // 第一轮扫描: 每个 thread 处理一部分列。
  float local_sum = 0.0f;
  float local_sumsq = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = row_ptr[col];
    local_sum += value;
    local_sumsq += value * value;
  }

  // 先做 warp 内规约。
  local_sum = warp_reduce_sum(local_sum);
  local_sumsq = warp_reduce_sum(local_sumsq);

  const int lane = threadIdx.x % kWarpSize;
  const int warp_id = threadIdx.x / kWarpSize;

  if (lane == 0) {
    warp_sums[warp_id] = local_sum;
    warp_sumsq[warp_id] = local_sumsq;
  }
  __syncthreads();

  // 第一个 warp 汇总所有 warp 的 partial sums。
  if (warp_id == 0) {
    const int num_warps = blockDim.x / kWarpSize;
    float sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
    float sumsq = (lane < num_warps) ? warp_sumsq[lane] : 0.0f;

    sum = warp_reduce_sum(sum);
    sumsq = warp_reduce_sum(sumsq);

    if (lane == 0) {
      const float mean = sum / static_cast<float>(cols);
      const float variance = sumsq / static_cast<float>(cols) - mean * mean;
      stats[0] = mean;
      stats[1] = 1.0f / sqrtf(variance + eps);
    }
  }
  __syncthreads();

  const float mean = stats[0];
  const float inv_std = stats[1];

  // 第二轮扫描: 直接写回输出。
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float normalized = (row_ptr[col] - mean) * inv_std;
    out_ptr[col] = normalized * gamma[col] + beta[col];
  }
}

LayerNormRun run_naive_layernorm(
    const std::vector<float>& host_input,
    const std::vector<float>& host_gamma,
    const std::vector<float>& host_beta,
    std::vector<float>& host_output,
    int rows,
    int cols,
    float eps) {
  float* device_input = nullptr;
  float* device_gamma = nullptr;
  float* device_beta = nullptr;
  float* device_output = nullptr;
  const size_t input_bytes = host_input.size() * sizeof(float);
  const size_t param_bytes = host_gamma.size() * sizeof(float);

  CHECK_CUDA(cudaMalloc(&device_input, input_bytes));
  CHECK_CUDA(cudaMalloc(&device_gamma, param_bytes));
  CHECK_CUDA(cudaMalloc(&device_beta, param_bytes));
  CHECK_CUDA(cudaMalloc(&device_output, input_bytes));

  CHECK_CUDA(cudaMemcpy(device_input, host_input.data(), input_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_gamma, host_gamma.data(), param_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_beta, host_beta.data(), param_bytes, cudaMemcpyHostToDevice));

  const int blocks = cuda_utils::ceil_div(rows, kThreadsPerBlock);
  layernorm_naive_kernel<<<blocks, kThreadsPerBlock>>>(
      device_input, device_gamma, device_beta, device_output, rows, cols, eps);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(host_output.data(), device_output, input_bytes, cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(device_input));
  CHECK_CUDA(cudaFree(device_gamma));
  CHECK_CUDA(cudaFree(device_beta));
  CHECK_CUDA(cudaFree(device_output));

  LayerNormRun run;
  run.rows = rows;
  run.cols = cols;
  return run;
}

LayerNormRun run_block_layernorm(
    const std::vector<float>& host_input,
    const std::vector<float>& host_gamma,
    const std::vector<float>& host_beta,
    std::vector<float>& host_output,
    int rows,
    int cols,
    float eps) {
  float* device_input = nullptr;
  float* device_gamma = nullptr;
  float* device_beta = nullptr;
  float* device_output = nullptr;
  const size_t input_bytes = host_input.size() * sizeof(float);
  const size_t param_bytes = host_gamma.size() * sizeof(float);

  CHECK_CUDA(cudaMalloc(&device_input, input_bytes));
  CHECK_CUDA(cudaMalloc(&device_gamma, param_bytes));
  CHECK_CUDA(cudaMalloc(&device_beta, param_bytes));
  CHECK_CUDA(cudaMalloc(&device_output, input_bytes));

  CHECK_CUDA(cudaMemcpy(device_input, host_input.data(), input_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_gamma, host_gamma.data(), param_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_beta, host_beta.data(), param_bytes, cudaMemcpyHostToDevice));

  const int blocks = rows;
  layernorm_block_kernel<<<blocks, kThreadsPerBlock>>>(
      device_input, device_gamma, device_beta, device_output, rows, cols, eps);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(host_output.data(), device_output, input_bytes, cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(device_input));
  CHECK_CUDA(cudaFree(device_gamma));
  CHECK_CUDA(cudaFree(device_beta));
  CHECK_CUDA(cudaFree(device_output));

  LayerNormRun run;
  run.rows = rows;
  run.cols = cols;
  return run;
}

LayerNormRun run_warp_layernorm(
    const std::vector<float>& host_input,
    const std::vector<float>& host_gamma,
    const std::vector<float>& host_beta,
    std::vector<float>& host_output,
    int rows,
    int cols,
    float eps) {
  float* device_input = nullptr;
  float* device_gamma = nullptr;
  float* device_beta = nullptr;
  float* device_output = nullptr;
  const size_t input_bytes = host_input.size() * sizeof(float);
  const size_t param_bytes = host_gamma.size() * sizeof(float);

  CHECK_CUDA(cudaMalloc(&device_input, input_bytes));
  CHECK_CUDA(cudaMalloc(&device_gamma, param_bytes));
  CHECK_CUDA(cudaMalloc(&device_beta, param_bytes));
  CHECK_CUDA(cudaMalloc(&device_output, input_bytes));

  CHECK_CUDA(cudaMemcpy(device_input, host_input.data(), input_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_gamma, host_gamma.data(), param_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_beta, host_beta.data(), param_bytes, cudaMemcpyHostToDevice));

  const int blocks = rows;
  layernorm_warp_kernel<<<blocks, kThreadsPerBlock>>>(
      device_input, device_gamma, device_beta, device_output, rows, cols, eps);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(host_output.data(), device_output, input_bytes, cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(device_input));
  CHECK_CUDA(cudaFree(device_gamma));
  CHECK_CUDA(cudaFree(device_beta));
  CHECK_CUDA(cudaFree(device_output));

  LayerNormRun run;
  run.rows = rows;
  run.cols = cols;
  return run;
}

void print_row_sample(const std::vector<float>& values, int cols, const char* label) {
  std::cout << label << ":";
  for (int col = 0; col < 8 && col < cols; ++col) {
    std::cout << ' ' << values[col];
  }
  std::cout << " ..." << '\n';
}

}  // namespace

int main() {
  constexpr int rows = 64;
  constexpr int cols = 256;
  constexpr float eps = 1e-5f;

  std::vector<float> host_input(rows * cols);
  std::vector<float> host_gamma(cols);
  std::vector<float> host_beta(cols);
  std::vector<float> host_output_naive(rows * cols, 0.0f);
  std::vector<float> host_output_block(rows * cols, 0.0f);
  std::vector<float> host_output_warp(rows * cols, 0.0f);
  std::vector<float> reference(rows * cols, 0.0f);

  fill_input(host_input, host_gamma, host_beta, rows, cols);
  layernorm_cpu(host_input, host_gamma, host_beta, reference, rows, cols, eps);

  const LayerNormRun naive_run = run_naive_layernorm(
      host_input, host_gamma, host_beta, host_output_naive, rows, cols, eps);
  const LayerNormRun block_run = run_block_layernorm(
      host_input, host_gamma, host_beta, host_output_block, rows, cols, eps);
  const LayerNormRun warp_run = run_warp_layernorm(
      host_input, host_gamma, host_beta, host_output_warp, rows, cols, eps);

  const bool naive_ok = check_output(host_output_naive, reference, rows, cols);
  const bool block_ok = check_output(host_output_block, reference, rows, cols);
  const bool warp_ok = check_output(host_output_warp, reference, rows, cols);

  if (!naive_ok || !block_ok || !warp_ok) {
    return EXIT_FAILURE;
  }

  std::cout << "App: layernorm" << '\n';
  std::cout << "Shape: " << rows << " x " << cols << '\n';
  std::cout << '\n';

  std::cout << "[naive version]" << '\n';
  std::cout << "  one thread computes one whole row" << '\n';
  std::cout << "  rows: " << naive_run.rows << ", cols: " << naive_run.cols << '\n';
  print_row_sample(host_output_naive, cols, "  sample row");
  std::cout << '\n';

  std::cout << "[block version]" << '\n';
  std::cout << "  one block computes one whole row" << '\n';
  std::cout << "  rows: " << block_run.rows << ", cols: " << block_run.cols << '\n';
  print_row_sample(host_output_block, cols, "  sample row");
  std::cout << '\n';

  std::cout << "[warp-aware version]" << '\n';
  std::cout << "  warp-level reduction with fused sum/sumsq" << '\n';
  std::cout << "  rows: " << warp_run.rows << ", cols: " << warp_run.cols << '\n';
  print_row_sample(host_output_warp, cols, "  sample row");
  std::cout << '\n';

  std::cout << "All versions: PASS" << '\n';
  return EXIT_SUCCESS;
}
