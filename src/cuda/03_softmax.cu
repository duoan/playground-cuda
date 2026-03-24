#include "common/cuda_utils.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

namespace {

// This file follows one teaching rule:
// keep the whole optimization ladder in one place.
//
// Current ladder:
// 1. CPU reference
// 2. naive CUDA (one thread per row)
// 3. block/shared-memory row softmax
// 4. online softmax
// 5. masked softmax
// 6. causal softmax

// 软最大值是按“行”做的，所以我们把输入想成一个二维矩阵:
// [rows, cols]
//
// 每一行独立做 softmax:
//   y[i] = exp(x[i] - max(x)) / sum_j exp(x[j] - max(x))
//
// 这里故意选一个比较好理解的 block 大小。
constexpr int kThreadsPerBlock = 256;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

struct SoftmaxRun {
  int rows = 0;
  int cols = 0;
};

void fill_input(std::vector<float>& input, int rows, int cols) {
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      // 这里不用太复杂的随机数，直接用一个可复现的模式。
      // 这样 CPU reference 和 GPU 输出更容易对比。
      input[row * cols + col] = static_cast<float>((row * 13 + col * 7) % 31) * 0.1f;
    }
  }
}

void fill_mask(std::vector<int>& mask, int rows, int cols) {
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      // Keep the first column unmasked so every row has at least one valid value.
      mask[row * cols + col] = (col == 0 || ((row + col) % 5 != 0)) ? 1 : 0;
    }
  }
}

void softmax_cpu(
    const std::vector<float>& input,
    std::vector<float>& output,
    int rows,
    int cols) {
  for (int row = 0; row < rows; ++row) {
    const float* row_ptr = input.data() + row * cols;
    float* out_ptr = output.data() + row * cols;

    // 第一步: 找最大值。
    // 这样做是为了数值稳定性。
    float max_value = row_ptr[0];
    for (int col = 1; col < cols; ++col) {
      max_value = std::max(max_value, row_ptr[col]);
    }

    // 第二步: 计算 exp(x - max) 的总和。
    double sum = 0.0;
    for (int col = 0; col < cols; ++col) {
      sum += std::exp(static_cast<double>(row_ptr[col] - max_value));
    }

    // 第三步: 归一化。
    for (int col = 0; col < cols; ++col) {
      out_ptr[col] = static_cast<float>(
          std::exp(static_cast<double>(row_ptr[col] - max_value)) / sum);
    }
  }
}

void softmax_online_cpu(
    const std::vector<float>& input,
    std::vector<float>& output,
    int rows,
    int cols) {
  for (int row = 0; row < rows; ++row) {
    const float* row_ptr = input.data() + row * cols;
    float* out_ptr = output.data() + row * cols;

    float row_max = kNegInf;
    double row_sum = 0.0;
    for (int col = 0; col < cols; ++col) {
      const float x = row_ptr[col];
      const float new_row_max = std::max(row_max, x);
      row_sum = row_sum * std::exp(static_cast<double>(row_max - new_row_max)) +
                std::exp(static_cast<double>(x - new_row_max));
      row_max = new_row_max;
    }

    for (int col = 0; col < cols; ++col) {
      out_ptr[col] = static_cast<float>(
          std::exp(static_cast<double>(row_ptr[col] - row_max)) / row_sum);
    }
  }
}

void softmax_masked_cpu(
    const std::vector<float>& input,
    const std::vector<int>& mask,
    std::vector<float>& output,
    int rows,
    int cols) {
  for (int row = 0; row < rows; ++row) {
    const float* row_ptr = input.data() + row * cols;
    const int* mask_ptr = mask.data() + row * cols;
    float* out_ptr = output.data() + row * cols;

    float row_max = kNegInf;
    double row_sum = 0.0;
    for (int col = 0; col < cols; ++col) {
      if (mask_ptr[col] == 0) {
        continue;
      }
      const float x = row_ptr[col];
      const float new_row_max = std::max(row_max, x);
      row_sum = row_sum * std::exp(static_cast<double>(row_max - new_row_max)) +
                std::exp(static_cast<double>(x - new_row_max));
      row_max = new_row_max;
    }

    for (int col = 0; col < cols; ++col) {
      if (mask_ptr[col] == 0) {
        out_ptr[col] = 0.0f;
      } else {
        out_ptr[col] = static_cast<float>(
            std::exp(static_cast<double>(row_ptr[col] - row_max)) / row_sum);
      }
    }
  }
}

void softmax_causal_cpu(
    const std::vector<float>& input,
    std::vector<float>& output,
    int rows,
    int cols) {
  for (int row = 0; row < rows; ++row) {
    const float* row_ptr = input.data() + row * cols;
    float* out_ptr = output.data() + row * cols;

    float row_max = kNegInf;
    double row_sum = 0.0;
    for (int col = 0; col <= row && col < cols; ++col) {
      const float x = row_ptr[col];
      const float new_row_max = std::max(row_max, x);
      row_sum = row_sum * std::exp(static_cast<double>(row_max - new_row_max)) +
                std::exp(static_cast<double>(x - new_row_max));
      row_max = new_row_max;
    }

    for (int col = 0; col < cols; ++col) {
      if (col > row) {
        out_ptr[col] = 0.0f;
      } else {
        out_ptr[col] = static_cast<float>(
            std::exp(static_cast<double>(row_ptr[col] - row_max)) / row_sum);
      }
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

// naive softmax:
// 一个 thread 负责一整行。
// 它会自己扫完整行，找到 max，算 sum，再算输出。
//
// 这个版本非常适合初学者，因为它的逻辑和 CPU 几乎一模一样。
// 缺点也很明显:
// - 一行只用一个 thread，GPU 并行度很差
// - 对长行来说，单线程做太多工作
__global__ void softmax_naive_kernel(
    const float* input,
    float* output,
    int rows,
    int cols) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }

  const float* row_ptr = input + row * cols;
  float* out_ptr = output + row * cols;

  float max_value = row_ptr[0];
  for (int col = 1; col < cols; ++col) {
    max_value = fmaxf(max_value, row_ptr[col]);
  }

  double sum = 0.0;
  for (int col = 0; col < cols; ++col) {
    sum += std::exp(static_cast<double>(row_ptr[col] - max_value));
  }

  for (int col = 0; col < cols; ++col) {
    out_ptr[col] = static_cast<float>(
        std::exp(static_cast<double>(row_ptr[col] - max_value)) / sum);
  }
}

// structured softmax:
// 一个 block 负责一整行。
// block 内的 thread 协作完成:
// 1. 找 row max
// 2. 算 exp sum
// 3. 写回输出
//
// 这里的关键不是“炫技”，而是让你看到:
// softmax 本质上是“两个 reduction + 一个 normalize”。
__global__ void softmax_block_kernel(
    const float* input,
    float* output,
    int rows,
    int cols) {
  __shared__ float shared_max[kThreadsPerBlock];
  __shared__ float shared_sum[kThreadsPerBlock];

  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const float* row_ptr = input + row * cols;
  float* out_ptr = output + row * cols;

  // 每个 thread 先扫自己负责的一部分列。
  float local_max = kNegInf;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    local_max = fmaxf(local_max, row_ptr[col]);
  }
  shared_max[threadIdx.x] = local_max;
  __syncthreads();

  // 第一次规约: 找整行的最大值。
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + offset]);
    }
    __syncthreads();
  }

  const float row_max = shared_max[0];

  // 第二步: 每个 thread 计算自己负责位置的 exp sum 部分。
  float local_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    local_sum += expf(row_ptr[col] - row_max);
  }
  shared_sum[threadIdx.x] = local_sum;
  __syncthreads();

  // 第二次规约: 求整行的 sum。
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + offset];
    }
    __syncthreads();
  }

  const float row_sum = shared_sum[0];

  // 第三步: 归一化写回。
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    out_ptr[col] = expf(row_ptr[col] - row_max) / row_sum;
  }
}

// online softmax:
// 这一版不再强调 block 协作，而是强调“如何在线维护 row_max 和 row_sum”。
// 这正是后面 FlashAttention 里最重要的数值技巧之一。
__global__ void softmax_online_kernel(
    const float* input,
    float* output,
    int rows,
    int cols) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }

  const float* row_ptr = input + row * cols;
  float* out_ptr = output + row * cols;

  float row_max = kNegInf;
  double row_sum = 0.0;
  for (int col = 0; col < cols; ++col) {
    const float x = row_ptr[col];
    const float new_row_max = fmaxf(row_max, x);
    row_sum = row_sum * std::exp(static_cast<double>(row_max - new_row_max)) +
              std::exp(static_cast<double>(x - new_row_max));
    row_max = new_row_max;
  }

  for (int col = 0; col < cols; ++col) {
    out_ptr[col] = static_cast<float>(
        std::exp(static_cast<double>(row_ptr[col] - row_max)) / row_sum);
  }
}

// masked softmax:
// 和 online softmax 的结构很像，只是加入了一个显式 mask。
// mask=0 的位置不会参与 max/sum 统计，最后输出也会被置零。
__global__ void softmax_masked_kernel(
    const float* input,
    const int* mask,
    float* output,
    int rows,
    int cols) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }

  const float* row_ptr = input + row * cols;
  const int* mask_ptr = mask + row * cols;
  float* out_ptr = output + row * cols;

  float row_max = kNegInf;
  double row_sum = 0.0;
  for (int col = 0; col < cols; ++col) {
    if (mask_ptr[col] == 0) {
      continue;
    }
    const float x = row_ptr[col];
    const float new_row_max = fmaxf(row_max, x);
    row_sum = row_sum * std::exp(static_cast<double>(row_max - new_row_max)) +
              std::exp(static_cast<double>(x - new_row_max));
    row_max = new_row_max;
  }

  for (int col = 0; col < cols; ++col) {
    if (mask_ptr[col] == 0) {
      out_ptr[col] = 0.0f;
    } else {
      out_ptr[col] = static_cast<float>(
          std::exp(static_cast<double>(row_ptr[col] - row_max)) / row_sum);
    }
  }
}

// causal softmax:
// 这是 attention 里最常见的上三角 mask 特例。
// 对于第 row 行，只允许看见 col <= row 的位置。
__global__ void softmax_causal_kernel(
    const float* input,
    float* output,
    int rows,
    int cols) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }

  const float* row_ptr = input + row * cols;
  float* out_ptr = output + row * cols;

  float row_max = kNegInf;
  double row_sum = 0.0;
  for (int col = 0; col <= row && col < cols; ++col) {
    const float x = row_ptr[col];
    const float new_row_max = fmaxf(row_max, x);
    row_sum = row_sum * std::exp(static_cast<double>(row_max - new_row_max)) +
              std::exp(static_cast<double>(x - new_row_max));
    row_max = new_row_max;
  }

  for (int col = 0; col < cols; ++col) {
    if (col > row) {
      out_ptr[col] = 0.0f;
    } else {
      out_ptr[col] = static_cast<float>(
          std::exp(static_cast<double>(row_ptr[col] - row_max)) / row_sum);
    }
  }
}

SoftmaxRun run_naive_softmax(
    const std::vector<float>& host_input,
    std::vector<float>& host_output,
    int rows,
    int cols) {
  float* device_input = nullptr;
  float* device_output = nullptr;
  const size_t bytes = host_input.size() * sizeof(float);

  CHECK_CUDA(cudaMalloc(&device_input, bytes));
  CHECK_CUDA(cudaMalloc(&device_output, bytes));
  CHECK_CUDA(cudaMemcpy(device_input, host_input.data(), bytes, cudaMemcpyHostToDevice));

  const int blocks = cuda_utils::ceil_div(rows, kThreadsPerBlock);
  softmax_naive_kernel<<<blocks, kThreadsPerBlock>>>(device_input, device_output, rows, cols);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(host_output.data(), device_output, bytes, cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(device_input));
  CHECK_CUDA(cudaFree(device_output));

  SoftmaxRun run;
  run.rows = rows;
  run.cols = cols;
  return run;
}

SoftmaxRun run_block_softmax(
    const std::vector<float>& host_input,
    std::vector<float>& host_output,
    int rows,
    int cols) {
  float* device_input = nullptr;
  float* device_output = nullptr;
  const size_t bytes = host_input.size() * sizeof(float);

  CHECK_CUDA(cudaMalloc(&device_input, bytes));
  CHECK_CUDA(cudaMalloc(&device_output, bytes));
  CHECK_CUDA(cudaMemcpy(device_input, host_input.data(), bytes, cudaMemcpyHostToDevice));

  const int blocks = rows;
  softmax_block_kernel<<<blocks, kThreadsPerBlock>>>(device_input, device_output, rows, cols);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(host_output.data(), device_output, bytes, cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(device_input));
  CHECK_CUDA(cudaFree(device_output));

  SoftmaxRun run;
  run.rows = rows;
  run.cols = cols;
  return run;
}

SoftmaxRun run_online_softmax(
    const std::vector<float>& host_input,
    std::vector<float>& host_output,
    int rows,
    int cols) {
  float* device_input = nullptr;
  float* device_output = nullptr;
  const size_t bytes = host_input.size() * sizeof(float);

  CHECK_CUDA(cudaMalloc(&device_input, bytes));
  CHECK_CUDA(cudaMalloc(&device_output, bytes));
  CHECK_CUDA(cudaMemcpy(device_input, host_input.data(), bytes, cudaMemcpyHostToDevice));

  const int blocks = cuda_utils::ceil_div(rows, kThreadsPerBlock);
  softmax_online_kernel<<<blocks, kThreadsPerBlock>>>(device_input, device_output, rows, cols);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(host_output.data(), device_output, bytes, cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(device_input));
  CHECK_CUDA(cudaFree(device_output));

  SoftmaxRun run;
  run.rows = rows;
  run.cols = cols;
  return run;
}

SoftmaxRun run_masked_softmax(
    const std::vector<float>& host_input,
    const std::vector<int>& host_mask,
    std::vector<float>& host_output,
    int rows,
    int cols) {
  float* device_input = nullptr;
  int* device_mask = nullptr;
  float* device_output = nullptr;
  const size_t bytes = host_input.size() * sizeof(float);
  const size_t mask_bytes = host_mask.size() * sizeof(int);

  CHECK_CUDA(cudaMalloc(&device_input, bytes));
  CHECK_CUDA(cudaMalloc(&device_mask, mask_bytes));
  CHECK_CUDA(cudaMalloc(&device_output, bytes));
  CHECK_CUDA(cudaMemcpy(device_input, host_input.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_mask, host_mask.data(), mask_bytes, cudaMemcpyHostToDevice));

  const int blocks = cuda_utils::ceil_div(rows, kThreadsPerBlock);
  softmax_masked_kernel<<<blocks, kThreadsPerBlock>>>(device_input, device_mask, device_output, rows, cols);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(host_output.data(), device_output, bytes, cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(device_input));
  CHECK_CUDA(cudaFree(device_mask));
  CHECK_CUDA(cudaFree(device_output));

  SoftmaxRun run;
  run.rows = rows;
  run.cols = cols;
  return run;
}

SoftmaxRun run_causal_softmax(
    const std::vector<float>& host_input,
    std::vector<float>& host_output,
    int rows,
    int cols) {
  float* device_input = nullptr;
  float* device_output = nullptr;
  const size_t bytes = host_input.size() * sizeof(float);

  CHECK_CUDA(cudaMalloc(&device_input, bytes));
  CHECK_CUDA(cudaMalloc(&device_output, bytes));
  CHECK_CUDA(cudaMemcpy(device_input, host_input.data(), bytes, cudaMemcpyHostToDevice));

  const int blocks = cuda_utils::ceil_div(rows, kThreadsPerBlock);
  softmax_causal_kernel<<<blocks, kThreadsPerBlock>>>(device_input, device_output, rows, cols);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(host_output.data(), device_output, bytes, cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(device_input));
  CHECK_CUDA(cudaFree(device_output));

  SoftmaxRun run;
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
  constexpr int cols = 257;
  constexpr int causal_rows = 64;
  constexpr int causal_cols = 64;

  std::vector<float> host_input(rows * cols);
  std::vector<float> host_output_naive(rows * cols, 0.0f);
  std::vector<float> host_output_block(rows * cols, 0.0f);
  std::vector<float> host_output_online(rows * cols, 0.0f);
  std::vector<int> host_mask(rows * cols);
  std::vector<float> host_output_masked(rows * cols, 0.0f);
  std::vector<float> reference(rows * cols, 0.0f);
  std::vector<float> causal_input(causal_rows * causal_cols);
  std::vector<float> host_output_causal(causal_rows * causal_cols, 0.0f);
  std::vector<float> reference_causal(causal_rows * causal_cols, 0.0f);

  fill_input(host_input, rows, cols);
  softmax_cpu(host_input, reference, rows, cols);
  softmax_online_cpu(host_input, host_output_online, rows, cols);
  fill_mask(host_mask, rows, cols);
  softmax_masked_cpu(host_input, host_mask, host_output_masked, rows, cols);

  fill_input(causal_input, causal_rows, causal_cols);
  softmax_causal_cpu(causal_input, reference_causal, causal_rows, causal_cols);

  const SoftmaxRun naive_run = run_naive_softmax(host_input, host_output_naive, rows, cols);
  const SoftmaxRun block_run = run_block_softmax(host_input, host_output_block, rows, cols);
  const SoftmaxRun online_run = run_online_softmax(host_input, host_output_online, rows, cols);
  const SoftmaxRun masked_run = run_masked_softmax(host_input, host_mask, host_output_masked, rows, cols);
  const SoftmaxRun causal_run = run_causal_softmax(causal_input, host_output_causal, causal_rows, causal_cols);

  const bool naive_ok = check_output(host_output_naive, reference, rows, cols);
  const bool block_ok = check_output(host_output_block, reference, rows, cols);
  const bool online_ok = check_output(host_output_online, reference, rows, cols);
  const bool masked_ok = check_output(host_output_masked, reference, rows, cols);
  const bool causal_ok = check_output(host_output_causal, reference_causal, causal_rows, causal_cols);

  if (!naive_ok || !block_ok || !online_ok || !masked_ok || !causal_ok) {
    return EXIT_FAILURE;
  }

  std::cout << "App: softmax" << '\n';
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

  std::cout << "[online version]" << '\n';
  std::cout << "  one thread computes one row with running max/sum" << '\n';
  std::cout << "  rows: " << online_run.rows << ", cols: " << online_run.cols << '\n';
  print_row_sample(host_output_online, cols, "  sample row");
  std::cout << '\n';

  std::cout << "[masked version]" << '\n';
  std::cout << "  mask skips invalid positions during max/sum/normalize" << '\n';
  std::cout << "  rows: " << masked_run.rows << ", cols: " << masked_run.cols << '\n';
  print_row_sample(host_output_masked, cols, "  sample row");
  std::cout << '\n';

  std::cout << "[causal version]" << '\n';
  std::cout << "  row i can only see columns <= i" << '\n';
  std::cout << "  rows: " << causal_run.rows << ", cols: " << causal_run.cols << '\n';
  print_row_sample(host_output_causal, causal_cols, "  sample row");
  std::cout << '\n';

  std::cout << "All versions: PASS" << '\n';
  return EXIT_SUCCESS;
}
