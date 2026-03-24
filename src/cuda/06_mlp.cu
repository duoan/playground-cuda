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
// 2. naive staged CUDA MLP
// 3. partially fused first-stage + activation version
// 4. tiled GEMM-style MLP with richer epilogue fusion

// 这是一个很小的 MLP 教学例子。
// 我们把它写成 batch 版，这样更容易看出矩阵乘法和激活函数是怎么组合的。
constexpr int kBatch = 4;
constexpr int kInputDim = 8;
constexpr int kHiddenDim = 16;
constexpr int kOutputDim = 4;
constexpr int kThreadsPerBlock = 256;
constexpr int kTiledThreadsPerBlock = 32;
constexpr int kInputTile = 4;

void fill_inputs(std::vector<float>& x) {
  for (int i = 0; i < static_cast<int>(x.size()); ++i) {
    x[i] = static_cast<float>((i % 5) - 2);
  }
}

void fill_weights(std::vector<float>& w1, std::vector<float>& b1,
                  std::vector<float>& w2, std::vector<float>& b2) {
  for (int i = 0; i < static_cast<int>(w1.size()); ++i) {
    w1[i] = static_cast<float>((i % 7) - 3) * 0.1f;
  }
  for (int i = 0; i < static_cast<int>(b1.size()); ++i) {
    b1[i] = static_cast<float>((i % 3) - 1) * 0.05f;
  }
  for (int i = 0; i < static_cast<int>(w2.size()); ++i) {
    w2[i] = static_cast<float>((i % 5) - 2) * 0.08f;
  }
  for (int i = 0; i < static_cast<int>(b2.size()); ++i) {
    b2[i] = static_cast<float>((i % 4) - 1) * 0.03f;
  }
}

__host__ __device__ inline float relu(float value) {
  return value > 0.0f ? value : 0.0f;
}

void mlp_cpu_reference(
    const std::vector<float>& x,
    const std::vector<float>& w1,
    const std::vector<float>& b1,
    const std::vector<float>& w2,
    const std::vector<float>& b2,
    std::vector<float>& y) {
  std::vector<float> hidden(kBatch * kHiddenDim, 0.0f);

  for (int batch = 0; batch < kBatch; ++batch) {
    for (int hidden_idx = 0; hidden_idx < kHiddenDim; ++hidden_idx) {
      float acc = b1[hidden_idx];
      for (int input_idx = 0; input_idx < kInputDim; ++input_idx) {
        acc += x[batch * kInputDim + input_idx] *
               w1[input_idx * kHiddenDim + hidden_idx];
      }
      hidden[batch * kHiddenDim + hidden_idx] = relu(acc);
    }
  }

  for (int batch = 0; batch < kBatch; ++batch) {
    for (int out_idx = 0; out_idx < kOutputDim; ++out_idx) {
      float acc = b2[out_idx];
      for (int hidden_idx = 0; hidden_idx < kHiddenDim; ++hidden_idx) {
        acc += hidden[batch * kHiddenDim + hidden_idx] *
               w2[hidden_idx * kOutputDim + out_idx];
      }
      y[batch * kOutputDim + out_idx] = acc;
    }
  }
}

bool check_output(const std::vector<float>& got, const std::vector<float>& expected, const char* label) {
  for (size_t i = 0; i < got.size(); ++i) {
    if (std::fabs(got[i] - expected[i]) > 1e-4f) {
      std::cerr << label << " mismatch at " << i
                << ": got " << got[i]
                << ", expected " << expected[i] << '\n';
      return false;
    }
  }
  return true;
}

// naive MLP:
// 1. 先算第一层线性变换
// 2. 再做 ReLU
// 3. 再算第二层线性变换
//
// 这种写法最容易理解，但会把中间结果写回 global memory。
// 对初学者来说，这很适合先建立“MLP 是几个简单算子串起来的”这个认知。
__global__ void mlp_naive_linear1_kernel(
    const float* x,
    const float* w1,
    const float* b1,
    float* hidden) {
  const int batch = blockIdx.x;
  const int hidden_idx = threadIdx.x;
  if (batch >= kBatch || hidden_idx >= kHiddenDim) {
    return;
  }

  float acc = b1[hidden_idx];
  for (int input_idx = 0; input_idx < kInputDim; ++input_idx) {
    acc += x[batch * kInputDim + input_idx] *
           w1[input_idx * kHiddenDim + hidden_idx];
  }
  hidden[batch * kHiddenDim + hidden_idx] = acc;
}

__global__ void relu_kernel(float* values, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    values[index] = relu(values[index]);
  }
}

__global__ void mlp_naive_linear2_kernel(
    const float* hidden,
    const float* w2,
    const float* b2,
    float* y) {
  const int batch = blockIdx.x;
  const int out_idx = threadIdx.x;
  if (batch >= kBatch || out_idx >= kOutputDim) {
    return;
  }

  float acc = b2[out_idx];
  for (int hidden_idx = 0; hidden_idx < kHiddenDim; ++hidden_idx) {
    acc += hidden[batch * kHiddenDim + hidden_idx] *
           w2[hidden_idx * kOutputDim + out_idx];
  }
  y[batch * kOutputDim + out_idx] = acc;
}

// fused MLP:
// 这里我们把第一层线性变换和 ReLU 融合在一起。
// 这比 naive 版本更像实际工程里的“epilogue fusion”。
//
// 对初学者来说，这个版本想传达的核心不是“更快”，
// 而是“很多时候可以在写出中间值之前，直接把激活做掉”。
__global__ void mlp_fused_linear1_relu_kernel(
    const float* x,
    const float* w1,
    const float* b1,
    float* hidden) {
  const int batch = blockIdx.x;
  const int hidden_idx = threadIdx.x;
  if (batch >= kBatch || hidden_idx >= kHiddenDim) {
    return;
  }

  float acc = b1[hidden_idx];
  for (int input_idx = 0; input_idx < kInputDim; ++input_idx) {
    acc += x[batch * kInputDim + input_idx] *
           w1[input_idx * kHiddenDim + hidden_idx];
  }
  hidden[batch * kHiddenDim + hidden_idx] = relu(acc);
}

__global__ void mlp_linear2_kernel(
    const float* hidden,
    const float* w2,
    const float* b2,
    float* y) {
  const int batch = blockIdx.x;
  const int out_idx = threadIdx.x;
  if (batch >= kBatch || out_idx >= kOutputDim) {
    return;
  }

  float acc = b2[out_idx];
  for (int hidden_idx = 0; hidden_idx < kHiddenDim; ++hidden_idx) {
    acc += hidden[batch * kHiddenDim + hidden_idx] *
           w2[hidden_idx * kOutputDim + out_idx];
  }
  y[batch * kOutputDim + out_idx] = acc;
}

// tiled fused MLP:
// 这一版把“输入 tile staging”也加进来。
//
// 组织方式是:
// 1. 一个 block 负责一个 batch
// 2. 先把输入向量 x 分 tile 搬进 shared memory
// 3. 隐藏层在同一个 kernel 里完成线性变换 + ReLU
// 4. 隐藏层结果留在 shared memory 中，直接做第二层输出
//
// 这比上一版更接近真正的 GPU kernel 思维:
// - 少一次全局内存往返
// - 输入数据可以被多个 hidden thread 复用
// - 输出 epilogue 直接在 kernel 里写回
__global__ void mlp_tiled_fused_kernel(
    const float* x,
    const float* w1,
    const float* b1,
    const float* w2,
    const float* b2,
    float* y) {
  __shared__ float x_tile[kInputTile];
  __shared__ float hidden_shared[kHiddenDim];

  const int batch = blockIdx.x;
  const int tid = threadIdx.x;
  if (batch >= kBatch) {
    return;
  }

  // 隐藏层先从 bias 开始。
  float hidden_acc = 0.0f;
  if (tid < kHiddenDim) {
    hidden_acc = b1[tid];
  }

  // 用 tile 方式扫描输入维度。
  // 这里 input dim 很小，所以它更偏“教学上的 tile”，
  // 但这种写法和大 GEMM 的 staging 思路是同一回事。
  for (int tile = 0; tile < kInputDim; tile += kInputTile) {
    if (tid < kInputTile) {
      const int input_idx = tile + tid;
      x_tile[tid] = (input_idx < kInputDim)
          ? x[batch * kInputDim + input_idx]
          : 0.0f;
    }
    __syncthreads();

    if (tid < kHiddenDim) {
      #pragma unroll
      for (int i = 0; i < kInputTile; ++i) {
        const int input_idx = tile + i;
        if (input_idx < kInputDim) {
          hidden_acc += x_tile[i] * w1[input_idx * kHiddenDim + tid];
        }
      }
    }

    __syncthreads();
  }

  if (tid < kHiddenDim) {
    hidden_shared[tid] = relu(hidden_acc);
  }
  __syncthreads();

  // 第二层输出直接从 shared memory 里的 hidden activations 读。
  if (tid < kOutputDim) {
    float acc = b2[tid];
    for (int hidden_idx = 0; hidden_idx < kHiddenDim; ++hidden_idx) {
      acc += hidden_shared[hidden_idx] * w2[hidden_idx * kOutputDim + tid];
    }
    y[batch * kOutputDim + tid] = acc;
  }
}

void run_naive_mlp(
    const std::vector<float>& x,
    const std::vector<float>& w1,
    const std::vector<float>& b1,
    const std::vector<float>& w2,
    const std::vector<float>& b2,
    std::vector<float>& y) {
  float* device_x = nullptr;
  float* device_w1 = nullptr;
  float* device_b1 = nullptr;
  float* device_w2 = nullptr;
  float* device_b2 = nullptr;
  float* device_hidden = nullptr;
  float* device_y = nullptr;

  CHECK_CUDA(cudaMalloc(&device_x, x.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_w1, w1.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_b1, b1.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_w2, w2.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_b2, b2.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_hidden, kBatch * kHiddenDim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_y, y.size() * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(device_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_w1, w1.data(), w1.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_b1, b1.data(), b1.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_w2, w2.data(), w2.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_b2, b2.data(), b2.size() * sizeof(float), cudaMemcpyHostToDevice));

  mlp_naive_linear1_kernel<<<kBatch, kHiddenDim>>>(device_x, device_w1, device_b1, device_hidden);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());

  relu_kernel<<<cuda_utils::ceil_div(kBatch * kHiddenDim, kThreadsPerBlock), kThreadsPerBlock>>>(
      device_hidden, kBatch * kHiddenDim);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());

  mlp_naive_linear2_kernel<<<kBatch, kOutputDim>>>(device_hidden, device_w2, device_b2, device_y);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(y.data(), device_y, y.size() * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(device_x));
  CHECK_CUDA(cudaFree(device_w1));
  CHECK_CUDA(cudaFree(device_b1));
  CHECK_CUDA(cudaFree(device_w2));
  CHECK_CUDA(cudaFree(device_b2));
  CHECK_CUDA(cudaFree(device_hidden));
  CHECK_CUDA(cudaFree(device_y));
}

void run_fused_mlp(
    const std::vector<float>& x,
    const std::vector<float>& w1,
    const std::vector<float>& b1,
    const std::vector<float>& w2,
    const std::vector<float>& b2,
    std::vector<float>& y) {
  float* device_x = nullptr;
  float* device_w1 = nullptr;
  float* device_b1 = nullptr;
  float* device_w2 = nullptr;
  float* device_b2 = nullptr;
  float* device_hidden = nullptr;
  float* device_y = nullptr;

  CHECK_CUDA(cudaMalloc(&device_x, x.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_w1, w1.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_b1, b1.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_w2, w2.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_b2, b2.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_hidden, kBatch * kHiddenDim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_y, y.size() * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(device_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_w1, w1.data(), w1.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_b1, b1.data(), b1.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_w2, w2.data(), w2.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_b2, b2.data(), b2.size() * sizeof(float), cudaMemcpyHostToDevice));

  // Fused 版本把第一层的线性变换和 ReLU 放在同一个 kernel 里。
  // 这是一种很常见的优化思路：减少中间结果的读写次数。
  mlp_fused_linear1_relu_kernel<<<kBatch, kHiddenDim>>>(device_x, device_w1, device_b1, device_hidden);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());

  mlp_linear2_kernel<<<kBatch, kOutputDim>>>(device_hidden, device_w2, device_b2, device_y);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(y.data(), device_y, y.size() * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(device_x));
  CHECK_CUDA(cudaFree(device_w1));
  CHECK_CUDA(cudaFree(device_b1));
  CHECK_CUDA(cudaFree(device_w2));
  CHECK_CUDA(cudaFree(device_b2));
  CHECK_CUDA(cudaFree(device_hidden));
  CHECK_CUDA(cudaFree(device_y));
}

void run_tiled_fused_mlp(
    const std::vector<float>& x,
    const std::vector<float>& w1,
    const std::vector<float>& b1,
    const std::vector<float>& w2,
    const std::vector<float>& b2,
    std::vector<float>& y) {
  float* device_x = nullptr;
  float* device_w1 = nullptr;
  float* device_b1 = nullptr;
  float* device_w2 = nullptr;
  float* device_b2 = nullptr;
  float* device_y = nullptr;

  CHECK_CUDA(cudaMalloc(&device_x, x.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_w1, w1.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_b1, b1.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_w2, w2.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_b2, b2.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_y, y.size() * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(device_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_w1, w1.data(), w1.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_b1, b1.data(), b1.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_w2, w2.data(), w2.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_b2, b2.data(), b2.size() * sizeof(float), cudaMemcpyHostToDevice));

  mlp_tiled_fused_kernel<<<kBatch, kTiledThreadsPerBlock>>>(
      device_x, device_w1, device_b1, device_w2, device_b2, device_y);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(y.data(), device_y, y.size() * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(device_x));
  CHECK_CUDA(cudaFree(device_w1));
  CHECK_CUDA(cudaFree(device_b1));
  CHECK_CUDA(cudaFree(device_w2));
  CHECK_CUDA(cudaFree(device_b2));
  CHECK_CUDA(cudaFree(device_y));
}

void print_batch_sample(const std::vector<float>& values, const char* label) {
  std::cout << label << ":";
  for (int i = 0; i < static_cast<int>(values.size()) && i < 8; ++i) {
    std::cout << ' ' << values[i];
  }
  std::cout << '\n';
}

}  // namespace

int main() {
  std::vector<float> x(kBatch * kInputDim);
  std::vector<float> w1(kInputDim * kHiddenDim);
  std::vector<float> b1(kHiddenDim);
  std::vector<float> w2(kHiddenDim * kOutputDim);
  std::vector<float> b2(kOutputDim);

  std::vector<float> naive_y(kBatch * kOutputDim, 0.0f);
  std::vector<float> fused_y(kBatch * kOutputDim, 0.0f);
  std::vector<float> tiled_fused_y(kBatch * kOutputDim, 0.0f);
  std::vector<float> reference_y(kBatch * kOutputDim, 0.0f);

  fill_inputs(x);
  fill_weights(w1, b1, w2, b2);

  mlp_cpu_reference(x, w1, b1, w2, b2, reference_y);
  run_naive_mlp(x, w1, b1, w2, b2, naive_y);
  run_fused_mlp(x, w1, b1, w2, b2, fused_y);
  run_tiled_fused_mlp(x, w1, b1, w2, b2, tiled_fused_y);

  const bool naive_ok = check_output(naive_y, reference_y, "naive");
  const bool fused_ok = check_output(fused_y, reference_y, "fused");
  const bool tiled_fused_ok = check_output(tiled_fused_y, reference_y, "tiled_fused");
  if (!naive_ok || !fused_ok || !tiled_fused_ok) {
    return EXIT_FAILURE;
  }

  std::cout << "App: mlp" << '\n';
  std::cout << "Batch: " << kBatch << '\n';
  std::cout << "Input dim: " << kInputDim << '\n';
  std::cout << "Hidden dim: " << kHiddenDim << '\n';
  std::cout << "Output dim: " << kOutputDim << '\n';
  std::cout << '\n';
  std::cout << "[naive version] PASS" << '\n';
  print_batch_sample(naive_y, "  sample output");
  std::cout << "[partially fused version] PASS" << '\n';
  print_batch_sample(fused_y, "  sample output");
  std::cout << "[tiled fused version] PASS" << '\n';
  print_batch_sample(tiled_fused_y, "  sample output");

  return EXIT_SUCCESS;
}
