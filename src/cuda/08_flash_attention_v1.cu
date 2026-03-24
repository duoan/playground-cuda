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
// 1. CPU/reference decomposition
// 2. simplified FlashAttention v1 teaching kernel
// 3. shared-memory staged teaching kernel

// Small sizes keep the code readable and easy to study.
// We are intentionally teaching the algorithm first, not chasing performance.
constexpr int kQueryCount = 4;
constexpr int kKeyCount = 16;
constexpr int kHeadDim = 8;
constexpr int kTileKeys = 4;
constexpr int kThreadsPerBlock = 32;
constexpr int kSharedThreadsPerBlock = 8;
constexpr float kEps = 1e-6f;

double dot_product(const float* a, const float* b, int dim) {
  double result = 0.0;
  for (int d = 0; d < dim; ++d) {
    result += static_cast<double>(a[d]) * static_cast<double>(b[d]);
  }
  return result;
}

void fill_inputs(std::vector<float>& q, std::vector<float>& k, std::vector<float>& v) {
  for (int row = 0; row < kQueryCount; ++row) {
    for (int d = 0; d < kHeadDim; ++d) {
      q[row * kHeadDim + d] = 0.1f * static_cast<float>(row + 1) + 0.03f * static_cast<float>(d);
    }
  }

  for (int row = 0; row < kKeyCount; ++row) {
    for (int d = 0; d < kHeadDim; ++d) {
      k[row * kHeadDim + d] = 0.05f * static_cast<float>(row + 1) + 0.02f * static_cast<float>(d + 1);
      v[row * kHeadDim + d] = 0.07f * static_cast<float>(row + 1) + 0.01f * static_cast<float>(d);
    }
  }
}

void flash_attention_cpu_reference(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    std::vector<float>& out) {
  for (int row = 0; row < kQueryCount; ++row) {
    double scores[kKeyCount];
    double row_max = -1.0e30;

    for (int key = 0; key < kKeyCount; ++key) {
      const double score = dot_product(
          &q[row * kHeadDim], &k[key * kHeadDim], kHeadDim);
      scores[key] = score;
      if (score > row_max) {
        row_max = score;
      }
    }

    double row_sum = 0.0;
    double accum[kHeadDim] = {0.0};
    for (int key = 0; key < kKeyCount; ++key) {
      const double weight = std::exp(scores[key] - row_max);
      row_sum += weight;
      for (int d = 0; d < kHeadDim; ++d) {
        accum[d] += weight * static_cast<double>(v[key * kHeadDim + d]);
      }
    }

    for (int d = 0; d < kHeadDim; ++d) {
      out[row * kHeadDim + d] = static_cast<float>(accum[d] / (row_sum + kEps));
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

void print_row_sample(const std::vector<float>& values, int row, const char* label) {
  std::cout << label << ":";
  for (int d = 0; d < kHeadDim; ++d) {
    std::cout << ' ' << values[row * kHeadDim + d];
  }
  std::cout << '\n';
}

__device__ float dot_row_device(const float* a, const float* b, int dim) {
  float result = 0.0f;
  for (int d = 0; d < dim; ++d) {
    result += a[d] * b[d];
  }
  return result;
}

// FlashAttention v1 teaching kernel:
// One thread handles one query row.
// The thread streams through the keys in small tiles and keeps only:
// - a running max
// - a running sum of exponentials
// - a running weighted output accumulator
//
// This is the core "online softmax" idea:
// we do not store the full score matrix.
__global__ void flash_attention_v1_kernel(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int query_count,
    int key_count,
    int head_dim) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= query_count) {
    return;
  }

  float running_max = -1.0e30f;
  float running_sum = 0.0f;
  float accum[kHeadDim];

  #pragma unroll
  for (int d = 0; d < kHeadDim; ++d) {
    accum[d] = 0.0f;
  }

  float scores[kTileKeys];

  for (int key_start = 0; key_start < key_count; key_start += kTileKeys) {
    const int tile_count = (key_start + kTileKeys <= key_count)
        ? kTileKeys
        : (key_count - key_start);

    float tile_max = -1.0e30f;
    #pragma unroll
    for (int t = 0; t < kTileKeys; ++t) {
      scores[t] = 0.0f;
    }

    for (int t = 0; t < tile_count; ++t) {
      const int key = key_start + t;
      const float score = dot_row_device(
          &q[row * head_dim], &k[key * head_dim], head_dim);
      scores[t] = score;
      if (score > tile_max) {
        tile_max = score;
      }
    }

    const float new_max = (running_max > tile_max) ? running_max : tile_max;
    const float old_scale = (running_sum == 0.0f) ? 0.0f : expf(running_max - new_max);

    for (int d = 0; d < kHeadDim; ++d) {
      accum[d] *= old_scale;
    }

    float tile_sum = 0.0f;
    for (int t = 0; t < tile_count; ++t) {
      const float weight = expf(scores[t] - new_max);
      tile_sum += weight;
      const int key = key_start + t;
      for (int d = 0; d < kHeadDim; ++d) {
        accum[d] += weight * v[key * head_dim + d];
      }
    }

    running_sum = running_sum * old_scale + tile_sum;
    running_max = new_max;
  }

  for (int d = 0; d < kHeadDim; ++d) {
    out[row * head_dim + d] = accum[d] / (running_sum + kEps);
  }
}

// FlashAttention v1 staged teaching kernel:
// The math is still the same online softmax, but now the block explicitly
// stages q, K, and V in shared memory before doing the math.
//
// This is the cleanest way to explain "SRAM reuse" without introducing warp
// specialization yet.
__global__ void flash_attention_v1_shared_kernel(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int query_count,
    int key_count,
    int head_dim) {
  const int row = blockIdx.x;
  if (row >= query_count) {
    return;
  }

  __shared__ float q_shared[kHeadDim];
  __shared__ float k_tile[kTileKeys][kHeadDim];
  __shared__ float v_tile[kTileKeys][kHeadDim];
  __shared__ float scores[kTileKeys];
  __shared__ float running_max_shared;
  __shared__ float running_sum_shared;

  if (threadIdx.x < kHeadDim) {
    q_shared[threadIdx.x] = q[row * head_dim + threadIdx.x];
  }
  if (threadIdx.x == 0) {
    running_max_shared = -1.0e30f;
    running_sum_shared = 0.0f;
  }
  __syncthreads();

  float accum[kHeadDim];
  #pragma unroll
  for (int d = 0; d < kHeadDim; ++d) {
    accum[d] = 0.0f;
  }

  for (int key_start = 0; key_start < key_count; key_start += kTileKeys) {
    const int token = threadIdx.x;
    const int key = key_start + token;

    if (token < kTileKeys && key < key_count) {
      #pragma unroll
      for (int d = 0; d < kHeadDim; ++d) {
        k_tile[token][d] = k[key * head_dim + d];
        v_tile[token][d] = v[key * head_dim + d];
      }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      float tile_max = -1.0e30f;
      const int tile_count = (key_start + kTileKeys <= key_count)
          ? kTileKeys
          : (key_count - key_start);

      for (int t = 0; t < tile_count; ++t) {
        float score = 0.0f;
        for (int d = 0; d < kHeadDim; ++d) {
          score += q_shared[d] * k_tile[t][d];
        }
        scores[t] = score;
        tile_max = fmaxf(tile_max, score);
      }

      const float new_max = fmaxf(running_max_shared, tile_max);
      const float old_scale = (running_sum_shared == 0.0f)
          ? 0.0f
          : expf(running_max_shared - new_max);

      for (int d = 0; d < kHeadDim; ++d) {
        accum[d] *= old_scale;
      }

      float tile_sum = 0.0f;
      for (int t = 0; t < tile_count; ++t) {
        const float weight = expf(scores[t] - new_max);
        tile_sum += weight;
        for (int d = 0; d < kHeadDim; ++d) {
          accum[d] += weight * v_tile[t][d];
        }
      }

      running_sum_shared = running_sum_shared * old_scale + tile_sum;
      running_max_shared = new_max;
    }

    __syncthreads();
  }

  if (threadIdx.x == 0) {
    for (int d = 0; d < kHeadDim; ++d) {
      out[row * head_dim + d] = accum[d] / (running_sum_shared + kEps);
    }
  }
}

}  // namespace

int main() {
  std::vector<float> host_q(kQueryCount * kHeadDim);
  std::vector<float> host_k(kKeyCount * kHeadDim);
  std::vector<float> host_v(kKeyCount * kHeadDim);
  std::vector<float> host_reference(kQueryCount * kHeadDim, 0.0f);
  std::vector<float> host_output(kQueryCount * kHeadDim, 0.0f);
  std::vector<float> host_staged_output(kQueryCount * kHeadDim, 0.0f);

  fill_inputs(host_q, host_k, host_v);
  flash_attention_cpu_reference(host_q, host_k, host_v, host_reference);

  float* device_q = nullptr;
  float* device_k = nullptr;
  float* device_v = nullptr;
  float* device_out = nullptr;

  const size_t q_bytes = host_q.size() * sizeof(float);
  const size_t k_bytes = host_k.size() * sizeof(float);
  const size_t v_bytes = host_v.size() * sizeof(float);
  const size_t out_bytes = host_output.size() * sizeof(float);

  CHECK_CUDA(cudaMalloc(&device_q, q_bytes));
  CHECK_CUDA(cudaMalloc(&device_k, k_bytes));
  CHECK_CUDA(cudaMalloc(&device_v, v_bytes));
  CHECK_CUDA(cudaMalloc(&device_out, out_bytes));
  float* device_staged_out = nullptr;
  CHECK_CUDA(cudaMalloc(&device_staged_out, out_bytes));

  CHECK_CUDA(cudaMemcpy(device_q, host_q.data(), q_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_k, host_k.data(), k_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_v, host_v.data(), v_bytes, cudaMemcpyHostToDevice));

  const int blocks = cuda_utils::ceil_div(kQueryCount, kThreadsPerBlock);
  flash_attention_v1_kernel<<<blocks, kThreadsPerBlock>>>(
      device_q, device_k, device_v, device_out, kQueryCount, kKeyCount, kHeadDim);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(host_output.data(), device_out, out_bytes, cudaMemcpyDeviceToHost));

  flash_attention_v1_shared_kernel<<<kQueryCount, kSharedThreadsPerBlock>>>(
      device_q, device_k, device_v, device_staged_out, kQueryCount, kKeyCount, kHeadDim);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(
      host_staged_output.data(), device_staged_out, out_bytes, cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(device_q));
  CHECK_CUDA(cudaFree(device_k));
  CHECK_CUDA(cudaFree(device_v));
  CHECK_CUDA(cudaFree(device_out));
  CHECK_CUDA(cudaFree(device_staged_out));

  if (!check_output(host_output, host_reference)) {
    return EXIT_FAILURE;
  }
  if (!check_output(host_staged_output, host_reference)) {
    return EXIT_FAILURE;
  }

  std::cout << "App: flash_attention_v1" << '\n';
  std::cout << "Idea: online softmax, then shared-memory staging" << '\n';
  print_row_sample(host_reference, 0, "Reference row 0");
  print_row_sample(host_output, 0, "GPU row 0");
  print_row_sample(host_staged_output, 0, "Staged GPU row 0");
  std::cout << "Check: PASS" << '\n';
  return EXIT_SUCCESS;
}
