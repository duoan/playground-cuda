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
// 2. simplified FlashAttention v2 teaching kernel
// 3. warp-specialized teaching kernel

constexpr int kQueryCount = 4;
constexpr int kKeyCount = 16;
constexpr int kHeadDim = 8;
constexpr int kTileKeys = 4;
constexpr int kThreadsPerBlock = kTileKeys;
constexpr int kWarpThreadsPerBlock = 32;
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

// FlashAttention v2 teaching kernel:
// One block handles one query row.
// The block first stages a key/value tile into shared memory, then a single
// thread walks the tile and updates the online softmax accumulators.
//
// This version is still educational, not fast. The goal is to show the shared
// memory staging pattern that real flash attention kernels use.
__global__ void flash_attention_v2_kernel(
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

  __shared__ float shared_k[kTileKeys][kHeadDim];
  __shared__ float shared_v[kTileKeys][kHeadDim];
  __shared__ float shared_scores[kTileKeys];

  float running_max = -1.0e30f;
  float running_sum = 0.0f;
  float accum[kHeadDim];

  #pragma unroll
  for (int d = 0; d < kHeadDim; ++d) {
    accum[d] = 0.0f;
  }

  for (int key_start = 0; key_start < key_count; key_start += kTileKeys) {
    const int local_key = threadIdx.x;
    const int key = key_start + local_key;

    if (local_key < kTileKeys && key < key_count) {
      #pragma unroll
      for (int d = 0; d < kHeadDim; ++d) {
        shared_k[local_key][d] = k[key * head_dim + d];
        shared_v[local_key][d] = v[key * head_dim + d];
      }
      shared_scores[local_key] = 0.0f;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      float tile_max = -1.0e30f;
      const int tile_count = (key_start + kTileKeys <= key_count)
          ? kTileKeys
          : (key_count - key_start);

      // Score each key in the tile against the query row.
      for (int t = 0; t < tile_count; ++t) {
        float score = 0.0f;
        for (int d = 0; d < kHeadDim; ++d) {
          score += q[row * head_dim + d] * shared_k[t][d];
        }
        shared_scores[t] = score;
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
        const float weight = expf(shared_scores[t] - new_max);
        tile_sum += weight;
        for (int d = 0; d < kHeadDim; ++d) {
          accum[d] += weight * shared_v[t][d];
        }
      }

      running_sum = running_sum * old_scale + tile_sum;
      running_max = new_max;
    }

    __syncthreads();
  }

  if (threadIdx.x == 0) {
    for (int d = 0; d < kHeadDim; ++d) {
      out[row * head_dim + d] = accum[d] / (running_sum + kEps);
    }
  }
}

}  // namespace

__global__ void flash_attention_v2_warp_kernel(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int query_count,
    int key_count,
    int head_dim);

int main() {
  std::vector<float> host_q(kQueryCount * kHeadDim);
  std::vector<float> host_k(kKeyCount * kHeadDim);
  std::vector<float> host_v(kKeyCount * kHeadDim);
  std::vector<float> host_reference(kQueryCount * kHeadDim, 0.0f);
  std::vector<float> host_output(kQueryCount * kHeadDim, 0.0f);
  std::vector<float> host_warp_output(kQueryCount * kHeadDim, 0.0f);

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
  float* device_warp_out = nullptr;
  CHECK_CUDA(cudaMalloc(&device_warp_out, out_bytes));

  CHECK_CUDA(cudaMemcpy(device_q, host_q.data(), q_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_k, host_k.data(), k_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_v, host_v.data(), v_bytes, cudaMemcpyHostToDevice));

  flash_attention_v2_kernel<<<kQueryCount, kThreadsPerBlock>>>(
      device_q, device_k, device_v, device_out, kQueryCount, kKeyCount, kHeadDim);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(host_output.data(), device_out, out_bytes, cudaMemcpyDeviceToHost));

  flash_attention_v2_warp_kernel<<<kQueryCount, kWarpThreadsPerBlock>>>(
      device_q, device_k, device_v, device_warp_out, kQueryCount, kKeyCount, kHeadDim);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(
      host_warp_output.data(), device_warp_out, out_bytes, cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(device_q));
  CHECK_CUDA(cudaFree(device_k));
  CHECK_CUDA(cudaFree(device_v));
  CHECK_CUDA(cudaFree(device_out));
  CHECK_CUDA(cudaFree(device_warp_out));

  if (!check_output(host_output, host_reference)) {
    return EXIT_FAILURE;
  }
  if (!check_output(host_warp_output, host_reference)) {
    return EXIT_FAILURE;
  }

  std::cout << "App: flash_attention_v2" << '\n';
  std::cout << "Idea: shared-memory tile staging, then warp specialization" << '\n';
  print_row_sample(host_reference, 0, "Reference row 0");
  print_row_sample(host_output, 0, "GPU row 0");
  print_row_sample(host_warp_output, 0, "Warp GPU row 0");
  std::cout << "Check: PASS" << '\n';
  return EXIT_SUCCESS;
}

// FlashAttention v2 warp-specialized teaching kernel:
// One block handles one query row.
// Different lane groups in the warp have different jobs:
// - lanes 0..7 load q and later update the output vector
// - lanes 0..3 stage K/V tiles and compute the tile scores
// - lane 0 performs the online-softmax state update
//
// This is still a teaching kernel, not a production warp-specialized kernel,
// but it makes the lane roles visible and easy to follow.
__global__ void flash_attention_v2_warp_kernel(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int query_count,
    int key_count,
    int head_dim) {
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  if (row >= query_count) {
    return;
  }

  __shared__ float q_shared[kHeadDim];
  __shared__ float shared_k[kTileKeys][kHeadDim];
  __shared__ float shared_v[kTileKeys][kHeadDim];
  __shared__ float shared_scores[kTileKeys];
  __shared__ float shared_weights[kTileKeys];
  __shared__ float running_max_shared;
  __shared__ float running_sum_shared;
  __shared__ float tile_max_shared;
  __shared__ float tile_sum_shared;
  __shared__ float old_scale_shared;

  if (lane < kHeadDim) {
    q_shared[lane] = q[row * head_dim + lane];
  }
  if (lane == 0) {
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
    if (lane < kTileKeys) {
      const int key = key_start + lane;
      if (key < key_count) {
        #pragma unroll
        for (int d = 0; d < kHeadDim; ++d) {
          shared_k[lane][d] = k[key * head_dim + d];
          shared_v[lane][d] = v[key * head_dim + d];
        }
      } else {
        #pragma unroll
        for (int d = 0; d < kHeadDim; ++d) {
          shared_k[lane][d] = 0.0f;
          shared_v[lane][d] = 0.0f;
        }
      }
    }
    __syncthreads();

    if (lane < kTileKeys) {
      float score = 0.0f;
      const int key = key_start + lane;
      if (key < key_count) {
        for (int d = 0; d < kHeadDim; ++d) {
          score += q_shared[d] * shared_k[lane][d];
        }
      } else {
        score = -1.0e30f;
      }
      shared_scores[lane] = score;
    }
    __syncthreads();

    if (lane == 0) {
      tile_max_shared = -1.0e30f;
      const int tile_count = (key_start + kTileKeys <= key_count)
          ? kTileKeys
          : (key_count - key_start);

      for (int t = 0; t < tile_count; ++t) {
        tile_max_shared = fmaxf(tile_max_shared, shared_scores[t]);
      }

      const float new_max = fmaxf(running_max_shared, tile_max_shared);
      old_scale_shared = (running_sum_shared == 0.0f)
          ? 0.0f
          : expf(running_max_shared - new_max);

      tile_sum_shared = 0.0f;
      for (int t = 0; t < tile_count; ++t) {
        const float weight = expf(shared_scores[t] - new_max);
        shared_weights[t] = weight;
        tile_sum_shared += weight;
      }
      for (int t = tile_count; t < kTileKeys; ++t) {
        shared_weights[t] = 0.0f;
      }

      running_sum_shared = running_sum_shared * old_scale_shared + tile_sum_shared;
      running_max_shared = new_max;
    }
    __syncthreads();

    if (lane < kHeadDim) {
      accum[lane] *= old_scale_shared;
      for (int t = 0; t < kTileKeys; ++t) {
        accum[lane] += shared_weights[t] * shared_v[t][lane];
      }
    }
    __syncthreads();
  }

  if (lane < kHeadDim) {
    out[row * head_dim + lane] = accum[lane] / (running_sum_shared + kEps);
  }
}
