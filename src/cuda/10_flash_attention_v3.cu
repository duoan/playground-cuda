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
// 2. simplified FlashAttention v3 teaching kernel
// 3. pipeline-oriented teaching kernel

constexpr int kQueryCount = 4;
constexpr int kKeyCount = 16;
constexpr int kHeadDim = 8;
constexpr int kWarpSize = 32;
constexpr int kPipelineStages = 2;
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

__device__ float warp_reduce_sum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

// FlashAttention v3 teaching kernel:
// One warp handles one query row.
// The first kHeadDim lanes act like the output vector lanes.
// They also cooperate to compute each attention score through warp reduction.
//
// This is the most "GPU-like" version in this set, but it is still pedagogical:
// the goal is to show warp specialization and lane roles, not production speed.
__global__ void flash_attention_v3_kernel(
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

  const unsigned int mask = 0xffffffffu;
  float accum[kHeadDim];
  #pragma unroll
  for (int d = 0; d < kHeadDim; ++d) {
    accum[d] = 0.0f;
  }

  float running_max = -1.0e30f;
  float running_sum = 0.0f;

  for (int key = 0; key < key_count; ++key) {
    // Each active lane contributes one partial product for the dot product.
    float partial = 0.0f;
    if (lane < head_dim) {
      partial = q[row * head_dim + lane] * k[key * head_dim + lane];
    }

    // The partials are summed across the warp.
    const float score_sum = warp_reduce_sum(partial);
    const float score = __shfl_sync(mask, score_sum, 0);

    float old_scale = 0.0f;
    float weight = 0.0f;

    if (lane == 0) {
      const float new_max = (running_max > score) ? running_max : score;
      old_scale = (running_sum == 0.0f) ? 0.0f : expf(running_max - new_max);
      weight = expf(score - new_max);
      running_sum = running_sum * old_scale + weight;
      running_max = new_max;
    }

    // Broadcast the control values from lane 0 so every active lane can update
    // its output component in the same way.
    old_scale = __shfl_sync(mask, old_scale, 0);
    weight = __shfl_sync(mask, weight, 0);

    if (lane < head_dim) {
      accum[lane] = accum[lane] * old_scale + weight * v[key * head_dim + lane];
    }
  }

  const float final_sum = __shfl_sync(mask, running_sum, 0);
  if (lane < head_dim) {
    out[row * head_dim + lane] = accum[lane] / (final_sum + kEps);
  }
}

// FlashAttention v3 pipeline-oriented teaching kernel:
// This version uses a conceptual double buffer for key/value staging.
// It is not an async-copy Hopper kernel, but it shows the shape of a pipeline:
// preload -> consume -> preload next -> consume next.
//
// The goal is to make the notion of "pipeline-oriented" easier to read before
// you learn hardware-specific async copy and tensor-core choreography.
__global__ void flash_attention_v3_pipeline_kernel(
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

  __shared__ float k_stage[kPipelineStages][kHeadDim];
  __shared__ float v_stage[kPipelineStages][kHeadDim];
  __shared__ float running_max_shared;
  __shared__ float running_sum_shared;

  if (lane == 0) {
    running_max_shared = -1.0e30f;
    running_sum_shared = 0.0f;
  }

  float accum[kHeadDim];
  #pragma unroll
  for (int d = 0; d < kHeadDim; ++d) {
    accum[d] = 0.0f;
  }

  // Preload the first key/value into stage 0.
  if (lane < head_dim) {
    k_stage[0][lane] = k[0 * head_dim + lane];
    v_stage[0][lane] = v[0 * head_dim + lane];
  }
  __syncthreads();

  for (int key = 0; key < key_count; ++key) {
    const int stage = key % kPipelineStages;
    const int next_stage = (key + 1) % kPipelineStages;

    // Prefetch the next key/value pair into the alternate stage.
    if (key + 1 < key_count && lane < head_dim) {
      k_stage[next_stage][lane] = k[(key + 1) * head_dim + lane];
      v_stage[next_stage][lane] = v[(key + 1) * head_dim + lane];
    }
    __syncthreads();

    float partial = 0.0f;
    if (lane < head_dim) {
      partial = q[row * head_dim + lane] * k_stage[stage][lane];
    }

    const float score_sum = warp_reduce_sum(partial);
    const float score = __shfl_sync(0xffffffffu, score_sum, 0);

    float old_scale = 0.0f;
    float weight = 0.0f;
    if (lane == 0) {
      const float new_max = (running_max_shared > score) ? running_max_shared : score;
      old_scale = (running_sum_shared == 0.0f) ? 0.0f : expf(running_max_shared - new_max);
      weight = expf(score - new_max);
      running_sum_shared = running_sum_shared * old_scale + weight;
      running_max_shared = new_max;
    }

    old_scale = __shfl_sync(0xffffffffu, old_scale, 0);
    weight = __shfl_sync(0xffffffffu, weight, 0);

    if (lane < head_dim) {
      accum[lane] = accum[lane] * old_scale + weight * v_stage[stage][lane];
    }
    __syncthreads();
  }

  const float final_sum = __shfl_sync(0xffffffffu, running_sum_shared, 0);
  if (lane < head_dim) {
    out[row * head_dim + lane] = accum[lane] / (final_sum + kEps);
  }
}

}  // namespace

int main() {
  std::vector<float> host_q(kQueryCount * kHeadDim);
  std::vector<float> host_k(kKeyCount * kHeadDim);
  std::vector<float> host_v(kKeyCount * kHeadDim);
  std::vector<float> host_reference(kQueryCount * kHeadDim, 0.0f);
  std::vector<float> host_output(kQueryCount * kHeadDim, 0.0f);
  std::vector<float> host_pipeline_output(kQueryCount * kHeadDim, 0.0f);

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
  float* device_pipeline_out = nullptr;
  CHECK_CUDA(cudaMalloc(&device_pipeline_out, out_bytes));

  CHECK_CUDA(cudaMemcpy(device_q, host_q.data(), q_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_k, host_k.data(), k_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_v, host_v.data(), v_bytes, cudaMemcpyHostToDevice));

  flash_attention_v3_kernel<<<kQueryCount, kWarpSize>>>(
      device_q, device_k, device_v, device_out, kQueryCount, kKeyCount, kHeadDim);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(host_output.data(), device_out, out_bytes, cudaMemcpyDeviceToHost));

  flash_attention_v3_pipeline_kernel<<<kQueryCount, kWarpSize>>>(
      device_q, device_k, device_v, device_pipeline_out, kQueryCount, kKeyCount, kHeadDim);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(
      host_pipeline_output.data(), device_pipeline_out, out_bytes, cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(device_q));
  CHECK_CUDA(cudaFree(device_k));
  CHECK_CUDA(cudaFree(device_v));
  CHECK_CUDA(cudaFree(device_out));
  CHECK_CUDA(cudaFree(device_pipeline_out));

  if (!check_output(host_output, host_reference)) {
    return EXIT_FAILURE;
  }
  if (!check_output(host_pipeline_output, host_reference)) {
    return EXIT_FAILURE;
  }

  std::cout << "App: flash_attention_v3" << '\n';
  std::cout << "Idea: warp-specialized streaming attention, then pipeline staging" << '\n';
  print_row_sample(host_reference, 0, "Reference row 0");
  print_row_sample(host_output, 0, "GPU row 0");
  print_row_sample(host_pipeline_output, 0, "Pipeline GPU row 0");
  std::cout << "Check: PASS" << '\n';
  return EXIT_SUCCESS;
}
