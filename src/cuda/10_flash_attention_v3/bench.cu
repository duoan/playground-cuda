// bench harness: runs every flash_attention_v3 version once for ncu.
// Each version lives in its own inline namespace so their `kTileKeys` etc
// constants don't collide when they differ.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../common/cuda_utils.cuh"

namespace {
// -D override: -DQUERY_COUNT=N
#ifndef QUERY_COUNT
#define QUERY_COUNT 4
#endif
constexpr int kQueryCount = QUERY_COUNT;
// -D override: -DKEY_COUNT=N
#ifndef KEY_COUNT
#define KEY_COUNT 16
#endif
constexpr int kKeyCount = KEY_COUNT;
// -D override: -DHEAD_DIM=N
#ifndef HEAD_DIM
#define HEAD_DIM 8
#endif
constexpr int kHeadDim = HEAD_DIM;
constexpr float kEps = 1e-6f;

void fill_inputs(std::vector<float>& q, std::vector<float>& k, std::vector<float>& v) {
    for (int row = 0; row < kQueryCount; ++row)
        for (int d = 0; d < kHeadDim; ++d)
            q[row * kHeadDim + d] = 0.1f * static_cast<float>(row + 1) + 0.03f * static_cast<float>(d);
    for (int row = 0; row < kKeyCount; ++row)
        for (int d = 0; d < kHeadDim; ++d) {
            k[row * kHeadDim + d] = 0.05f * static_cast<float>(row + 1) + 0.02f * static_cast<float>(d + 1);
            v[row * kHeadDim + d] = 0.07f * static_cast<float>(row + 1) + 0.01f * static_cast<float>(d);
        }
}
}  // namespace

namespace warp_streaming_ns {

using ::kQueryCount;
using ::kKeyCount;
using ::kHeadDim;
using ::kEps;


constexpr int kWarpSize = 32;

__device__ float warp_reduce_sum(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

// One warp per query row.  Each score is a warp-wide dot product across the
// head dimension: lane d contributes q[d]*k[d], warp shuffles sum them.  The
// online-softmax state lives in registers on lane 0 and is broadcast back.
__global__ void fa_v3_warp_kernel(const float* q, const float* k, const float* v, float* out,
                                  int query_count, int key_count, int head_dim) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= query_count) return;

    const unsigned int mask = 0xffffffffu;
    float accum[kHeadDim];
    #pragma unroll
    for (int d = 0; d < kHeadDim; ++d) accum[d] = 0.0f;

    float running_max = -1.0e30f;
    float running_sum = 0.0f;

    for (int key = 0; key < key_count; ++key) {
        float partial = 0.0f;
        if (lane < head_dim) partial = q[row * head_dim + lane] * k[key * head_dim + lane];
        const float score = __shfl_sync(mask, warp_reduce_sum(partial), 0);

        float old_scale = 0.0f, weight = 0.0f;
        if (lane == 0) {
            const float new_max = (running_max > score) ? running_max : score;
            old_scale = (running_sum == 0.0f) ? 0.0f : expf(running_max - new_max);
            weight = expf(score - new_max);
            running_sum = running_sum * old_scale + weight;
            running_max = new_max;
        }
        old_scale = __shfl_sync(mask, old_scale, 0);
        weight = __shfl_sync(mask, weight, 0);

        if (lane < head_dim)
            accum[lane] = accum[lane] * old_scale + weight * v[key * head_dim + lane];
    }

    const float final_sum = __shfl_sync(mask, running_sum, 0);
    if (lane < head_dim) out[row * head_dim + lane] = accum[lane] / (final_sum + kEps);
}

void launch_warp_streaming(const float* dq, const float* dk, const float* dv, float* dout) {
    fa_v3_warp_kernel<<<kQueryCount, kWarpSize>>>(dq, dk, dv, dout, kQueryCount, kKeyCount, kHeadDim);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
}

}  // namespace warp_streaming_ns

namespace pipeline_ns {

using ::kQueryCount;
using ::kKeyCount;
using ::kHeadDim;
using ::kEps;


constexpr int kWarpSize = 32;
constexpr int kPipelineStages = 2;

__device__ float warp_reduce_sum(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

// Same warp-per-row structure as v3/warp_streaming, but now K/V are staged
// through a two-slot shared-memory buffer.  While the warp scores key i, it
// also prefetches key i+1 into the alternate slot: preload -> consume ->
// preload next -> consume next.  Conceptual double buffer, not async-copy.
__global__ void fa_v3_pipeline_kernel(const float* q, const float* k, const float* v, float* out,
                                      int query_count, int key_count, int head_dim) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= query_count) return;

    __shared__ float k_stage[kPipelineStages][kHeadDim];
    __shared__ float v_stage[kPipelineStages][kHeadDim];
    __shared__ float running_max_shared, running_sum_shared;

    if (lane == 0) { running_max_shared = -1.0e30f; running_sum_shared = 0.0f; }

    float accum[kHeadDim];
    #pragma unroll
    for (int d = 0; d < kHeadDim; ++d) accum[d] = 0.0f;

    if (lane < head_dim) {
        k_stage[0][lane] = k[0 * head_dim + lane];
        v_stage[0][lane] = v[0 * head_dim + lane];
    }
    __syncthreads();

    for (int key = 0; key < key_count; ++key) {
        const int stage = key % kPipelineStages;
        const int next_stage = (key + 1) % kPipelineStages;
        if (key + 1 < key_count && lane < head_dim) {
            k_stage[next_stage][lane] = k[(key + 1) * head_dim + lane];
            v_stage[next_stage][lane] = v[(key + 1) * head_dim + lane];
        }
        __syncthreads();

        float partial = 0.0f;
        if (lane < head_dim) partial = q[row * head_dim + lane] * k_stage[stage][lane];
        const float score = __shfl_sync(0xffffffffu, warp_reduce_sum(partial), 0);

        float old_scale = 0.0f, weight = 0.0f;
        if (lane == 0) {
            const float new_max = (running_max_shared > score) ? running_max_shared : score;
            old_scale = (running_sum_shared == 0.0f) ? 0.0f : expf(running_max_shared - new_max);
            weight = expf(score - new_max);
            running_sum_shared = running_sum_shared * old_scale + weight;
            running_max_shared = new_max;
        }
        old_scale = __shfl_sync(0xffffffffu, old_scale, 0);
        weight = __shfl_sync(0xffffffffu, weight, 0);

        if (lane < head_dim)
            accum[lane] = accum[lane] * old_scale + weight * v_stage[stage][lane];
        __syncthreads();
    }

    const float final_sum = __shfl_sync(0xffffffffu, running_sum_shared, 0);
    if (lane < head_dim) out[row * head_dim + lane] = accum[lane] / (final_sum + kEps);
}

void launch_pipeline(const float* dq, const float* dk, const float* dv, float* dout) {
    fa_v3_pipeline_kernel<<<kQueryCount, kWarpSize>>>(dq, dk, dv, dout, kQueryCount, kKeyCount, kHeadDim);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
}

}  // namespace pipeline_ns

int main() {
    std::vector<float> q(kQueryCount * kHeadDim), k(kKeyCount * kHeadDim), v(kKeyCount * kHeadDim);
    std::vector<float> out(kQueryCount * kHeadDim);
    fill_inputs(q, k, v);
    float *dq, *dk, *dv, *dout;
    CHECK_CUDA(cudaMalloc(&dq, q.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dk, k.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dv, v.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dout, out.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dq, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dk, k.data(), k.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dv, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));

    warp_streaming_ns::launch_warp_streaming(dq, dk, dv, dout);
    pipeline_ns::launch_pipeline(dq, dk, dv, dout);

    CHECK_CUDA(cudaFree(dq));
    CHECK_CUDA(cudaFree(dk));
    CHECK_CUDA(cudaFree(dv));
    CHECK_CUDA(cudaFree(dout));
    std::cout << "flash_attention_v3 bench harness done\n";
    return EXIT_SUCCESS;
}
