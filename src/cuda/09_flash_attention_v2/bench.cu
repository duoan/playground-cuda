// bench harness: runs every flash_attention_v2 version once for ncu.
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

namespace tile_staged_ns {

using ::kQueryCount;
using ::kKeyCount;
using ::kHeadDim;
using ::kEps;


constexpr int kTileKeys = 4;
constexpr int kThreadsPerBlock = kTileKeys;

// One block per query row. kTileKeys threads cooperatively load a K/V tile
// into shared memory; then thread 0 scores the tile and updates the online
// softmax state.  Same math as v1, but the shared-memory tile loading is
// spread across the whole block instead of one thread.
__global__ void fa_v2_kernel(const float* q, const float* k, const float* v, float* out,
                             int query_count, int key_count, int head_dim) {
    const int row = blockIdx.x;
    if (row >= query_count) return;

    __shared__ float shared_k[kTileKeys][kHeadDim];
    __shared__ float shared_v[kTileKeys][kHeadDim];
    __shared__ float shared_scores[kTileKeys];

    float running_max = -1.0e30f;
    float running_sum = 0.0f;
    float accum[kHeadDim];
    #pragma unroll
    for (int d = 0; d < kHeadDim; ++d) accum[d] = 0.0f;

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
            const int tile_count = (key_start + kTileKeys <= key_count) ? kTileKeys : (key_count - key_start);
            for (int t = 0; t < tile_count; ++t) {
                float s = 0.0f;
                for (int d = 0; d < kHeadDim; ++d) s += q[row * head_dim + d] * shared_k[t][d];
                shared_scores[t] = s;
                if (s > tile_max) tile_max = s;
            }
            const float new_max = (running_max > tile_max) ? running_max : tile_max;
            const float old_scale = (running_sum == 0.0f) ? 0.0f : expf(running_max - new_max);
            for (int d = 0; d < kHeadDim; ++d) accum[d] *= old_scale;
            float tile_sum = 0.0f;
            for (int t = 0; t < tile_count; ++t) {
                const float w = expf(shared_scores[t] - new_max);
                tile_sum += w;
                for (int d = 0; d < kHeadDim; ++d) accum[d] += w * shared_v[t][d];
            }
            running_sum = running_sum * old_scale + tile_sum;
            running_max = new_max;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
        for (int d = 0; d < kHeadDim; ++d) out[row * head_dim + d] = accum[d] / (running_sum + kEps);
}

void launch_tile_staged(const float* dq, const float* dk, const float* dv, float* dout) {
    fa_v2_kernel<<<kQueryCount, kThreadsPerBlock>>>(dq, dk, dv, dout, kQueryCount, kKeyCount, kHeadDim);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
}

}  // namespace tile_staged_ns

namespace warp_specialised_ns {

using ::kQueryCount;
using ::kKeyCount;
using ::kHeadDim;
using ::kEps;


constexpr int kTileKeys = 4;
constexpr int kWarpThreadsPerBlock = 32;

// One warp per query row.  Lane roles:
//   lanes < kHeadDim : load q, and later update accum[lane]
//   lanes < kTileKeys: load K/V tile rows and compute per-key partial scores
//   lane == 0        : run the online-softmax state update.
// This makes the FlashAttention v2 lane specialisation visible without warp
// primitives.
__global__ void fa_v2_warp_kernel(const float* q, const float* k, const float* v, float* out,
                                  int query_count, int key_count, int head_dim) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= query_count) return;

    __shared__ float q_shared[kHeadDim];
    __shared__ float shared_k[kTileKeys][kHeadDim];
    __shared__ float shared_v[kTileKeys][kHeadDim];
    __shared__ float shared_scores[kTileKeys];
    __shared__ float shared_weights[kTileKeys];
    __shared__ float running_max_shared, running_sum_shared;
    __shared__ float tile_max_shared, tile_sum_shared, old_scale_shared;

    if (lane < kHeadDim) q_shared[lane] = q[row * head_dim + lane];
    if (lane == 0) { running_max_shared = -1.0e30f; running_sum_shared = 0.0f; }
    __syncthreads();

    float accum[kHeadDim];
    #pragma unroll
    for (int d = 0; d < kHeadDim; ++d) accum[d] = 0.0f;

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
                for (int d = 0; d < kHeadDim; ++d) { shared_k[lane][d] = 0.0f; shared_v[lane][d] = 0.0f; }
            }
        }
        __syncthreads();

        if (lane < kTileKeys) {
            float s = 0.0f;
            const int key = key_start + lane;
            if (key < key_count) for (int d = 0; d < kHeadDim; ++d) s += q_shared[d] * shared_k[lane][d];
            else s = -1.0e30f;
            shared_scores[lane] = s;
        }
        __syncthreads();

        if (lane == 0) {
            tile_max_shared = -1.0e30f;
            const int tile_count = (key_start + kTileKeys <= key_count) ? kTileKeys : (key_count - key_start);
            for (int t = 0; t < tile_count; ++t) tile_max_shared = fmaxf(tile_max_shared, shared_scores[t]);
            const float new_max = fmaxf(running_max_shared, tile_max_shared);
            old_scale_shared = (running_sum_shared == 0.0f) ? 0.0f : expf(running_max_shared - new_max);
            tile_sum_shared = 0.0f;
            for (int t = 0; t < tile_count; ++t) {
                const float w = expf(shared_scores[t] - new_max);
                shared_weights[t] = w;
                tile_sum_shared += w;
            }
            for (int t = tile_count; t < kTileKeys; ++t) shared_weights[t] = 0.0f;
            running_sum_shared = running_sum_shared * old_scale_shared + tile_sum_shared;
            running_max_shared = new_max;
        }
        __syncthreads();

        if (lane < kHeadDim) {
            accum[lane] *= old_scale_shared;
            for (int t = 0; t < kTileKeys; ++t) accum[lane] += shared_weights[t] * shared_v[t][lane];
        }
        __syncthreads();
    }

    if (lane < kHeadDim) out[row * head_dim + lane] = accum[lane] / (running_sum_shared + kEps);
}

void launch_warp_specialised(const float* dq, const float* dk, const float* dv, float* dout) {
    fa_v2_warp_kernel<<<kQueryCount, kWarpThreadsPerBlock>>>(dq, dk, dv, dout, kQueryCount, kKeyCount, kHeadDim);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
}

}  // namespace warp_specialised_ns

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

    tile_staged_ns::launch_tile_staged(dq, dk, dv, dout);
    warp_specialised_ns::launch_warp_specialised(dq, dk, dv, dout);

    CHECK_CUDA(cudaFree(dq));
    CHECK_CUDA(cudaFree(dk));
    CHECK_CUDA(cudaFree(dv));
    CHECK_CUDA(cudaFree(dout));
    std::cout << "flash_attention_v2 bench harness done\n";
    return EXIT_SUCCESS;
}
