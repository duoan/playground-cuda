// bench harness: runs every flash_attention_v1 version once for ncu.
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

namespace online_ns {

using ::kQueryCount;
using ::kKeyCount;
using ::kHeadDim;
using ::kEps;


constexpr int kTileKeys = 4;
constexpr int kThreadsPerBlock = 32;

__device__ float dot_row_device(const float* a, const float* b, int dim) {
    float r = 0.0f;
    for (int d = 0; d < dim; ++d) r += a[d] * b[d];
    return r;
}

// One thread streams the keys for one query row. All state (max, sum, accum)
// stays in registers; nothing is written back except the final output.
__global__ void fa_v1_online_kernel(const float* q, const float* k, const float* v, float* out,
                                    int query_count, int key_count, int head_dim) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= query_count) return;

    float running_max = -1.0e30f;
    float running_sum = 0.0f;
    float accum[kHeadDim];
    #pragma unroll
    for (int d = 0; d < kHeadDim; ++d) accum[d] = 0.0f;

    float scores[kTileKeys];

    for (int key_start = 0; key_start < key_count; key_start += kTileKeys) {
        const int tile_count = (key_start + kTileKeys <= key_count) ? kTileKeys : (key_count - key_start);
        float tile_max = -1.0e30f;
        #pragma unroll
        for (int t = 0; t < kTileKeys; ++t) scores[t] = 0.0f;
        for (int t = 0; t < tile_count; ++t) {
            const int key = key_start + t;
            const float s = dot_row_device(&q[row * head_dim], &k[key * head_dim], head_dim);
            scores[t] = s;
            if (s > tile_max) tile_max = s;
        }
        const float new_max = (running_max > tile_max) ? running_max : tile_max;
        const float old_scale = (running_sum == 0.0f) ? 0.0f : expf(running_max - new_max);
        for (int d = 0; d < kHeadDim; ++d) accum[d] *= old_scale;
        float tile_sum = 0.0f;
        for (int t = 0; t < tile_count; ++t) {
            const float w = expf(scores[t] - new_max);
            tile_sum += w;
            const int key = key_start + t;
            for (int d = 0; d < kHeadDim; ++d) accum[d] += w * v[key * head_dim + d];
        }
        running_sum = running_sum * old_scale + tile_sum;
        running_max = new_max;
    }

    for (int d = 0; d < kHeadDim; ++d) out[row * head_dim + d] = accum[d] / (running_sum + kEps);
}

void launch_online(const float* dq, const float* dk, const float* dv, float* dout) {
    const int blocks = cuda_utils::ceil_div(kQueryCount, kThreadsPerBlock);
    fa_v1_online_kernel<<<blocks, kThreadsPerBlock>>>(dq, dk, dv, dout, kQueryCount, kKeyCount, kHeadDim);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
}

}  // namespace online_ns

namespace shared_ns {

using ::kQueryCount;
using ::kKeyCount;
using ::kHeadDim;
using ::kEps;


constexpr int kTileKeys = 4;
constexpr int kThreadsPerBlock = 8;

// One block per query row. K/V tiles staged to shared memory. Same online
// softmax math as v1, but data reuse now lives in SRAM instead of registers.
__global__ void fa_v1_shared_kernel(const float* q, const float* k, const float* v, float* out,
                                    int query_count, int key_count, int head_dim) {
    const int row = blockIdx.x;
    if (row >= query_count) return;

    __shared__ float q_shared[kHeadDim];
    __shared__ float k_tile[kTileKeys][kHeadDim];
    __shared__ float v_tile[kTileKeys][kHeadDim];
    __shared__ float scores[kTileKeys];
    __shared__ float running_max_shared;
    __shared__ float running_sum_shared;

    if (threadIdx.x < kHeadDim) q_shared[threadIdx.x] = q[row * head_dim + threadIdx.x];
    if (threadIdx.x == 0) { running_max_shared = -1.0e30f; running_sum_shared = 0.0f; }
    __syncthreads();

    float accum[kHeadDim];
    #pragma unroll
    for (int d = 0; d < kHeadDim; ++d) accum[d] = 0.0f;

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
            const int tile_count = (key_start + kTileKeys <= key_count) ? kTileKeys : (key_count - key_start);
            for (int t = 0; t < tile_count; ++t) {
                float s = 0.0f;
                for (int d = 0; d < kHeadDim; ++d) s += q_shared[d] * k_tile[t][d];
                scores[t] = s;
                tile_max = fmaxf(tile_max, s);
            }
            const float new_max = fmaxf(running_max_shared, tile_max);
            const float old_scale = (running_sum_shared == 0.0f) ? 0.0f : expf(running_max_shared - new_max);
            for (int d = 0; d < kHeadDim; ++d) accum[d] *= old_scale;
            float tile_sum = 0.0f;
            for (int t = 0; t < tile_count; ++t) {
                const float w = expf(scores[t] - new_max);
                tile_sum += w;
                for (int d = 0; d < kHeadDim; ++d) accum[d] += w * v_tile[t][d];
            }
            running_sum_shared = running_sum_shared * old_scale + tile_sum;
            running_max_shared = new_max;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
        for (int d = 0; d < kHeadDim; ++d) out[row * head_dim + d] = accum[d] / (running_sum_shared + kEps);
}

void launch_shared(const float* dq, const float* dk, const float* dv, float* dout) {
    fa_v1_shared_kernel<<<kQueryCount, kThreadsPerBlock>>>(dq, dk, dv, dout, kQueryCount, kKeyCount, kHeadDim);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
}

}  // namespace shared_ns

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

    online_ns::launch_online(dq, dk, dv, dout);
    shared_ns::launch_shared(dq, dk, dv, dout);

    CHECK_CUDA(cudaFree(dq));
    CHECK_CUDA(cudaFree(dk));
    CHECK_CUDA(cudaFree(dv));
    CHECK_CUDA(cudaFree(dout));
    std::cout << "flash_attention_v1 bench harness done\n";
    return EXIT_SUCCESS;
}
