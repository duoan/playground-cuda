// bench harness: runs every attention version once for ncu.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../common/cuda_utils.cuh"

namespace {

// -D override: -DSEQ_LEN=N
#ifndef SEQ_LEN
#define SEQ_LEN 8
#endif
constexpr int kSeqLen = SEQ_LEN;
// -D override: -DHEAD_DIM=N
#ifndef HEAD_DIM
#define HEAD_DIM 8
#endif
constexpr int kHeadDim = HEAD_DIM;
constexpr int kTileTokens = 4;
__host__ __device__ inline float attn_scale() { return 1.0f / sqrtf(static_cast<float>(kHeadDim)); }

// naive
__global__ void attn_naive_scores_kernel(const float* q, const float* k, float* scores, bool causal) {
    const int row = blockIdx.x, col = threadIdx.x;
    if (row >= kSeqLen || col >= kSeqLen) return;
    if (!causal || col <= row) {
        float acc = 0.0f;
        for (int d = 0; d < kHeadDim; ++d) acc += q[row * kHeadDim + d] * k[col * kHeadDim + d];
        scores[row * kSeqLen + col] = acc * attn_scale();
    } else scores[row * kSeqLen + col] = -1e30f;
}
__global__ void attn_naive_softmax_kernel(float* scores) {
    const int row = blockIdx.x;
    if (row >= kSeqLen) return;
    __shared__ float s[kSeqLen];
    const int col = threadIdx.x;
    s[col] = scores[row * kSeqLen + col];
    __syncthreads();
    float m = s[0];
    for (int i = 1; i < kSeqLen; ++i) m = fmaxf(m, s[i]);
    float sum = 0.0f;
    for (int i = 0; i < kSeqLen; ++i) { s[i] = expf(s[i] - m); sum += s[i]; }
    scores[row * kSeqLen + col] = s[col] / sum;
}
__global__ void attn_naive_value_kernel(const float* p, const float* v, float* out) {
    const int row = blockIdx.x, d = threadIdx.x;
    if (row >= kSeqLen || d >= kHeadDim) return;
    float acc = 0.0f;
    for (int c = 0; c < kSeqLen; ++c) acc += p[row * kSeqLen + c] * v[c * kHeadDim + d];
    out[row * kHeadDim + d] = acc;
}

// tiled
__global__ void attn_tiled_kernel(const float* q, const float* k, const float* v, float* out, bool causal) {
    const int row = blockIdx.x, d = threadIdx.x;
    if (row >= kSeqLen || d >= kHeadDim) return;
    __shared__ float q_shared[kHeadDim];
    __shared__ float k_tile[kTileTokens][kHeadDim];
    __shared__ float v_tile[kTileTokens][kHeadDim];
    __shared__ float scores[kTileTokens];
    __shared__ float weights[kTileTokens];
    __shared__ float running_max_shared, running_sum_shared, tile_max_shared, tile_sum_shared, old_scale_shared;
    if (d < kHeadDim) q_shared[d] = q[row * kHeadDim + d];
    __syncthreads();
    if (d == 0) { running_max_shared = -1e30f; running_sum_shared = 0.0f; }
    __syncthreads();
    float acc = 0.0f;
    for (int tile_start = 0; tile_start < kSeqLen; tile_start += kTileTokens) {
        for (int t = 0; t < kTileTokens; ++t) {
            const int si = tile_start + t;
            if (si < kSeqLen) { k_tile[t][d] = k[si * kHeadDim + d]; v_tile[t][d] = v[si * kHeadDim + d]; }
            else { k_tile[t][d] = 0.0f; v_tile[t][d] = 0.0f; }
        }
        __syncthreads();
        if (d == 0) {
            tile_max_shared = -1e30f;
            for (int t = 0; t < kTileTokens; ++t) {
                const int si = tile_start + t;
                const bool ok = si < kSeqLen && (!causal || si <= row);
                if (ok) {
                    float s = 0.0f;
                    for (int i = 0; i < kHeadDim; ++i) s += q_shared[i] * k_tile[t][i];
                    s *= attn_scale();
                    scores[t] = s;
                    tile_max_shared = fmaxf(tile_max_shared, s);
                } else scores[t] = -1e30f;
            }
            tile_sum_shared = 0.0f;
            for (int t = 0; t < kTileTokens; ++t)
                if (tile_start + t < kSeqLen) tile_sum_shared += expf(scores[t] - tile_max_shared);
            const float new_max = fmaxf(running_max_shared, tile_max_shared);
            old_scale_shared = expf(running_max_shared - new_max);
            running_sum_shared = running_sum_shared * old_scale_shared + tile_sum_shared * expf(tile_max_shared - new_max);
            running_max_shared = new_max;
            for (int t = 0; t < kTileTokens; ++t)
                weights[t] = (tile_start + t < kSeqLen) ? expf(scores[t] - new_max) : 0.0f;
        }
        __syncthreads();
        acc = acc * old_scale_shared;
        for (int t = 0; t < kTileTokens; ++t) acc += weights[t] * v_tile[t][d];
        __syncthreads();
    }
    out[row * kHeadDim + d] = acc / running_sum_shared;
}

void fill(std::vector<float>& q, std::vector<float>& k, std::vector<float>& v) {
    for (int i = 0; i < static_cast<int>(q.size()); ++i) q[i] = static_cast<float>((i % 5) - 2) * 0.2f;
    for (int i = 0; i < static_cast<int>(k.size()); ++i) k[i] = static_cast<float>((i % 7) - 3) * 0.15f;
    for (int i = 0; i < static_cast<int>(v.size()); ++i) v[i] = static_cast<float>((i % 6) - 2) * 0.1f;
}

}  // namespace

int main() {
    std::vector<float> q(kSeqLen * kHeadDim), k(kSeqLen * kHeadDim), v(kSeqLen * kHeadDim);
    std::vector<float> out(kSeqLen * kHeadDim);
    fill(q, k, v);
    float *dq, *dk, *dv, *dscores, *dout;
    CHECK_CUDA(cudaMalloc(&dq, q.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dk, k.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dv, v.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dscores, kSeqLen * kSeqLen * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dout, out.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dq, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dk, k.data(), k.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dv, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));

    attn_naive_scores_kernel<<<kSeqLen, kSeqLen>>>(dq, dk, dscores, false);
    attn_naive_softmax_kernel<<<kSeqLen, kSeqLen>>>(dscores);
    attn_naive_value_kernel<<<kSeqLen, kHeadDim>>>(dscores, dv, dout);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    attn_tiled_kernel<<<kSeqLen, kHeadDim>>>(dq, dk, dv, dout, false);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaFree(dq));
    CHECK_CUDA(cudaFree(dk));
    CHECK_CUDA(cudaFree(dv));
    CHECK_CUDA(cudaFree(dscores));
    CHECK_CUDA(cudaFree(dout));
    std::cout << "attention bench harness done\n";
    return EXIT_SUCCESS;
}
