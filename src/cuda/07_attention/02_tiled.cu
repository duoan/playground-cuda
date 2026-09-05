// version: tiled (single kernel, online softmax, no [seq x seq] materialisation)
//
// One block per query row. Sweep K/V a tile at a time, maintain
// (running_max, running_sum, acc) with the online-softmax rescale trick, and
// emit the final normalised output at the end.  Zero global-memory bytes for
// the intermediate score matrix.

#include <algorithm>
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

__global__ void attention_tiled_kernel(const float* q, const float* k, const float* v,
                                       float* out, bool causal) {
    const int row = blockIdx.x;
    const int d = threadIdx.x;
    if (row >= kSeqLen || d >= kHeadDim) return;

    __shared__ float q_shared[kHeadDim];
    __shared__ float k_tile[kTileTokens][kHeadDim];
    __shared__ float v_tile[kTileTokens][kHeadDim];
    __shared__ float scores[kTileTokens];
    __shared__ float weights[kTileTokens];
    __shared__ float running_max_shared;
    __shared__ float running_sum_shared;
    __shared__ float tile_max_shared;
    __shared__ float tile_sum_shared;
    __shared__ float old_scale_shared;

    if (d < kHeadDim) q_shared[d] = q[row * kHeadDim + d];
    __syncthreads();

    if (d == 0) {
        running_max_shared = -1e30f;
        running_sum_shared = 0.0f;
    }
    __syncthreads();

    float acc = 0.0f;

    for (int tile_start = 0; tile_start < kSeqLen; tile_start += kTileTokens) {
        for (int token = 0; token < kTileTokens; ++token) {
            const int seq_idx = tile_start + token;
            if (seq_idx < kSeqLen) {
                k_tile[token][d] = k[seq_idx * kHeadDim + d];
                v_tile[token][d] = v[seq_idx * kHeadDim + d];
            } else {
                k_tile[token][d] = 0.0f;
                v_tile[token][d] = 0.0f;
            }
        }
        __syncthreads();

        if (d == 0) {
            tile_max_shared = -1e30f;
            for (int token = 0; token < kTileTokens; ++token) {
                const int seq_idx = tile_start + token;
                const bool allowed = seq_idx < kSeqLen && (!causal || seq_idx <= row);
                if (allowed) {
                    float score = 0.0f;
                    for (int i = 0; i < kHeadDim; ++i) score += q_shared[i] * k_tile[token][i];
                    score *= attn_scale();
                    scores[token] = score;
                    tile_max_shared = fmaxf(tile_max_shared, score);
                } else {
                    scores[token] = -1e30f;
                }
            }
            tile_sum_shared = 0.0f;
            for (int token = 0; token < kTileTokens; ++token)
                if (tile_start + token < kSeqLen)
                    tile_sum_shared += expf(scores[token] - tile_max_shared);
            const float new_max = fmaxf(running_max_shared, tile_max_shared);
            old_scale_shared = expf(running_max_shared - new_max);
            running_sum_shared = running_sum_shared * old_scale_shared +
                                 tile_sum_shared * expf(tile_max_shared - new_max);
            running_max_shared = new_max;
            for (int token = 0; token < kTileTokens; ++token) {
                if (tile_start + token < kSeqLen)
                    weights[token] = expf(scores[token] - new_max);
                else
                    weights[token] = 0.0f;
            }
        }
        __syncthreads();

        acc = acc * old_scale_shared;
        for (int token = 0; token < kTileTokens; ++token) acc += weights[token] * v_tile[token][d];
        __syncthreads();
    }

    out[row * kHeadDim + d] = acc / running_sum_shared;
}

void launch(const float* dq, const float* dk, const float* dv, float* /*dscores*/, float* dout, bool causal) {
    attention_tiled_kernel<<<kSeqLen, kHeadDim>>>(dq, dk, dv, dout, causal);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
}

void fill(std::vector<float>& q, std::vector<float>& k, std::vector<float>& v) {
    for (int i = 0; i < static_cast<int>(q.size()); ++i) q[i] = static_cast<float>((i % 5) - 2) * 0.2f;
    for (int i = 0; i < static_cast<int>(k.size()); ++i) k[i] = static_cast<float>((i % 7) - 3) * 0.15f;
    for (int i = 0; i < static_cast<int>(v.size()); ++i) v[i] = static_cast<float>((i % 6) - 2) * 0.1f;
}

double dot_row(const std::vector<float>& a, int ar, const std::vector<float>& b, int br) {
    double acc = 0.0;
    for (int d = 0; d < kHeadDim; ++d)
        acc += static_cast<double>(a[ar * kHeadDim + d]) * static_cast<double>(b[br * kHeadDim + d]);
    return acc;
}

void attention_cpu(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
                   std::vector<float>& out, bool causal) {
    std::vector<double> scores(kSeqLen * kSeqLen, 0.0), probs(kSeqLen * kSeqLen, 0.0);
    for (int row = 0; row < kSeqLen; ++row) {
        double row_max = -1e30;
        for (int col = 0; col < kSeqLen; ++col) {
            const bool ok = !causal || col <= row;
            const double s = ok ? dot_row(q, row, k, col) * static_cast<double>(attn_scale()) : -1e30;
            scores[row * kSeqLen + col] = s;
            row_max = std::max(row_max, s);
        }
        double row_sum = 0.0;
        for (int col = 0; col < kSeqLen; ++col) {
            const double e = std::exp(scores[row * kSeqLen + col] - row_max);
            probs[row * kSeqLen + col] = e;
            row_sum += e;
        }
        for (int d = 0; d < kHeadDim; ++d) {
            double acc = 0.0;
            for (int col = 0; col < kSeqLen; ++col)
                acc += (probs[row * kSeqLen + col] / row_sum) * static_cast<double>(v[col * kHeadDim + d]);
            out[row * kHeadDim + d] = static_cast<float>(acc);
        }
    }
}

bool check(const std::vector<float>& g, const std::vector<float>& e) {
    for (size_t i = 0; i < g.size(); ++i)
        if (std::fabs(g[i] - e[i]) > 1e-4f) {
            std::cerr << "mismatch " << i << ": " << g[i] << " vs " << e[i] << '\n';
            return false;
        }
    return true;
}

void run(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
         std::vector<float>& out, bool causal) {
    float *dq, *dk, *dv, *dout;
    CHECK_CUDA(cudaMalloc(&dq, q.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dk, k.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dv, v.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dout, out.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dq, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dk, k.data(), k.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dv, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
    launch(dq, dk, dv, nullptr, dout, causal);
    CHECK_CUDA(cudaMemcpy(out.data(), dout, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(dq));
    CHECK_CUDA(cudaFree(dk));
    CHECK_CUDA(cudaFree(dv));
    CHECK_CUDA(cudaFree(dout));
}

}  // namespace

int main() {
    std::vector<float> q(kSeqLen * kHeadDim), k(kSeqLen * kHeadDim), v(kSeqLen * kHeadDim);
    std::vector<float> out(kSeqLen * kHeadDim, 0.0f), out_c(kSeqLen * kHeadDim, 0.0f);
    std::vector<float> ref(kSeqLen * kHeadDim, 0.0f), ref_c(kSeqLen * kHeadDim, 0.0f);
    fill(q, k, v);
    attention_cpu(q, k, v, ref, false);
    attention_cpu(q, k, v, ref_c, true);
    run(q, k, v, out, false);
    run(q, k, v, out_c, true);
    if (!check(out, ref) || !check(out_c, ref_c)) return EXIT_FAILURE;
    std::cout << "attention [tiled] PASS  seq=" << kSeqLen << " head=" << kHeadDim << '\n';
    return EXIT_SUCCESS;
}
