// practice: attention version = tiled
// TODO: One kernel, block per query row. Sweep K/V tiles, keep online (max,sum,acc).
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../../common/cuda_utils.cuh"

namespace {

constexpr int kSeqLen = 8;
constexpr int kHeadDim = 8;
constexpr int kTileTokens = 4;
__host__ __device__ inline float attn_scale() { return 1.0f / sqrtf(static_cast<float>(kHeadDim)); }

// TODO: kernel(s) for this version.

// TODO: launch. See ../../07_attention/02_tiled.cu for the reference.
void launch(const float* dq, const float* dk, const float* dv, float* dscores, float* dout, bool causal) {
    (void)dq; (void)dk; (void)dv; (void)dscores; (void)dout; (void)causal;
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
    std::cout << "attention [tiled] practice PASS  seq=" << kSeqLen << " head=" << kHeadDim << '\n';
    return EXIT_SUCCESS;
}
