// version: warp_streaming (one warp per row, per-key warp-reduced score, no smem staging)
//
// Baseline for chapter v3: replace shared-memory scoring with warp shuffles.
// Each key contributes exactly one score computed via __shfl_down_sync across
// the head-dim lanes; lane 0 holds the online-softmax state.

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

double dot_product(const float* a, const float* b, int dim) {
    double result = 0.0;
    for (int d = 0; d < dim; ++d) result += static_cast<double>(a[d]) * static_cast<double>(b[d]);
    return result;
}

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

void flash_attention_cpu(const std::vector<float>& q, const std::vector<float>& k,
                         const std::vector<float>& v, std::vector<float>& out) {
    for (int row = 0; row < kQueryCount; ++row) {
        double scores[kKeyCount];
        double row_max = -1.0e30;
        for (int key = 0; key < kKeyCount; ++key) {
            const double s = dot_product(&q[row * kHeadDim], &k[key * kHeadDim], kHeadDim);
            scores[key] = s;
            if (s > row_max) row_max = s;
        }
        double row_sum = 0.0;
        double accum[kHeadDim] = {0.0};
        for (int key = 0; key < kKeyCount; ++key) {
            const double w = std::exp(scores[key] - row_max);
            row_sum += w;
            for (int d = 0; d < kHeadDim; ++d) accum[d] += w * static_cast<double>(v[key * kHeadDim + d]);
        }
        for (int d = 0; d < kHeadDim; ++d) out[row * kHeadDim + d] = static_cast<float>(accum[d] / (row_sum + kEps));
    }
}

bool check(const std::vector<float>& got, const std::vector<float>& expected) {
    for (size_t i = 0; i < got.size(); ++i)
        if (std::fabs(got[i] - expected[i]) > 1e-4f) {
            std::cerr << "mismatch " << i << ": " << got[i] << " vs " << expected[i] << '\n';
            return false;
        }
    return true;
}

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

void launch(const float* dq, const float* dk, const float* dv, float* dout) {
    fa_v3_warp_kernel<<<kQueryCount, kWarpSize>>>(dq, dk, dv, dout, kQueryCount, kKeyCount, kHeadDim);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
}

}  // namespace

int main() {
    std::vector<float> q(kQueryCount * kHeadDim), k(kKeyCount * kHeadDim), v(kKeyCount * kHeadDim);
    std::vector<float> out(kQueryCount * kHeadDim, 0.0f), ref(kQueryCount * kHeadDim, 0.0f);
    fill_inputs(q, k, v);
    flash_attention_cpu(q, k, v, ref);

    float *dq, *dk, *dv, *dout;
    const size_t qb = q.size() * sizeof(float), kb = k.size() * sizeof(float);
    const size_t vb = v.size() * sizeof(float), ob = out.size() * sizeof(float);
    CHECK_CUDA(cudaMalloc(&dq, qb));
    CHECK_CUDA(cudaMalloc(&dk, kb));
    CHECK_CUDA(cudaMalloc(&dv, vb));
    CHECK_CUDA(cudaMalloc(&dout, ob));
    CHECK_CUDA(cudaMemcpy(dq, q.data(), qb, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dk, k.data(), kb, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dv, v.data(), vb, cudaMemcpyHostToDevice));

    launch(dq, dk, dv, dout);

    CHECK_CUDA(cudaMemcpy(out.data(), dout, ob, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(dq));
    CHECK_CUDA(cudaFree(dk));
    CHECK_CUDA(cudaFree(dv));
    CHECK_CUDA(cudaFree(dout));

    if (!check(out, ref)) return EXIT_FAILURE;
    std::cout << "flash_attention_v3 [warp_streaming] PASS  q=" << kQueryCount
              << " k=" << kKeyCount << " d=" << kHeadDim << '\n';
    return EXIT_SUCCESS;
}
