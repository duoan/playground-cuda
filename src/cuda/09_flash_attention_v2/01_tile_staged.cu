// version: tile_staged (block-cooperative K/V tile loading, single-thread online state)
//
// Between v1/shared and v2/warp: the whole block loads K/V into shared memory
// (kTileKeys threads, one per row of the tile), then thread 0 runs the online
// softmax update.  Makes the "cooperative load + serial state update" split
// explicit before we move to a proper warp-specialized version.

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

void launch(const float* dq, const float* dk, const float* dv, float* dout) {
    fa_v2_kernel<<<kQueryCount, kThreadsPerBlock>>>(dq, dk, dv, dout, kQueryCount, kKeyCount, kHeadDim);
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
    std::cout << "flash_attention_v2 [tile_staged] PASS  q=" << kQueryCount
              << " k=" << kKeyCount << " d=" << kHeadDim << '\n';
    return EXIT_SUCCESS;
}
