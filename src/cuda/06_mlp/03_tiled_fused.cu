// version: tiled_fused (single kernel — input tile in smem, hidden kept in smem, both matmuls in one launch)
//
// One block per batch row: stage inputs into shared memory a tile at a time,
// accumulate the first matmul + relu, then read the hidden activations back
// out of shared memory to feed the second matmul.  Zero global-memory
// round-trips for the intermediate hidden state.
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../common/cuda_utils.cuh"

namespace {

// -D override: -DBATCH=N
#ifndef BATCH
#define BATCH 4
#endif
constexpr int kBatch = BATCH;
// -D override: -DINPUT_DIM=N
#ifndef INPUT_DIM
#define INPUT_DIM 8
#endif
constexpr int kInputDim = INPUT_DIM;
// -D override: -DHIDDEN_DIM=N
#ifndef HIDDEN_DIM
#define HIDDEN_DIM 16
#endif
constexpr int kHiddenDim = HIDDEN_DIM;
// -D override: -DOUTPUT_DIM=N
#ifndef OUTPUT_DIM
#define OUTPUT_DIM 4
#endif
constexpr int kOutputDim = OUTPUT_DIM;
__host__ __device__ inline float relu(float value) {
    return value > 0.0f ? value : 0.0f;
}

constexpr int kTiledThreadsPerBlock = (kHiddenDim > kOutputDim ? kHiddenDim : kOutputDim);
constexpr int kInputTile = 4;

__global__ void mlp_tiled_fused_kernel(const float* x, const float* w1, const float* b1,
                                       const float* w2, const float* b2, float* y) {
    __shared__ float x_tile[kInputTile];
    __shared__ float hidden_shared[kHiddenDim];
    const int batch = blockIdx.x;
    const int tid = threadIdx.x;
    if (batch >= kBatch) return;

    float hidden_acc = 0.0f;
    if (tid < kHiddenDim) hidden_acc = b1[tid];

    for (int tile = 0; tile < kInputDim; tile += kInputTile) {
        if (tid < kInputTile) {
            const int input_idx = tile + tid;
            x_tile[tid] = (input_idx < kInputDim) ? x[batch * kInputDim + input_idx] : 0.0f;
        }
        __syncthreads();
        if (tid < kHiddenDim) {
            #pragma unroll
            for (int i = 0; i < kInputTile; ++i) {
                const int input_idx = tile + i;
                if (input_idx < kInputDim) hidden_acc += x_tile[i] * w1[input_idx * kHiddenDim + tid];
            }
        }
        __syncthreads();
    }
    if (tid < kHiddenDim) hidden_shared[tid] = relu(hidden_acc);
    __syncthreads();

    if (tid < kOutputDim) {
        float acc = b2[tid];
        for (int i = 0; i < kHiddenDim; ++i) acc += hidden_shared[i] * w2[i * kOutputDim + tid];
        y[batch * kOutputDim + tid] = acc;
    }
}

void launch(const float* dx, const float* dw1, const float* db1, const float* dw2, const float* db2,
            float* /*dhidden*/, float* dy) {
    mlp_tiled_fused_kernel<<<kBatch, kTiledThreadsPerBlock>>>(dx, dw1, db1, dw2, db2, dy);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
}

void fill_inputs(std::vector<float>& x) {
    for (int i = 0; i < static_cast<int>(x.size()); ++i) x[i] = static_cast<float>((i % 5) - 2);
}

void fill_weights(std::vector<float>& w1, std::vector<float>& b1, std::vector<float>& w2, std::vector<float>& b2) {
    for (int i = 0; i < static_cast<int>(w1.size()); ++i) w1[i] = static_cast<float>((i % 7) - 3) * 0.1f;
    for (int i = 0; i < static_cast<int>(b1.size()); ++i) b1[i] = static_cast<float>((i % 3) - 1) * 0.05f;
    for (int i = 0; i < static_cast<int>(w2.size()); ++i) w2[i] = static_cast<float>((i % 5) - 2) * 0.08f;
    for (int i = 0; i < static_cast<int>(b2.size()); ++i) b2[i] = static_cast<float>((i % 4) - 1) * 0.03f;
}

void mlp_cpu(const std::vector<float>& x, const std::vector<float>& w1, const std::vector<float>& b1,
             const std::vector<float>& w2, const std::vector<float>& b2, std::vector<float>& y) {
    std::vector<float> h(kBatch * kHiddenDim, 0.0f);
    for (int b = 0; b < kBatch; ++b)
        for (int hi = 0; hi < kHiddenDim; ++hi) {
            float acc = b1[hi];
            for (int i = 0; i < kInputDim; ++i) acc += x[b * kInputDim + i] * w1[i * kHiddenDim + hi];
            h[b * kHiddenDim + hi] = relu(acc);
        }
    for (int b = 0; b < kBatch; ++b)
        for (int o = 0; o < kOutputDim; ++o) {
            float acc = b2[o];
            for (int hi = 0; hi < kHiddenDim; ++hi) acc += h[b * kHiddenDim + hi] * w2[hi * kOutputDim + o];
            y[b * kOutputDim + o] = acc;
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

}  // namespace

int main() {
    std::vector<float> x(kBatch * kInputDim), w1(kInputDim * kHiddenDim), b1(kHiddenDim);
    std::vector<float> w2(kHiddenDim * kOutputDim), b2(kOutputDim);
    std::vector<float> y(kBatch * kOutputDim, 0.0f), ref(kBatch * kOutputDim, 0.0f);
    fill_inputs(x);
    fill_weights(w1, b1, w2, b2);
    mlp_cpu(x, w1, b1, w2, b2, ref);

    float *dx, *dw1, *db1, *dw2, *db2, *dhidden, *dy;
    CHECK_CUDA(cudaMalloc(&dx, x.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dw1, w1.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&db1, b1.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dw2, w2.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&db2, b2.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dhidden, kBatch * kHiddenDim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dy, y.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw1, w1.data(), w1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(db1, b1.data(), b1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw2, w2.data(), w2.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(db2, b2.data(), b2.size() * sizeof(float), cudaMemcpyHostToDevice));

    launch(dx, dw1, db1, dw2, db2, dhidden, dy);

    CHECK_CUDA(cudaMemcpy(y.data(), dy, y.size() * sizeof(float), cudaMemcpyDeviceToHost));
    if (!check(y, ref)) return EXIT_FAILURE;
    std::cout << "mlp [tiled_fused] PASS  batch=" << kBatch << " hidden=" << kHiddenDim << '\n';

    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dw1));
    CHECK_CUDA(cudaFree(db1));
    CHECK_CUDA(cudaFree(dw2));
    CHECK_CUDA(cudaFree(db2));
    CHECK_CUDA(cudaFree(dhidden));
    CHECK_CUDA(cudaFree(dy));
    return EXIT_SUCCESS;
}
