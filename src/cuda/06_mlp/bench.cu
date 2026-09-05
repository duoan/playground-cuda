// bench harness: runs every mlp version once for ncu.
// Kernels are byte-for-byte copies of the per-version files; only the
// kernel names get per-version suffixes so the ncu regex can group them.

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
constexpr int kThreadsPerBlock = 256;
constexpr int kTiledThreadsPerBlock = 32;
constexpr int kInputTile = 4;

__host__ __device__ inline float relu(float v) { return v > 0.0f ? v : 0.0f; }

// naive
__global__ void mlp_naive_linear1_kernel(const float* x, const float* w1, const float* b1, float* h) {
    const int b = blockIdx.x, hi = threadIdx.x;
    if (b >= kBatch || hi >= kHiddenDim) return;
    float acc = b1[hi];
    for (int i = 0; i < kInputDim; ++i) acc += x[b * kInputDim + i] * w1[i * kHiddenDim + hi];
    h[b * kHiddenDim + hi] = acc;
}
__global__ void mlp_naive_relu_kernel(float* v, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = relu(v[i]);
}
__global__ void mlp_naive_linear2_kernel(const float* h, const float* w2, const float* b2, float* y) {
    const int b = blockIdx.x, o = threadIdx.x;
    if (b >= kBatch || o >= kOutputDim) return;
    float acc = b2[o];
    for (int i = 0; i < kHiddenDim; ++i) acc += h[b * kHiddenDim + i] * w2[i * kOutputDim + o];
    y[b * kOutputDim + o] = acc;
}

// fused
__global__ void mlp_fused_linear1_relu_kernel(const float* x, const float* w1, const float* b1, float* h) {
    const int b = blockIdx.x, hi = threadIdx.x;
    if (b >= kBatch || hi >= kHiddenDim) return;
    float acc = b1[hi];
    for (int i = 0; i < kInputDim; ++i) acc += x[b * kInputDim + i] * w1[i * kHiddenDim + hi];
    h[b * kHiddenDim + hi] = relu(acc);
}
__global__ void mlp_fused_linear2_kernel(const float* h, const float* w2, const float* b2, float* y) {
    const int b = blockIdx.x, o = threadIdx.x;
    if (b >= kBatch || o >= kOutputDim) return;
    float acc = b2[o];
    for (int i = 0; i < kHiddenDim; ++i) acc += h[b * kHiddenDim + i] * w2[i * kOutputDim + o];
    y[b * kOutputDim + o] = acc;
}

// tiled_fused
__global__ void mlp_tiled_fused_kernel(const float* x, const float* w1, const float* b1,
                                       const float* w2, const float* b2, float* y) {
    __shared__ float x_tile[kInputTile];
    __shared__ float hidden_shared[kHiddenDim];
    const int b = blockIdx.x, tid = threadIdx.x;
    if (b >= kBatch) return;
    float acc = 0.0f;
    if (tid < kHiddenDim) acc = b1[tid];
    for (int tile = 0; tile < kInputDim; tile += kInputTile) {
        if (tid < kInputTile) {
            const int ii = tile + tid;
            x_tile[tid] = (ii < kInputDim) ? x[b * kInputDim + ii] : 0.0f;
        }
        __syncthreads();
        if (tid < kHiddenDim) {
            #pragma unroll
            for (int i = 0; i < kInputTile; ++i) {
                const int ii = tile + i;
                if (ii < kInputDim) acc += x_tile[i] * w1[ii * kHiddenDim + tid];
            }
        }
        __syncthreads();
    }
    if (tid < kHiddenDim) hidden_shared[tid] = relu(acc);
    __syncthreads();
    if (tid < kOutputDim) {
        float o_acc = b2[tid];
        for (int i = 0; i < kHiddenDim; ++i) o_acc += hidden_shared[i] * w2[i * kOutputDim + tid];
        y[b * kOutputDim + tid] = o_acc;
    }
}

void fill(std::vector<float>& x, std::vector<float>& w1, std::vector<float>& b1,
          std::vector<float>& w2, std::vector<float>& b2) {
    for (int i = 0; i < static_cast<int>(x.size()); ++i) x[i] = static_cast<float>((i % 5) - 2);
    for (int i = 0; i < static_cast<int>(w1.size()); ++i) w1[i] = static_cast<float>((i % 7) - 3) * 0.1f;
    for (int i = 0; i < static_cast<int>(b1.size()); ++i) b1[i] = static_cast<float>((i % 3) - 1) * 0.05f;
    for (int i = 0; i < static_cast<int>(w2.size()); ++i) w2[i] = static_cast<float>((i % 5) - 2) * 0.08f;
    for (int i = 0; i < static_cast<int>(b2.size()); ++i) b2[i] = static_cast<float>((i % 4) - 1) * 0.03f;
}

}  // namespace

int main() {
    std::vector<float> x(kBatch * kInputDim), w1(kInputDim * kHiddenDim), b1(kHiddenDim);
    std::vector<float> w2(kHiddenDim * kOutputDim), b2(kOutputDim);
    std::vector<float> y(kBatch * kOutputDim, 0.0f);
    fill(x, w1, b1, w2, b2);
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

    mlp_naive_linear1_kernel<<<kBatch, kHiddenDim>>>(dx, dw1, db1, dhidden);
    mlp_naive_relu_kernel<<<cuda_utils::ceil_div(kBatch * kHiddenDim, kThreadsPerBlock), kThreadsPerBlock>>>(dhidden, kBatch * kHiddenDim);
    mlp_naive_linear2_kernel<<<kBatch, kOutputDim>>>(dhidden, dw2, db2, dy);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    mlp_fused_linear1_relu_kernel<<<kBatch, kHiddenDim>>>(dx, dw1, db1, dhidden);
    mlp_fused_linear2_kernel<<<kBatch, kOutputDim>>>(dhidden, dw2, db2, dy);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    mlp_tiled_fused_kernel<<<kBatch, kTiledThreadsPerBlock>>>(dx, dw1, db1, dw2, db2, dy);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dw1));
    CHECK_CUDA(cudaFree(db1));
    CHECK_CUDA(cudaFree(dw2));
    CHECK_CUDA(cudaFree(db2));
    CHECK_CUDA(cudaFree(dhidden));
    CHECK_CUDA(cudaFree(dy));
    std::cout << "mlp bench harness done\n";
    return EXIT_SUCCESS;
}
