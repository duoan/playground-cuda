// practice: mlp version = naive
// TODO: Three kernels: linear1 -> relu -> linear2.
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../../common/cuda_utils.cuh"

namespace {

constexpr int kBatch = 4;
constexpr int kInputDim = 8;
constexpr int kHiddenDim = 16;
constexpr int kOutputDim = 4;
constexpr int kThreadsPerBlock = 256;

__host__ __device__ inline float relu(float value) {
    return value > 0.0f ? value : 0.0f;
}

// TODO: kernel(s) for this version.

// TODO: launch. See the reference in ../../06_mlp/01_naive.cu once you're done.
void launch(const float* dx, const float* dw1, const float* db1, const float* dw2, const float* db2,
            float* dhidden, float* dy) {
    (void)dx; (void)dw1; (void)db1; (void)dw2; (void)db2; (void)dhidden; (void)dy;
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
    std::cout << "mlp [naive] practice PASS  batch=" << kBatch << " hidden=" << kHiddenDim << '\n';

    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dw1));
    CHECK_CUDA(cudaFree(db1));
    CHECK_CUDA(cudaFree(dw2));
    CHECK_CUDA(cudaFree(db2));
    CHECK_CUDA(cudaFree(dhidden));
    CHECK_CUDA(cudaFree(dy));
    return EXIT_SUCCESS;
}
