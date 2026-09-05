// practice: matmul version = cublas (K0)
//
// The "how fast is fast" baseline. Call cublasSgemm (or cublasGemmEx
// with CUBLAS_COMPUTE_32F_FAST_TF32) — the vendor library dispatches
// to a CUTLASS-based tensor-core GEMM.
//
// The one non-trivial detail: cuBLAS is column-major. We have row-major
//   A(m×k) * B(k×n) = C(m×n)
// The identity  (A B)^T = B^T A^T  becomes, in column-major:
//   pass B and A swapped, and swap m ↔ n:
//     cublasSgemm(handle, N, N, n, m, k, α, B, n, A, k, β, C, n)

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cublas_v2.h>

#include "../../common/cuda_utils.cuh"

namespace {

#define CHECK_CUBLAS(x) do {                                          \
    cublasStatus_t s = (x);                                           \
    if (s != CUBLAS_STATUS_SUCCESS) {                                 \
        std::cerr << "cuBLAS error " << int(s) << " at "              \
                  << __FILE__ << ":" << __LINE__ << '\n';             \
        std::abort();                                                 \
    }                                                                 \
} while (0)

// TODO: create a static handle on first call, enable TF32 tensor cores,
// then dispatch cublasSgemm with the swapped-argument trick above.
void launch(const float* a, const float* b, float* c, int m, int n, int k) {
    (void)a; (void)b; (void)c; (void)m; (void)n; (void)k;
}

// ---- host boilerplate: identical across every practice version ----

void fill_inputs(std::vector<float>& a, std::vector<float>& b, int m, int n, int k) {
    for (int row = 0; row < m; ++row)
        for (int col = 0; col < k; ++col)
            a[row * k + col] = static_cast<float>((row + col) % 7);
    for (int row = 0; row < k; ++row)
        for (int col = 0; col < n; ++col)
            b[row * n + col] = static_cast<float>((row * 2 + col) % 5);
}

void matmul_cpu(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c,
                int m, int n, int k) {
    for (int row = 0; row < m; ++row)
        for (int col = 0; col < n; ++col) {
            float acc = 0.0f;
            for (int inner = 0; inner < k; ++inner)
                acc += a[row * k + inner] * b[inner * n + col];
            c[row * n + col] = acc;
        }
}

bool check_output(const std::vector<float>& got, const std::vector<float>& expected) {
    // TF32 has 10 mantissa bits; use a relative tolerance ~5e-3.
    for (size_t i = 0; i < got.size(); ++i) {
        const float diff = std::fabs(got[i] - expected[i]);
        const float rel  = diff / std::max(std::fabs(expected[i]), 1.f);
        if (rel > 5e-3f) {
            std::cerr << "Mismatch at " << i << ": got " << got[i]
                      << ", expected " << expected[i] << " (rel " << rel << ")\n";
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    int m = 128, n = 128, k = 128;
    if (argc >= 4) {
        m = std::atoi(argv[1]);
        n = std::atoi(argv[2]);
        k = std::atoi(argv[3]);
    }
    std::vector<float> host_a(m * k), host_b(k * n), host_c(m * n, 0.f), reference(m * n, 0.f);
    fill_inputs(host_a, host_b, m, n, k);
    matmul_cpu(host_a, host_b, reference, m, n, k);

    float *device_a = nullptr, *device_b = nullptr, *device_c = nullptr;
    CHECK_CUDA(cudaMalloc(&device_a, host_a.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_b, host_b.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_c, host_c.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), host_a.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), host_b.size() * sizeof(float), cudaMemcpyHostToDevice));

    launch(device_a, device_b, device_c, m, n, k);

    CHECK_CUDA(cudaMemcpy(host_c.data(), device_c, host_c.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(device_a));
    CHECK_CUDA(cudaFree(device_b));
    CHECK_CUDA(cudaFree(device_c));

    if (!check_output(host_c, reference)) return EXIT_FAILURE;
    std::cout << "matmul [cublas] practice PASS  m=" << m << " n=" << n << " k=" << k << '\n';
    return EXIT_SUCCESS;
}
