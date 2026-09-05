// version: cuBLAS baseline (K0)
//
// This is the "how fast is fast" reference: a call to the vendor
// library. cuBLAS on A100 dispatches to CUTLASS-based FP32 GEMM
// kernels; with `CUBLAS_COMPUTE_32F_FAST_TF32` it internally rounds
// the FP32 inputs to TF32 (10 mantissa bits) and uses tensor cores.
//
// diff vs any hand-written kernel below:
// - not a single kernel — cuBLAS heuristics pick a persistent /
//   split-K / stream-K kernel depending on shape
// - all "K1..K8" tricks are already in there (tiling, register
//   blocking, vectorized loads, MMA); measuring against K0 tells you
//   how much production perf leaves on the table
//
// Note on layout: cuBLAS is column-major. We have row-major A (m×k),
// B (k×n), C (m×n). Use the identity  (A B)^T = B^T A^T. In
// column-major, C^T (n×m) equals (B viewed as n×k row-major, which is
// column-major k×n) times (A viewed as k×m row-major, column-major
// m×k). Equivalent: pass B first, A second, and swap m/n:
//   cublasSgemm(handle, N, N, n, m, k, alpha, B, n, A, k, beta, C, n)

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cublas_v2.h>

#include "../common/cuda_utils.cuh"

namespace {

#define CHECK_CUBLAS(x) do {                                          \
    cublasStatus_t s = (x);                                           \
    if (s != CUBLAS_STATUS_SUCCESS) {                                 \
        std::cerr << "cuBLAS error " << int(s) << " at "              \
                  << __FILE__ << ":" << __LINE__ << '\n';             \
        std::abort();                                                 \
    }                                                                 \
} while (0)

void launch(const float* a, const float* b, float* c, int m, int n, int k) {
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        CHECK_CUBLAS(cublasCreate(&handle));
        // Allow TF32 tensor cores (default on Ampere+ for compute32f but
        // being explicit makes the intent obvious).
        CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    }

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // Row-major A(m×k) * B(k×n) = C(m×n) is equivalent to computing
    //   C^T (n×m column-major) = B^T (n×k) * A^T (k×m)
    // and B (k×n row-major) viewed as column-major is already n×k^T,
    // so no transpose flag is needed.  Just swap A↔B and m↔n.
    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             n, m, k,
                             &alpha,
                             b, n,   // leading dim of row-major B
                             a, k,   // leading dim of row-major A
                             &beta,
                             c, n));
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
}

// ---- host boilerplate: identical across every version in this folder ----

void fill_inputs(std::vector<float>& a, std::vector<float>& b, int m, int n, int k) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < k; ++col) {
            a[row * k + col] = static_cast<float>((row + col) % 7);
        }
    }
    for (int row = 0; row < k; ++row) {
        for (int col = 0; col < n; ++col) {
            b[row * n + col] = static_cast<float>((row * 2 + col) % 5);
        }
    }
}

void matmul_cpu(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c,
                int m, int n, int k) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float acc = 0.0f;
            for (int inner = 0; inner < k; ++inner) {
                acc += a[row * k + inner] * b[inner * n + col];
            }
            c[row * n + col] = acc;
        }
    }
}

bool check_output(const std::vector<float>& got, const std::vector<float>& expected) {
    // TF32 has 10 mantissa bits, so relative error ~1e-3 is expected.
    for (size_t i = 0; i < got.size(); ++i) {
        const float diff = std::fabs(got[i] - expected[i]);
        const float rel  = diff / std::max(std::fabs(expected[i]), 1.0f);
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

    std::vector<float> host_a(m * k);
    std::vector<float> host_b(k * n);
    std::vector<float> host_c(m * n, 0.0f);
    std::vector<float> reference(m * n, 0.0f);
    fill_inputs(host_a, host_b, m, n, k);
    matmul_cpu(host_a, host_b, reference, m, n, k);

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;
    CHECK_CUDA(cudaMalloc(&device_a, host_a.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_b, host_b.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_c, host_c.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), host_a.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), host_b.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    launch(device_a, device_b, device_c, m, n, k);

    CHECK_CUDA(cudaMemcpy(host_c.data(), device_c, host_c.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(device_a));
    CHECK_CUDA(cudaFree(device_b));
    CHECK_CUDA(cudaFree(device_c));

    if (!check_output(host_c, reference)) {
        return EXIT_FAILURE;
    }
    std::cout << "matmul [cublas] PASS  m=" << m << " n=" << n << " k=" << k << '\n';
    return EXIT_SUCCESS;
}
