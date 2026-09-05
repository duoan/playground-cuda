// practice: matmul version = wmma (K8)
//
// Same math as K7 but using the higher-level nvcuda::wmma fragment
// API. Fragment shape 16×16×16 is the only FP16→FP32 shape supported
// on Ampere.
//
// The API hides:
//   - the ldmatrix instructions used to load A/B into per-lane regs
//   - the mma.sync instruction that actually runs on tensor cores
//   - the store layout for the accumulator
// The compiler lowers wmma:: calls to exactly the PTX we wrote by hand
// in K7.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_fp16.h>
#include <mma.h>

#include "../../common/cuda_utils.cuh"

namespace {

using half_t = __half;
using namespace nvcuda;

constexpr int kWarpsM = 2, kWarpsN = 2;
constexpr int kThreadsPerBlock = kWarpsM * kWarpsN * 32;   // 128
constexpr int kWmmaM = 16, kWmmaN = 16, kWmmaK = 16;
constexpr int kBlockTileM = kWarpsM * kWmmaM;   // 32
constexpr int kBlockTileN = kWarpsN * kWmmaN;   // 32
constexpr int kBlockTileK = kWmmaK;             // 16

// TODO:
// - declare wmma::fragment<accumulator, 16,16,16, float> c_frag; fill_fragment(c_frag, 0)
// - cooperative-load A(32×16) and B(16×32) halfs to SMEM per K tile
// - wmma::load_matrix_sync into matrix_a / matrix_b fragments (row_major)
// - wmma::mma_sync(c_frag, a_frag, b_frag, c_frag)
// - wmma::store_matrix_sync back to C with mem_row_major, ldc = n
__global__ void matmul_kernel(const half_t* a, const half_t* b, float* c, int m, int n, int k) {
    (void)a; (void)b; (void)c; (void)m; (void)n; (void)k;
}

void launch(const half_t* a, const half_t* b, float* c, int m, int n, int k) {
    (void)a; (void)b; (void)c; (void)m; (void)n; (void)k;
}

// ---- host boilerplate (FP16 track) ----

void fill_inputs(std::vector<half_t>& a, std::vector<half_t>& b, int m, int n, int k) {
    for (int row = 0; row < m; ++row)
        for (int col = 0; col < k; ++col)
            a[row * k + col] = __float2half(static_cast<float>((row + col) % 7));
    for (int row = 0; row < k; ++row)
        for (int col = 0; col < n; ++col)
            b[row * n + col] = __float2half(static_cast<float>((row * 2 + col) % 5));
}

void matmul_cpu(const std::vector<half_t>& a, const std::vector<half_t>& b, std::vector<float>& c,
                int m, int n, int k) {
    for (int row = 0; row < m; ++row)
        for (int col = 0; col < n; ++col) {
            float acc = 0.f;
            for (int inner = 0; inner < k; ++inner)
                acc += __half2float(a[row * k + inner]) * __half2float(b[inner * n + col]);
            c[row * n + col] = acc;
        }
}

bool check_output(const std::vector<float>& got, const std::vector<float>& expected) {
    for (size_t i = 0; i < got.size(); ++i) {
        const float diff = std::fabs(got[i] - expected[i]);
        const float rel  = diff / std::max(std::fabs(expected[i]), 1.f);
        if (rel > 1e-2f) {
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
    std::vector<half_t> host_a(m * k), host_b(k * n);
    std::vector<float>  host_c(m * n, 0.f), reference(m * n, 0.f);
    fill_inputs(host_a, host_b, m, n, k);
    matmul_cpu(host_a, host_b, reference, m, n, k);

    half_t *device_a = nullptr, *device_b = nullptr;
    float  *device_c = nullptr;
    CHECK_CUDA(cudaMalloc(&device_a, host_a.size() * sizeof(half_t)));
    CHECK_CUDA(cudaMalloc(&device_b, host_b.size() * sizeof(half_t)));
    CHECK_CUDA(cudaMalloc(&device_c, host_c.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), host_a.size() * sizeof(half_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), host_b.size() * sizeof(half_t), cudaMemcpyHostToDevice));

    launch(device_a, device_b, device_c, m, n, k);

    CHECK_CUDA(cudaMemcpy(host_c.data(), device_c, host_c.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(device_a));
    CHECK_CUDA(cudaFree(device_b));
    CHECK_CUDA(cudaFree(device_c));

    if (!check_output(host_c, reference)) return EXIT_FAILURE;
    std::cout << "matmul [wmma] practice PASS  m=" << m << " n=" << n << " k=" << k << '\n';
    return EXIT_SUCCESS;
}
