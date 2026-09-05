// practice: matmul version = block_tile_1d (K4)
//
// Each thread now owns kThreadTileM = 8 output rows in a single
// column — an 8×1 register column. Block: 64×8 threads → 64×64 output.
// Inside the K loop, load `b_frag` once from SMEM and reuse it
// against 8 different a_frags → lifts B reuse into registers.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../../common/cuda_utils.cuh"

namespace {

constexpr int kBlockTileM = 64;
constexpr int kBlockTileN = 64;
constexpr int kBlockTileK = 8;
constexpr int kThreadTileM = 8;
constexpr int kThreadsX = kBlockTileN;
constexpr int kThreadsY = kBlockTileM / kThreadTileM;
constexpr int kThreadsPerBlock = kThreadsX * kThreadsY;

// TODO: implement block_tile_1d kernel.
// - each thread owns rows [ty*8 .. ty*8+7] × col tx of the C block
// - flat tid = ty*kThreadsX + tx; 512 threads load 64×8=512 halfs of A,
//   and separately 8×64=512 halfs of B (one float each)
// - inner K loop: cache b_tile[inner][tx] once, reuse across 8 a_frags
__global__ void matmul_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    (void)a; (void)b; (void)c; (void)m; (void)n; (void)k;
}

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
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::fabs(got[i] - expected[i]) > 1e-4f) {
            std::cerr << "Mismatch at " << i << ": got " << got[i]
                      << ", expected " << expected[i] << '\n';
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
    std::cout << "matmul [block_tile_1d] practice PASS  m=" << m << " n=" << n << " k=" << k << '\n';
    return EXIT_SUCCESS;
}
