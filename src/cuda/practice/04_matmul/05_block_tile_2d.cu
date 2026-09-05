// practice: matmul version = block_tile_2d (K5)
//
// Each thread now computes a kThreadTileM × kThreadTileN = 2 × 2
// register block of C. Block: 16 × 16 threads → 32 × 32 output.
// Every (a_frag, b_frag) loaded from SMEM is used 4× (once per acc[i][j]).

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../../common/cuda_utils.cuh"

namespace {

// TODO: each thread computes a 2x2 register block of outputs.
// - block 16 x 16 covers 32 x 32 output
// - A tile 32 x 16 (each thread loads 2 rows), B tile 16 x 32 (each thread loads 2 cols)
// - inner loop: read 2 A frags, 2 B frags, do 4 FMAs into acc[2][2]
__global__ void matmul_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    (void)a; (void)b; (void)c; (void)m; (void)n; (void)k;
}

// TODO: block = 16 x 16; grid = ceil(n/32) x ceil(m/32).
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
    CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), host_a.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), host_b.size() * sizeof(float), cudaMemcpyHostToDevice));

    launch(device_a, device_b, device_c, m, n, k);

    CHECK_CUDA(cudaMemcpy(host_c.data(), device_c, host_c.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(device_a));
    CHECK_CUDA(cudaFree(device_b));
    CHECK_CUDA(cudaFree(device_c));

    if (!check_output(host_c, reference)) return EXIT_FAILURE;
    std::cout << "matmul [block_tile_2d] practice PASS  m=" << m << " n=" << n << " k=" << k << '\n';
    return EXIT_SUCCESS;
}
