// version: shared-memory tiled (K3 in the ladder)
//
// diff vs naive_coalesced (K2):
// - each block cooperatively stages a kTile × kTile piece of A and B
//   in shared memory before every thread walks its dot product
// - block sweeps K in kTile-sized steps
//
// Point: coalescing (K2) fixes the *transaction* problem; SMEM tiling
// fixes the *reuse* problem. Now every value loaded from HBM is used
// kTile times inside the block, so effective HBM bandwidth drops from
// O(N^3) to O(N^3 / kTile).

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../common/cuda_utils.cuh"

namespace {

constexpr int kTile = 16;

// tiled kernel: block owns a kTile × kTile output tile.
__global__ void matmul_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float a_tile[kTile][kTile];
    __shared__ float b_tile[kTile][kTile];

    const int row = blockIdx.y * kTile + threadIdx.y;
    const int col = blockIdx.x * kTile + threadIdx.x;
    float acc = 0.0f;

    for (int tile_k = 0; tile_k < k; tile_k += kTile) {
        const int a_col = tile_k + threadIdx.x;
        const int b_row = tile_k + threadIdx.y;
        a_tile[threadIdx.y][threadIdx.x] =
            (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
        b_tile[threadIdx.y][threadIdx.x] =
            (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int inner = 0; inner < kTile; ++inner) {
            acc += a_tile[threadIdx.y][inner] * b_tile[inner][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = acc;
    }
}

void launch(const float* a, const float* b, float* c, int m, int n, int k) {
    dim3 block(kTile, kTile);
    dim3 grid(cuda_utils::ceil_div(n, kTile), cuda_utils::ceil_div(m, kTile));
    matmul_kernel<<<grid, block>>>(a, b, c, m, n, k);
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
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::fabs(got[i] - expected[i]) > 1e-4f) {
            std::cerr << "Mismatch at " << i << ": got " << got[i] << ", expected " << expected[i]
                      << '\n';
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
    std::cout << "matmul [smem_tiled] PASS  m=" << m << " n=" << n << " k=" << k << '\n';
    return EXIT_SUCCESS;
}
