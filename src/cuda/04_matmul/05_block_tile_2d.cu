// version: 2D block tiling (K5 in the ladder)
//
// diff vs block_tile_1d (K4):
// - each thread computes a kThreadTileM × kThreadTileN = 2 × 2
//   register block of outputs (2D tile), not just an M×1 column
// - block is 16 × 16 threads covering a 32 × 32 output tile
// - A tile is 32 × 16 (each thread loads 2 rows); B tile is 16 × 32
//
// Point: the 1D tile in K4 lifted `a_frag` reuse into registers;
// making the thread tile 2D also lifts `b_frag` reuse. Now every
// (a_frag, b_frag) pair loaded from SMEM is used four times
// (once per acc[i][j]) — 2× the arithmetic intensity of K4 for the
// same SMEM traffic.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../common/cuda_utils.cuh"

namespace {

constexpr int kThreadsX = 16;
constexpr int kThreadsY = 16;
constexpr int kThreadTileM = 2;
constexpr int kThreadTileN = 2;
constexpr int kBlockTileM = kThreadsY * kThreadTileM;  // 32
constexpr int kBlockTileN = kThreadsX * kThreadTileN;  // 32
constexpr int kBlockTileK = 16;

__global__ void matmul_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float a_tile[kBlockTileM][kBlockTileK];
    __shared__ float b_tile[kBlockTileK][kBlockTileN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row_base = blockIdx.y * kBlockTileM + ty * kThreadTileM;
    const int col_base = blockIdx.x * kBlockTileN + tx * kThreadTileN;

    float acc[kThreadTileM][kThreadTileN] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

    for (int tile_k = 0; tile_k < k; tile_k += kBlockTileK) {
        // A tile: each thread loads 2 rows (ty, ty + kThreadsY) at column tx.
        const int a_col = tile_k + tx;
        const int a_row0 = blockIdx.y * kBlockTileM + ty;
        const int a_row1 = a_row0 + kThreadsY;

        a_tile[ty][tx] = (a_row0 < m && a_col < k) ? a[a_row0 * k + a_col] : 0.0f;
        a_tile[ty + kThreadsY][tx] =
            (a_row1 < m && a_col < k) ? a[a_row1 * k + a_col] : 0.0f;

        // B tile: each thread loads 2 cols (tx, tx + kThreadsX) at row ty.
        const int b_row = tile_k + ty;
        const int b_col0 = blockIdx.x * kBlockTileN + tx;
        const int b_col1 = b_col0 + kThreadsX;

        b_tile[ty][tx] = (b_row < k && b_col0 < n) ? b[b_row * n + b_col0] : 0.0f;
        b_tile[ty][tx + kThreadsX] =
            (b_row < k && b_col1 < n) ? b[b_row * n + b_col1] : 0.0f;

        __syncthreads();

#pragma unroll
        for (int inner = 0; inner < kBlockTileK; ++inner) {
            const float a_frag0 = a_tile[ty * kThreadTileM + 0][inner];
            const float a_frag1 = a_tile[ty * kThreadTileM + 1][inner];
            const float b_frag0 = b_tile[inner][tx * kThreadTileN + 0];
            const float b_frag1 = b_tile[inner][tx * kThreadTileN + 1];
            acc[0][0] += a_frag0 * b_frag0;
            acc[0][1] += a_frag0 * b_frag1;
            acc[1][0] += a_frag1 * b_frag0;
            acc[1][1] += a_frag1 * b_frag1;
        }
        __syncthreads();
    }

    for (int i = 0; i < kThreadTileM; ++i) {
        for (int j = 0; j < kThreadTileN; ++j) {
            const int row = row_base + i;
            const int col = col_base + j;
            if (row < m && col < n) {
                c[row * n + col] = acc[i][j];
            }
        }
    }
}

void launch(const float* a, const float* b, float* c, int m, int n, int k) {
    dim3 block(kThreadsX, kThreadsY);
    dim3 grid(cuda_utils::ceil_div(n, kBlockTileN), cuda_utils::ceil_div(m, kBlockTileM));
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
    std::cout << "matmul [block_tile_2d] PASS  m=" << m << " n=" << n << " k=" << k << '\n';
    return EXIT_SUCCESS;
}
