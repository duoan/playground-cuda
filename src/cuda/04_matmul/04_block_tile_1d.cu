// version: 1D block tiling (K4)
//
// diff vs smem_tiled (K3):
// - each thread now computes kThreadTileM = 8 output rows in one
//   column (an M×1 register column), instead of a single scalar
// - block is kThreadsX × kThreadsY = 64 × 8 threads covering a
//   kBlockTileM × kBlockTileN = 64 × 64 output tile
// - inside the K inner loop, each thread caches its `b_frag` (one
//   scalar) once and reuses it against 8 different a_frags
//
// Point: K3's SMEM tiling gave every value O(kTile) reuse; K4 lifts
// that reuse into *registers*. `b_frag` is loaded from SMEM once and
// then used 8 times (once per acc[i]) before being overwritten,
// cutting SMEM traffic on B by 8× and letting FMAs pipeline out of
// register file — the fastest possible operand path on Ampere.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../common/cuda_utils.cuh"

namespace {

// One block: kBlockTileM = 64 rows × kBlockTileN = 64 cols of C, computed
// with kThreadsX × kThreadsY = 64 × 8 = 512 threads. Each thread owns
// an 8×1 column of outputs (kThreadTileM = 8 rows, 1 col).
constexpr int kBlockTileM = 64;
constexpr int kBlockTileN = 64;
constexpr int kBlockTileK = 8;
constexpr int kThreadTileM = 8;

constexpr int kThreadsX = kBlockTileN;                     // 64 threads along N
constexpr int kThreadsY = kBlockTileM / kThreadTileM;      // 8 threads along M
constexpr int kThreadsPerBlock = kThreadsX * kThreadsY;    // 512

__global__ void matmul_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float a_tile[kBlockTileM][kBlockTileK];
    __shared__ float b_tile[kBlockTileK][kBlockTileN];

    const int tx = threadIdx.x;                            // 0..63 in N direction
    const int ty = threadIdx.y;                            // 0..7  in M direction
    const int tid = ty * kThreadsX + tx;                   // flat 0..511

    // Each thread owns rows [ty*8 .. ty*8+7] × col tx in the C tile.
    const int row_base = blockIdx.y * kBlockTileM + ty * kThreadTileM;
    const int col      = blockIdx.x * kBlockTileN + tx;

    float acc[kThreadTileM] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int tile_k = 0; tile_k < k; tile_k += kBlockTileK) {
        // Load kBlockTileM × kBlockTileK = 64 × 8 = 512 floats of A: one per thread.
        {
            const int ar = tid / kBlockTileK;
            const int ac = tid % kBlockTileK;
            const int g_row = blockIdx.y * kBlockTileM + ar;
            const int g_col = tile_k + ac;
            a_tile[ar][ac] = (g_row < m && g_col < k) ? a[g_row * k + g_col] : 0.f;
        }
        // Load kBlockTileK × kBlockTileN = 8 × 64 = 512 floats of B: one per thread.
        {
            const int br = tid / kBlockTileN;
            const int bc = tid % kBlockTileN;
            const int g_row = tile_k + br;
            const int g_col = blockIdx.x * kBlockTileN + bc;
            b_tile[br][bc] = (g_row < k && g_col < n) ? b[g_row * n + g_col] : 0.f;
        }
        __syncthreads();

#pragma unroll
        for (int inner = 0; inner < kBlockTileK; ++inner) {
            // Load ONE b_frag from SMEM, reuse it against 8 a_frags.
            const float b_frag = b_tile[inner][tx];
#pragma unroll
            for (int i = 0; i < kThreadTileM; ++i) {
                const float a_frag = a_tile[ty * kThreadTileM + i][inner];
                acc[i] += a_frag * b_frag;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < kThreadTileM; ++i) {
        const int row = row_base + i;
        if (row < m && col < n) {
            c[row * n + col] = acc[i];
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
    std::cout << "matmul [block_tile_1d] PASS  m=" << m << " n=" << n << " k=" << k << '\n';
    return EXIT_SUCCESS;
}
