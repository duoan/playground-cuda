// version: WMMA (K8) — nvcuda::wmma API, FP16 → FP32 accumulate
//
// diff vs vectorized (K6):
// - inputs are now half (FP16); accumulator stays float (FP32)
// - the inner K loop is replaced by `wmma::mma_sync` on 16×16×16
//   fragments, which the compiler lowers to a single HMMA.16816
//   instruction executed by a whole warp on tensor cores
// - one warp owns one 16×16 output tile of C; a 128-thread (4 warps)
//   block owns a 32×32 output tile (2 warps × 2 warps grid)
//
// Point: HMMA is the Ampere tensor-core instruction. One HMMA does
// 16 × 16 × 16 = 4 096 half-precision multiply-adds in a handful of
// cycles — several dozen times more FMA throughput than the CUDA-core
// FFMA path K6 uses. Everything before this version is fighting for
// %-of-CUDA-core-peak; this version fights for %-of-tensor-core-peak
// (about 8× higher on A100).
//
// The kernel keeps a shared-memory staging buffer for A/B so the
// global loads are still coalesced, but the actual math is now
// register↔TC↔register. No SMEM traffic per HMMA.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_fp16.h>
#include <mma.h>

#include "../common/cuda_utils.cuh"

namespace {

using half_t = __half;
using namespace nvcuda;

// A 128-thread block = 4 warps. Arrange them 2×2 in the (M, N) plane.
constexpr int kWarpsM = 2;
constexpr int kWarpsN = 2;
constexpr int kWarpSize = 32;
constexpr int kThreadsPerBlock = kWarpsM * kWarpsN * kWarpSize;   // 128

// WMMA fragment shape supported on Ampere for FP16 inputs, FP32 acc.
constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;

// Block-tile dimensions match the fragment layout: 2 warps × 16 each.
constexpr int kBlockTileM = kWarpsM * kWmmaM;   // 32
constexpr int kBlockTileN = kWarpsN * kWmmaN;   // 32
constexpr int kBlockTileK = kWmmaK;             // 16

__global__ void matmul_kernel(const half_t* a, const half_t* b, float* c, int m, int n, int k) {
    __shared__ half_t a_tile[kBlockTileM][kBlockTileK];
    __shared__ half_t b_tile[kBlockTileK][kBlockTileN];

    const int warp_id = threadIdx.x / kWarpSize;
    const int warp_m = warp_id / kWarpsN;               // 0..kWarpsM-1
    const int warp_n = warp_id % kWarpsN;               // 0..kWarpsN-1

    // Row/col of the C tile owned by this warp (in block-local coords).
    const int row_base = blockIdx.y * kBlockTileM + warp_m * kWmmaM;
    const int col_base = blockIdx.x * kBlockTileN + warp_n * kWmmaN;

    wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int tile_k = 0; tile_k < k; tile_k += kBlockTileK) {
        // Cooperative load of a 32×16 A tile and 16×32 B tile into SMEM.
        // 128 threads × 4 halfs each = 512 halfs, matches a_tile (32×16)
        // and separately b_tile (16×32) sizes when we split the block.
        {
            // A tile: 32×16 = 512 halfs; each thread loads 4 halfs.
            constexpr int kElemsA = kBlockTileM * kBlockTileK;   // 512
            constexpr int kPerThread = kElemsA / kThreadsPerBlock;  // 4
#pragma unroll
            for (int i = 0; i < kPerThread; ++i) {
                const int idx = i * kThreadsPerBlock + threadIdx.x;
                const int r = idx / kBlockTileK;
                const int c_ = idx % kBlockTileK;
                const int g_row = blockIdx.y * kBlockTileM + r;
                const int g_col = tile_k + c_;
                a_tile[r][c_] = (g_row < m && g_col < k) ? a[g_row * k + g_col] : half_t(0);
            }
        }
        {
            // B tile: 16×32 = 512 halfs; each thread loads 4 halfs.
            constexpr int kElemsB = kBlockTileK * kBlockTileN;   // 512
            constexpr int kPerThread = kElemsB / kThreadsPerBlock;  // 4
#pragma unroll
            for (int i = 0; i < kPerThread; ++i) {
                const int idx = i * kThreadsPerBlock + threadIdx.x;
                const int r = idx / kBlockTileN;
                const int c_ = idx % kBlockTileN;
                const int g_row = tile_k + r;
                const int g_col = blockIdx.x * kBlockTileN + c_;
                b_tile[r][c_] = (g_row < k && g_col < n) ? b[g_row * n + g_col] : half_t(0);
            }
        }
        __syncthreads();

        // Load A/B fragments from SMEM into registers, run one HMMA.
        wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half_t, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half_t, wmma::row_major> b_frag;

        // SMEM row stride = kBlockTileK for A, kBlockTileN for B.
        wmma::load_matrix_sync(a_frag, &a_tile[warp_m * kWmmaM][0], kBlockTileK);
        wmma::load_matrix_sync(b_frag, &b_tile[0][warp_n * kWmmaN], kBlockTileN);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    // Store the 16×16 accumulator directly to global C.
    if (row_base < m && col_base < n) {
        wmma::store_matrix_sync(&c[row_base * n + col_base], c_frag, n,
                                wmma::mem_row_major);
    }
}

void launch(const half_t* a, const half_t* b, float* c, int m, int n, int k) {
    if ((m % kBlockTileM) || (n % kBlockTileN) || (k % kBlockTileK)) {
        std::cerr << "wmma kernel requires m,n multiples of 32, k multiple of 16 "
                  << "(got m=" << m << " n=" << n << " k=" << k << ")\n";
        std::abort();
    }
    dim3 block(kThreadsPerBlock);
    dim3 grid(n / kBlockTileN, m / kBlockTileM);
    matmul_kernel<<<grid, block>>>(a, b, c, m, n, k);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
}

// ---- host boilerplate (FP16 track) ----

void fill_inputs(std::vector<half_t>& a, std::vector<half_t>& b, int m, int n, int k) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < k; ++col) {
            a[row * k + col] = __float2half(static_cast<float>((row + col) % 7));
        }
    }
    for (int row = 0; row < k; ++row) {
        for (int col = 0; col < n; ++col) {
            b[row * n + col] = __float2half(static_cast<float>((row * 2 + col) % 5));
        }
    }
}

void matmul_cpu(const std::vector<half_t>& a, const std::vector<half_t>& b, std::vector<float>& c,
                int m, int n, int k) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float acc = 0.0f;
            for (int inner = 0; inner < k; ++inner) {
                acc += __half2float(a[row * k + inner]) * __half2float(b[inner * n + col]);
            }
            c[row * n + col] = acc;
        }
    }
}

bool check_output(const std::vector<float>& got, const std::vector<float>& expected) {
    // FP16 has 10 mantissa bits; allow ~1e-2 relative error.
    for (size_t i = 0; i < got.size(); ++i) {
        const float diff = std::fabs(got[i] - expected[i]);
        const float rel  = diff / std::max(std::fabs(expected[i]), 1.0f);
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

    std::vector<half_t> host_a(m * k);
    std::vector<half_t> host_b(k * n);
    std::vector<float>  host_c(m * n, 0.0f);
    std::vector<float>  reference(m * n, 0.0f);
    fill_inputs(host_a, host_b, m, n, k);
    matmul_cpu(host_a, host_b, reference, m, n, k);

    half_t* device_a = nullptr;
    half_t* device_b = nullptr;
    float*  device_c = nullptr;
    CHECK_CUDA(cudaMalloc(&device_a, host_a.size() * sizeof(half_t)));
    CHECK_CUDA(cudaMalloc(&device_b, host_b.size() * sizeof(half_t)));
    CHECK_CUDA(cudaMalloc(&device_c, host_c.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), host_a.size() * sizeof(half_t),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), host_b.size() * sizeof(half_t),
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
    std::cout << "matmul [wmma] PASS  m=" << m << " n=" << n << " k=" << k << '\n';
    return EXIT_SUCCESS;
}
