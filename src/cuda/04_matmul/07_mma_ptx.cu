// version: raw MMA PTX (K7) — inline `mma.sync.aligned.m16n8k16`
//
// diff vs vectorized (K6):
// - inputs are FP16, accumulator is FP32
// - a whole *warp* (32 threads) cooperates on one HMMA instruction
//   which computes a 16 × 8 output tile in one shot (M=16, N=8, K=16)
// - each warp owns kMmaM × kMmaN = 16 × 8 of C
// - registers are loaded with `ldmatrix.sync.aligned.x4.m8n8` from
//   SMEM, which is the "correct" way to get halves into the exact
//   register layout mma.sync expects (no by-hand permutation)
//
// This is the version *underneath* WMMA (K8). The nvcuda::wmma API
// eventually lowers to exactly these ldmatrix / mma.sync PTX. Writing
// it by hand shows what a "tensor-core-native" kernel looks like at
// the lowest level.
//
// Grid layout: one warp per (16×8) C tile. A block has kWarpsM ×
// kWarpsN = 2 × 2 = 4 warps, so a block owns 32 × 16 of C.

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <vector>

#include <cuda_fp16.h>

#include "../common/cuda_utils.cuh"

namespace {

using half_t = __half;

constexpr int kWarpSize   = 32;
constexpr int kWarpsM     = 2;
constexpr int kWarpsN     = 2;
constexpr int kWarpsPerBlock = kWarpsM * kWarpsN;
constexpr int kThreadsPerBlock = kWarpsPerBlock * kWarpSize;   // 128

// Fragment shape supported by mma.sync.aligned.m16n8k16 on sm_80.
constexpr int kMmaM = 16;
constexpr int kMmaN = 8;
constexpr int kMmaK = 16;

constexpr int kBlockTileM = kWarpsM * kMmaM;    // 32
constexpr int kBlockTileN = kWarpsN * kMmaN;    // 16
constexpr int kBlockTileK = kMmaK;              // 16

// Helper: cast a shared-memory pointer to a 32-bit generic address for PTX.
__device__ __forceinline__ uint32_t smem_ptr_u32(const void* p) {
    uint32_t out;
    asm volatile("{ .reg .u64  ptr; cvta.to.shared.u64 ptr, %1; cvt.u32.u64 %0, ptr; }"
                 : "=r"(out) : "l"(p));
    return out;
}

// ldmatrix.sync.aligned.x4.m8n8: 32 threads collectively load four
// 8×8 half tiles from SMEM into four .b32 registers (each register is
// 2 halves). Used to get A in the exact per-lane layout mma.sync wants.
__device__ __forceinline__
void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t smem_addr) {
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                 "{%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_addr));
}

// ldmatrix.sync.aligned.x2.m8n8.trans: loads two 8×8 half tiles with
// a transpose — this is exactly what mma.sync's B operand expects
// when B is stored row-major in SMEM.
__device__ __forceinline__
void ldmatrix_x2_trans(uint32_t& r0, uint32_t& r1, uint32_t smem_addr) {
    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 "
                 "{%0, %1}, [%2];\n"
                 : "=r"(r0), "=r"(r1) : "r"(smem_addr));
}

// mma.sync.aligned.m16n8k16: D = A * B + C  in FP32, A/B are FP16.
__device__ __forceinline__
void mma_m16n8k16(float&  d0, float&  d1, float&  d2, float&  d3,
                  uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                  uint32_t b0, uint32_t b1,
                  float   c0, float   c1, float   c2, float   c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__global__ void matmul_kernel(const half_t* a, const half_t* b, float* c, int m, int n, int k) {
    __shared__ half_t a_tile[kBlockTileM][kBlockTileK];
    __shared__ half_t b_tile[kBlockTileK][kBlockTileN];

    const int lane    = threadIdx.x % kWarpSize;
    const int warp_id = threadIdx.x / kWarpSize;
    const int warp_m  = warp_id / kWarpsN;                // 0..kWarpsM-1
    const int warp_n  = warp_id % kWarpsN;                // 0..kWarpsN-1

    // C output owned by this warp.
    const int row_base = blockIdx.y * kBlockTileM + warp_m * kMmaM;
    const int col_base = blockIdx.x * kBlockTileN + warp_n * kMmaN;

    // Accumulator: 4 floats per lane for a 16×8 warp tile.
    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    for (int tile_k = 0; tile_k < k; tile_k += kBlockTileK) {
        // Cooperative SMEM staging. Block: 128 threads.
        // A tile = 32×16 = 512 halfs → 4 halfs / thread.
        {
            constexpr int kElems = kBlockTileM * kBlockTileK;
            constexpr int kPerThread = kElems / kThreadsPerBlock;    // 4
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
        // B tile = 16×16 = 256 halfs → 2 halfs / thread.
        {
            constexpr int kElems = kBlockTileK * kBlockTileN;
            constexpr int kPerThread = kElems / kThreadsPerBlock;    // 2
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

        // ---- load A fragment (16×16 halfs) with ldmatrix.x4 ----
        // ldmatrix expects a "per-row starting address" from each of the
        // 32 lanes. For an M×K=16×16 tile, that's:
        //   lane i in [0,8):   row i,      col 0
        //   lane i in [8,16):  row i-8,    col 8
        //   lane i in [16,24): row i,      col 0    ← same 8 rows again (x4 = 4 tiles: 2 M × 2 K)
        //   lane i in [24,32): row i-8,    col 8
        // ldmatrix returns 4 .b32 registers per lane; each register is
        // 2 halfs from that lane's row segment.
        uint32_t a_reg[4];
        {
            const int row_in_tile = warp_m * kMmaM + (lane % 16);
            const int col_in_tile = (lane / 16) * 8;
            const uint32_t smem_addr = smem_ptr_u32(&a_tile[row_in_tile][col_in_tile]);
            ldmatrix_x4(a_reg[0], a_reg[1], a_reg[2], a_reg[3], smem_addr);
        }

        // ---- load B fragment (16×8 halfs) with ldmatrix.x2.trans ----
        // mma.sync expects B in "col-major" register layout, so we ask
        // ldmatrix to transpose the 8×8 tile on load.
        uint32_t b_reg[2];
        {
            const int row_in_tile = (lane % 8);                       // 0..7
            const int col_in_tile = warp_n * kMmaN + (lane / 8) * 0;  // starts at warp_n*8
            // For x2.trans we need starting addresses for two 8×8 sub-tiles
            // (one for K=[0,8), one for K=[8,16)). ldmatrix.x2 with a single
            // starting row address does this: it treats lanes 0..7 as tile
            // 0, lanes 8..15 as tile 1, lanes 16..31 unused (only x2).
            const int r = (lane < 16) ? ((lane % 8) + (lane / 8) * 8) : 0;
            (void)row_in_tile; (void)col_in_tile;
            const uint32_t smem_addr = smem_ptr_u32(&b_tile[r][warp_n * kMmaN]);
            ldmatrix_x2_trans(b_reg[0], b_reg[1], smem_addr);
        }

        // ---- one HMMA ----
        mma_m16n8k16(acc[0], acc[1], acc[2], acc[3],
                     a_reg[0], a_reg[1], a_reg[2], a_reg[3],
                     b_reg[0], b_reg[1],
                     acc[0], acc[1], acc[2], acc[3]);

        __syncthreads();
    }

    // ---- store 16×8 acc back to C ----
    // For mma.m16n8k16 f32 accumulator, per-lane output layout is:
    //   group        = laneid / 4        (0..7 → row group)
    //   threadInGroup = laneid % 4       (0..3 → col group)
    // Elements per lane:
    //   d0 : row = group          col = threadInGroup*2 + 0
    //   d1 : row = group          col = threadInGroup*2 + 1
    //   d2 : row = group + 8      col = threadInGroup*2 + 0
    //   d3 : row = group + 8      col = threadInGroup*2 + 1
    const int group          = lane / 4;
    const int threadInGroup  = lane % 4;

    const int r0 = row_base + group;
    const int r1 = row_base + group + 8;
    const int c0 = col_base + threadInGroup * 2 + 0;
    const int c1 = col_base + threadInGroup * 2 + 1;

    if (r0 < m && c0 < n) c[r0 * n + c0] = acc[0];
    if (r0 < m && c1 < n) c[r0 * n + c1] = acc[1];
    if (r1 < m && c0 < n) c[r1 * n + c0] = acc[2];
    if (r1 < m && c1 < n) c[r1 * n + c1] = acc[3];
}

void launch(const half_t* a, const half_t* b, float* c, int m, int n, int k) {
    if ((m % kBlockTileM) || (n % kBlockTileN) || (k % kBlockTileK)) {
        std::cerr << "mma_ptx kernel requires m %" << kBlockTileM
                  << ", n %" << kBlockTileN << ", k %" << kBlockTileK << " == 0"
                  << " (got m=" << m << " n=" << n << " k=" << k << ")\n";
        std::abort();
    }
    dim3 block(kThreadsPerBlock);
    dim3 grid(n / kBlockTileN, m / kBlockTileM);
    matmul_kernel<<<grid, block>>>(a, b, c, m, n, k);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
}

// ---- host boilerplate (FP16 track, identical to K8) ----

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
    std::cout << "matmul [mma_ptx] PASS  m=" << m << " n=" << n << " k=" << k << '\n';
    return EXIT_SUCCESS;
}
