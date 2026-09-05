// bench harness: runs every matmul version once for ncu.
// Kernel bodies are byte-for-byte copies of the per-version files; the
// only difference is a per-version suffix on the kernel name so the ncu
// regex ("matmul_") groups them.
//
// The 9 versions in this ladder split into two dtype tracks:
//   K0 cuBLAS                             (FP32 in / FP32 out, TF32 internal)
//   K1..K6 FP32×FP32 → FP32               (naive_miscoalesced .. vectorized)
//   K7..K8 FP16×FP16 → FP32               (mma_ptx .. wmma)
// bench allocates one FP32 buffer pair and one FP16 buffer pair; the
// FP16 pair is filled by a device-side cast so we don't have to worry
// about host code paths for FP16.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>

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

using half_t = __half;
using namespace nvcuda;

// ============================================================
// K1: naive miscoalesced (row/col mapping swapped)
// ============================================================
__global__ void matmul_k1_naive_miscoalesced_kernel(const float* a, const float* b, float* c,
                                                   int m, int n, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) return;
    float acc = 0.f;
    for (int inner = 0; inner < k; ++inner) acc += a[row * k + inner] * b[inner * n + col];
    c[row * n + col] = acc;
}

// ============================================================
// K2: naive coalesced
// ============================================================
__global__ void matmul_k2_naive_coalesced_kernel(const float* a, const float* b, float* c,
                                                 int m, int n, int k) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;
    float acc = 0.f;
    for (int inner = 0; inner < k; ++inner) acc += a[row * k + inner] * b[inner * n + col];
    c[row * n + col] = acc;
}

// ============================================================
// K3: SMEM tiled (16×16 tile)
// ============================================================
constexpr int kK3Tile = 16;
__global__ void matmul_k3_smem_tiled_kernel(const float* a, const float* b, float* c,
                                            int m, int n, int k) {
    __shared__ float a_tile[kK3Tile][kK3Tile];
    __shared__ float b_tile[kK3Tile][kK3Tile];
    const int row = blockIdx.y * kK3Tile + threadIdx.y;
    const int col = blockIdx.x * kK3Tile + threadIdx.x;
    float acc = 0.f;
    for (int tile_k = 0; tile_k < k; tile_k += kK3Tile) {
        const int a_col = tile_k + threadIdx.x;
        const int b_row = tile_k + threadIdx.y;
        a_tile[threadIdx.y][threadIdx.x] = (row < m && a_col < k) ? a[row * k + a_col] : 0.f;
        b_tile[threadIdx.y][threadIdx.x] = (b_row < k && col < n) ? b[b_row * n + col] : 0.f;
        __syncthreads();
#pragma unroll
        for (int inner = 0; inner < kK3Tile; ++inner)
            acc += a_tile[threadIdx.y][inner] * b_tile[inner][threadIdx.x];
        __syncthreads();
    }
    if (row < m && col < n) c[row * n + col] = acc;
}

// ============================================================
// K4: 1D block tile (each thread owns 8×1 column of C)
// ============================================================
constexpr int kK4BlockM = 64, kK4BlockN = 64, kK4BlockK = 8, kK4ThreadM = 8;
constexpr int kK4ThreadsX = kK4BlockN;               // 64
constexpr int kK4ThreadsY = kK4BlockM / kK4ThreadM;  // 8
constexpr int kK4Threads = kK4ThreadsX * kK4ThreadsY; // 512
__global__ void matmul_k4_block_tile_1d_kernel(const float* a, const float* b, float* c,
                                               int m, int n, int k) {
    __shared__ float a_tile[kK4BlockM][kK4BlockK];
    __shared__ float b_tile[kK4BlockK][kK4BlockN];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * kK4ThreadsX + tx;
    const int row_base = blockIdx.y * kK4BlockM + ty * kK4ThreadM;
    const int col      = blockIdx.x * kK4BlockN + tx;
    float acc[kK4ThreadM] = {};
    for (int tile_k = 0; tile_k < k; tile_k += kK4BlockK) {
        {
            const int ar = tid / kK4BlockK, ac = tid % kK4BlockK;
            const int g_row = blockIdx.y * kK4BlockM + ar, g_col = tile_k + ac;
            a_tile[ar][ac] = (g_row < m && g_col < k) ? a[g_row * k + g_col] : 0.f;
        }
        {
            const int br = tid / kK4BlockN, bc = tid % kK4BlockN;
            const int g_row = tile_k + br, g_col = blockIdx.x * kK4BlockN + bc;
            b_tile[br][bc] = (g_row < k && g_col < n) ? b[g_row * n + g_col] : 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int inner = 0; inner < kK4BlockK; ++inner) {
            const float b_frag = b_tile[inner][tx];
#pragma unroll
            for (int i = 0; i < kK4ThreadM; ++i)
                acc[i] += a_tile[ty * kK4ThreadM + i][inner] * b_frag;
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < kK4ThreadM; ++i) {
        const int row = row_base + i;
        if (row < m && col < n) c[row * n + col] = acc[i];
    }
}

// ============================================================
// K5: 2D block tile (each thread owns 2×2 register block of C)
// ============================================================
constexpr int kK5ThreadsX = 16, kK5ThreadsY = 16, kK5TM = 2, kK5TN = 2;
constexpr int kK5BlockM = kK5ThreadsY * kK5TM;   // 32
constexpr int kK5BlockN = kK5ThreadsX * kK5TN;   // 32
constexpr int kK5BlockK = 16;
__global__ void matmul_k5_block_tile_2d_kernel(const float* a, const float* b, float* c,
                                               int m, int n, int k) {
    __shared__ float a_tile[kK5BlockM][kK5BlockK];
    __shared__ float b_tile[kK5BlockK][kK5BlockN];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row_base = blockIdx.y * kK5BlockM + ty * kK5TM;
    const int col_base = blockIdx.x * kK5BlockN + tx * kK5TN;
    float acc[kK5TM][kK5TN] = {};
    for (int tile_k = 0; tile_k < k; tile_k += kK5BlockK) {
        const int a_col = tile_k + tx;
        const int a_row0 = blockIdx.y * kK5BlockM + ty;
        const int a_row1 = a_row0 + kK5ThreadsY;
        a_tile[ty][tx] = (a_row0 < m && a_col < k) ? a[a_row0 * k + a_col] : 0.f;
        a_tile[ty + kK5ThreadsY][tx] = (a_row1 < m && a_col < k) ? a[a_row1 * k + a_col] : 0.f;
        const int b_row = tile_k + ty;
        const int b_col0 = blockIdx.x * kK5BlockN + tx;
        const int b_col1 = b_col0 + kK5ThreadsX;
        b_tile[ty][tx] = (b_row < k && b_col0 < n) ? b[b_row * n + b_col0] : 0.f;
        b_tile[ty][tx + kK5ThreadsX] = (b_row < k && b_col1 < n) ? b[b_row * n + b_col1] : 0.f;
        __syncthreads();
#pragma unroll
        for (int inner = 0; inner < kK5BlockK; ++inner) {
            const float a_f0 = a_tile[ty * kK5TM + 0][inner];
            const float a_f1 = a_tile[ty * kK5TM + 1][inner];
            const float b_f0 = b_tile[inner][tx * kK5TN + 0];
            const float b_f1 = b_tile[inner][tx * kK5TN + 1];
            acc[0][0] += a_f0 * b_f0; acc[0][1] += a_f0 * b_f1;
            acc[1][0] += a_f1 * b_f0; acc[1][1] += a_f1 * b_f1;
        }
        __syncthreads();
    }
    for (int i = 0; i < kK5TM; ++i)
        for (int j = 0; j < kK5TN; ++j) {
            const int row = row_base + i, col = col_base + j;
            if (row < m && col < n) c[row * n + col] = acc[i][j];
        }
}

// ============================================================
// K6: vectorized (float4) loads
// ============================================================
constexpr int kK6BlockM = 64, kK6BlockN = 64, kK6BlockK = 8;
constexpr int kK6TM = 4, kK6TN = 4;
constexpr int kK6ThreadsX = kK6BlockN / kK6TN;    // 16
constexpr int kK6ThreadsY = kK6BlockM / kK6TM;    // 16
constexpr int kK6F4A = (kK6BlockM * kK6BlockK) / 4;     // 128
__device__ __forceinline__ float4 load_f4(const float* p) {
    return *reinterpret_cast<const float4*>(p);
}
__device__ __forceinline__ void store_f4(float* p, float4 v) {
    *reinterpret_cast<float4*>(p) = v;
}
__global__ void matmul_k6_vectorized_kernel(const float* a, const float* b, float* c,
                                            int m, int n, int k) {
    __shared__ float a_tile[kK6BlockM][kK6BlockK];
    __shared__ float b_tile[kK6BlockK][kK6BlockN];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * kK6ThreadsX + tx;
    const int row_base = blockIdx.y * kK6BlockM + ty * kK6TM;
    const int col_base = blockIdx.x * kK6BlockN + tx * kK6TN;
    float acc[kK6TM][kK6TN] = {};
    for (int tile_k = 0; tile_k < k; tile_k += kK6BlockK) {
        if (tid < kK6F4A) {
            const int a_r = tid / (kK6BlockK / 4);
            const int a_c = (tid % (kK6BlockK / 4)) * 4;
            const int g_row = blockIdx.y * kK6BlockM + a_r;
            const int g_col = tile_k + a_c;
            float4 v = (g_row < m && g_col + 3 < k) ? load_f4(&a[g_row * k + g_col])
                                                    : float4{0.f, 0.f, 0.f, 0.f};
            store_f4(&a_tile[a_r][a_c], v);
        } else {
            const int local = tid - kK6F4A;
            const int b_r = local / (kK6BlockN / 4);
            const int b_c = (local % (kK6BlockN / 4)) * 4;
            const int g_row = tile_k + b_r;
            const int g_col = blockIdx.x * kK6BlockN + b_c;
            float4 v = (g_row < k && g_col + 3 < n) ? load_f4(&b[g_row * n + g_col])
                                                    : float4{0.f, 0.f, 0.f, 0.f};
            store_f4(&b_tile[b_r][b_c], v);
        }
        __syncthreads();
#pragma unroll
        for (int inner = 0; inner < kK6BlockK; ++inner) {
            float a_frag[kK6TM], b_frag[kK6TN];
#pragma unroll
            for (int i = 0; i < kK6TM; ++i) a_frag[i] = a_tile[ty * kK6TM + i][inner];
#pragma unroll
            for (int j = 0; j < kK6TN; ++j) b_frag[j] = b_tile[inner][tx * kK6TN + j];
#pragma unroll
            for (int i = 0; i < kK6TM; ++i)
#pragma unroll
                for (int j = 0; j < kK6TN; ++j) acc[i][j] += a_frag[i] * b_frag[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < kK6TM; ++i) {
        const int row = row_base + i;
        if (row < m && col_base + 3 < n) {
            float4 out{acc[i][0], acc[i][1], acc[i][2], acc[i][3]};
            store_f4(&c[row * n + col_base], out);
        }
    }
}

// ============================================================
// K7: raw MMA PTX (FP16→FP32, mma.sync.aligned.m16n8k16)
// ============================================================
constexpr int kK7WarpsM = 2, kK7WarpsN = 2;
constexpr int kK7Warps = kK7WarpsM * kK7WarpsN;
constexpr int kK7Threads = kK7Warps * 32;              // 128
constexpr int kK7MmaM = 16, kK7MmaN = 8, kK7MmaK = 16;
constexpr int kK7BlockM = kK7WarpsM * kK7MmaM;         // 32
constexpr int kK7BlockN = kK7WarpsN * kK7MmaN;         // 16
constexpr int kK7BlockK = kK7MmaK;                     // 16
__device__ __forceinline__ uint32_t k7_smem_u32(const void* p) {
    uint32_t out;
    asm volatile("{ .reg .u64 pp; cvta.to.shared.u64 pp, %1; cvt.u32.u64 %0, pp; }"
                 : "=r"(out) : "l"(p));
    return out;
}
__global__ void matmul_k7_mma_ptx_kernel(const half_t* a, const half_t* b, float* c,
                                         int m, int n, int k) {
    __shared__ half_t a_tile[kK7BlockM][kK7BlockK];
    __shared__ half_t b_tile[kK7BlockK][kK7BlockN];
    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int warp_m = warp_id / kK7WarpsN;
    const int warp_n = warp_id % kK7WarpsN;
    const int row_base = blockIdx.y * kK7BlockM + warp_m * kK7MmaM;
    const int col_base = blockIdx.x * kK7BlockN + warp_n * kK7MmaN;
    float acc[4] = {};
    for (int tile_k = 0; tile_k < k; tile_k += kK7BlockK) {
        {
            constexpr int kPerT = (kK7BlockM * kK7BlockK) / kK7Threads;
#pragma unroll
            for (int i = 0; i < kPerT; ++i) {
                const int idx = i * kK7Threads + threadIdx.x;
                const int r = idx / kK7BlockK, c_ = idx % kK7BlockK;
                const int g_row = blockIdx.y * kK7BlockM + r, g_col = tile_k + c_;
                a_tile[r][c_] = (g_row < m && g_col < k) ? a[g_row * k + g_col] : half_t(0);
            }
        }
        {
            constexpr int kPerT = (kK7BlockK * kK7BlockN) / kK7Threads;
#pragma unroll
            for (int i = 0; i < kPerT; ++i) {
                const int idx = i * kK7Threads + threadIdx.x;
                const int r = idx / kK7BlockN, c_ = idx % kK7BlockN;
                const int g_row = tile_k + r, g_col = blockIdx.x * kK7BlockN + c_;
                b_tile[r][c_] = (g_row < k && g_col < n) ? b[g_row * n + g_col] : half_t(0);
            }
        }
        __syncthreads();
        uint32_t a_reg[4], b_reg[2];
        {
            const int row_in = warp_m * kK7MmaM + (lane % 16);
            const int col_in = (lane / 16) * 8;
            const uint32_t addr = k7_smem_u32(&a_tile[row_in][col_in]);
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(a_reg[0]), "=r"(a_reg[1]), "=r"(a_reg[2]), "=r"(a_reg[3])
                         : "r"(addr));
        }
        {
            const int r = (lane < 16) ? ((lane % 8) + (lane / 8) * 8) : 0;
            const uint32_t addr = k7_smem_u32(&b_tile[r][warp_n * kK7MmaN]);
            asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
                         : "=r"(b_reg[0]), "=r"(b_reg[1]) : "r"(addr));
        }
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                     : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                     : "r"(a_reg[0]), "r"(a_reg[1]), "r"(a_reg[2]), "r"(a_reg[3]),
                       "r"(b_reg[0]), "r"(b_reg[1]));
        __syncthreads();
    }
    const int group = lane / 4, tig = lane % 4;
    const int r0 = row_base + group, r1 = row_base + group + 8;
    const int c0 = col_base + tig * 2, c1 = c0 + 1;
    if (r0 < m && c0 < n) c[r0 * n + c0] = acc[0];
    if (r0 < m && c1 < n) c[r0 * n + c1] = acc[1];
    if (r1 < m && c0 < n) c[r1 * n + c0] = acc[2];
    if (r1 < m && c1 < n) c[r1 * n + c1] = acc[3];
}

// ============================================================
// K8: WMMA (nvcuda::wmma fragments)
// ============================================================
constexpr int kK8WarpsM = 2, kK8WarpsN = 2;
constexpr int kK8Threads = kK8WarpsM * kK8WarpsN * 32;   // 128
constexpr int kK8WmmaM = 16, kK8WmmaN = 16, kK8WmmaK = 16;
constexpr int kK8BlockM = kK8WarpsM * kK8WmmaM;   // 32
constexpr int kK8BlockN = kK8WarpsN * kK8WmmaN;   // 32
constexpr int kK8BlockK = kK8WmmaK;
__global__ void matmul_k8_wmma_kernel(const half_t* a, const half_t* b, float* c,
                                      int m, int n, int k) {
    __shared__ half_t a_tile[kK8BlockM][kK8BlockK];
    __shared__ half_t b_tile[kK8BlockK][kK8BlockN];
    const int warp_id = threadIdx.x / 32;
    const int warp_m = warp_id / kK8WarpsN, warp_n = warp_id % kK8WarpsN;
    const int row_base = blockIdx.y * kK8BlockM + warp_m * kK8WmmaM;
    const int col_base = blockIdx.x * kK8BlockN + warp_n * kK8WmmaN;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.f);
    for (int tile_k = 0; tile_k < k; tile_k += kK8BlockK) {
        {
            constexpr int kPerT = (kK8BlockM * kK8BlockK) / kK8Threads;
#pragma unroll
            for (int i = 0; i < kPerT; ++i) {
                const int idx = i * kK8Threads + threadIdx.x;
                const int r = idx / kK8BlockK, c_ = idx % kK8BlockK;
                const int g_row = blockIdx.y * kK8BlockM + r, g_col = tile_k + c_;
                a_tile[r][c_] = (g_row < m && g_col < k) ? a[g_row * k + g_col] : half_t(0);
            }
        }
        {
            constexpr int kPerT = (kK8BlockK * kK8BlockN) / kK8Threads;
#pragma unroll
            for (int i = 0; i < kPerT; ++i) {
                const int idx = i * kK8Threads + threadIdx.x;
                const int r = idx / kK8BlockN, c_ = idx % kK8BlockN;
                const int g_row = tile_k + r, g_col = blockIdx.x * kK8BlockN + c_;
                b_tile[r][c_] = (g_row < k && g_col < n) ? b[g_row * n + g_col] : half_t(0);
            }
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half_t, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half_t, wmma::row_major> b_frag;
        wmma::load_matrix_sync(a_frag, &a_tile[warp_m * kK8WmmaM][0], kK8BlockK);
        wmma::load_matrix_sync(b_frag, &b_tile[0][warp_n * kK8WmmaN], kK8BlockN);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }
    if (row_base < m && col_base < n)
        wmma::store_matrix_sync(&c[row_base * n + col_base], c_frag, n, wmma::mem_row_major);
}

// ============================================================
// helpers
// ============================================================
void fill_inputs(std::vector<float>& a, std::vector<float>& b, int m, int n, int k) {
    for (int row = 0; row < m; ++row)
        for (int col = 0; col < k; ++col)
            a[row * k + col] = static_cast<float>((row + col) % 7);
    for (int row = 0; row < k; ++row)
        for (int col = 0; col < n; ++col)
            b[row * n + col] = static_cast<float>((row * 2 + col) % 5);
}

__global__ void float_to_half_kernel(const float* src, half_t* dst, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2half(src[idx]);
}

}  // namespace

int main(int argc, char** argv) {
    int m = 128, n = 128, k = 128;
    if (argc >= 4) {
        m = std::atoi(argv[1]);
        n = std::atoi(argv[2]);
        k = std::atoi(argv[3]);
    }

    std::vector<float> host_a(m * k), host_b(k * n);
    fill_inputs(host_a, host_b, m, n, k);

    // FP32 buffers
    float *device_a = nullptr, *device_b = nullptr, *device_c = nullptr;
    CHECK_CUDA(cudaMalloc(&device_a, m * k * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_b, k * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_c, m * n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), m * k * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), k * n * sizeof(float), cudaMemcpyHostToDevice));

    // FP16 buffers (device-side cast from the FP32 copies)
    half_t *device_ah = nullptr, *device_bh = nullptr;
    CHECK_CUDA(cudaMalloc(&device_ah, m * k * sizeof(half_t)));
    CHECK_CUDA(cudaMalloc(&device_bh, k * n * sizeof(half_t)));
    {
        const int bs = 256;
        float_to_half_kernel<<<cuda_utils::ceil_div(m * k, bs), bs>>>(device_a, device_ah, m * k);
        float_to_half_kernel<<<cuda_utils::ceil_div(k * n, bs), bs>>>(device_b, device_bh, k * n);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // K0 cuBLAS
    {
        static cublasHandle_t h = nullptr;
        if (!h) { CHECK_CUBLAS(cublasCreate(&h));
                  CHECK_CUBLAS(cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH)); }
        const float alpha = 1.f, beta = 0.f;
        CHECK_CUBLAS(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                 &alpha, device_b, n, device_a, k, &beta, device_c, n));
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    // K1
    {
        dim3 block(16, 16);
        dim3 grid(cuda_utils::ceil_div(m, 16), cuda_utils::ceil_div(n, 16));
        matmul_k1_naive_miscoalesced_kernel<<<grid, block>>>(device_a, device_b, device_c, m, n, k);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    // K2
    {
        dim3 block(16, 16);
        dim3 grid(cuda_utils::ceil_div(n, 16), cuda_utils::ceil_div(m, 16));
        matmul_k2_naive_coalesced_kernel<<<grid, block>>>(device_a, device_b, device_c, m, n, k);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    // K3
    {
        dim3 block(kK3Tile, kK3Tile);
        dim3 grid(cuda_utils::ceil_div(n, kK3Tile), cuda_utils::ceil_div(m, kK3Tile));
        matmul_k3_smem_tiled_kernel<<<grid, block>>>(device_a, device_b, device_c, m, n, k);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    // K4
    {
        dim3 block(kK4ThreadsX, kK4ThreadsY);
        dim3 grid(cuda_utils::ceil_div(n, kK4BlockN), cuda_utils::ceil_div(m, kK4BlockM));
        matmul_k4_block_tile_1d_kernel<<<grid, block>>>(device_a, device_b, device_c, m, n, k);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    // K5
    {
        dim3 block(kK5ThreadsX, kK5ThreadsY);
        dim3 grid(cuda_utils::ceil_div(n, kK5BlockN), cuda_utils::ceil_div(m, kK5BlockM));
        matmul_k5_block_tile_2d_kernel<<<grid, block>>>(device_a, device_b, device_c, m, n, k);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    // K6
    if (!(n & 3) && !(k & 3)) {
        dim3 block(kK6ThreadsX, kK6ThreadsY);
        dim3 grid(cuda_utils::ceil_div(n, kK6BlockN), cuda_utils::ceil_div(m, kK6BlockM));
        matmul_k6_vectorized_kernel<<<grid, block>>>(device_a, device_b, device_c, m, n, k);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    // K7 (needs FP16 inputs; requires m%32=0 n%16=0 k%16=0)
    if (!(m % kK7BlockM) && !(n % kK7BlockN) && !(k % kK7BlockK)) {
        dim3 block(kK7Threads);
        dim3 grid(n / kK7BlockN, m / kK7BlockM);
        matmul_k7_mma_ptx_kernel<<<grid, block>>>(device_ah, device_bh, device_c, m, n, k);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    // K8 WMMA (requires m%32=0 n%32=0 k%16=0)
    if (!(m % kK8BlockM) && !(n % kK8BlockN) && !(k % kK8BlockK)) {
        dim3 block(kK8Threads);
        dim3 grid(n / kK8BlockN, m / kK8BlockM);
        matmul_k8_wmma_kernel<<<grid, block>>>(device_ah, device_bh, device_c, m, n, k);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    CHECK_CUDA(cudaFree(device_a));
    CHECK_CUDA(cudaFree(device_b));
    CHECK_CUDA(cudaFree(device_c));
    CHECK_CUDA(cudaFree(device_ah));
    CHECK_CUDA(cudaFree(device_bh));
    std::cout << "matmul bench harness done  m=" << m << " n=" << n << " k=" << k << '\n';
    return EXIT_SUCCESS;
}
