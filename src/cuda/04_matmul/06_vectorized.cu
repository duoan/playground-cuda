// version: vectorized loads (K6)
//
// diff vs block_tile_2d (K5):
// - kThreadTileM = kThreadTileN = 4 (each thread owns a 4×4 register tile)
// - block tile scaled to kBlockTileM = kBlockTileN = 64, kBlockTileK = 8
// - A- and B-tile fills use `float4` loads/stores (128 bit LDG/STS)
// - final C write is also `float4`
//
// Point: the previous versions issued 32 × 4 B = 128 B of loads per
// warp, which is *already* a coalesced 128 B transaction — but nvcc
// emitted 4 separate LDG.E.32 instructions. Fusing them into one
// LDG.E.128 halves the number of memory instructions the SM has to
// issue (fewer scheduler slots, same bytes), and the same for STS/STG
// on the SMEM store and C store paths. On memory-instruction-heavy
// kernels this is a real 1.5–2× speedup even though the byte count
// hasn't changed.
//
// Constraint: n and k must be multiples of 4 for the float4 code
// path. The reference kernels in this repo run with n = k =
// multiples of 128 so this always holds; we assert it at launch time.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../common/cuda_utils.cuh"

namespace {

// Block owns 64×64 output tile; 16×16 threads → each thread owns 4×4.
constexpr int kBlockTileM = 64;
constexpr int kBlockTileN = 64;
constexpr int kBlockTileK = 8;
constexpr int kThreadTileM = 4;
constexpr int kThreadTileN = 4;
constexpr int kThreadsX = kBlockTileN / kThreadTileN;   // 16
constexpr int kThreadsY = kBlockTileM / kThreadTileM;   // 16
constexpr int kThreadsPerBlock = kThreadsX * kThreadsY; // 256

__device__ __forceinline__ float4 load_float4(const float* p) {
    return *reinterpret_cast<const float4*>(p);
}
__device__ __forceinline__ void store_float4(float* p, float4 v) {
    *reinterpret_cast<float4*>(p) = v;
}

__global__ void matmul_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float a_tile[kBlockTileM][kBlockTileK];
    __shared__ float b_tile[kBlockTileK][kBlockTileN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * kThreadsX + tx;                 // 0..255

    const int row_base = blockIdx.y * kBlockTileM + ty * kThreadTileM;
    const int col_base = blockIdx.x * kBlockTileN + tx * kThreadTileN;

    float acc[kThreadTileM][kThreadTileN] = {};

    // A tile = 64×8 = 512 floats = 128 float4s; B tile identical. 256
    // threads means each half of the block (tid < 128 / tid >= 128) can
    // load exactly one float4 for its side.
    constexpr int kFloat4PerTileA = (kBlockTileM * kBlockTileK) / 4;   // 128
    constexpr int kFloat4PerTileB = (kBlockTileK * kBlockTileN) / 4;   // 128
    static_assert(kFloat4PerTileA + kFloat4PerTileB == kThreadsPerBlock,
                  "one float4 per thread per K-tile assumed");

    for (int tile_k = 0; tile_k < k; tile_k += kBlockTileK) {
        // Split the 256 threads: first 128 load A, second 128 load B.
        // Each thread issues exactly one float4 (16 B) global load,
        // producing one 128 B coalesced HBM transaction per warp.
        if (tid < kFloat4PerTileA) {
            const int a_r = tid / (kBlockTileK / 4);         // 0..63
            const int a_c = (tid % (kBlockTileK / 4)) * 4;   // 0 or 4
            const int g_row = blockIdx.y * kBlockTileM + a_r;
            const int g_col = tile_k + a_c;
            float4 v = (g_row < m && g_col + 3 < k)
                       ? load_float4(&a[g_row * k + g_col])
                       : float4{0.f, 0.f, 0.f, 0.f};
            store_float4(&a_tile[a_r][a_c], v);
        } else {
            const int local = tid - kFloat4PerTileA;         // 0..127
            const int b_r = local / (kBlockTileN / 4);       // 0..7
            const int b_c = (local % (kBlockTileN / 4)) * 4; // 0,4,...,60
            const int g_row = tile_k + b_r;
            const int g_col = blockIdx.x * kBlockTileN + b_c;
            float4 v = (g_row < k && g_col + 3 < n)
                       ? load_float4(&b[g_row * n + g_col])
                       : float4{0.f, 0.f, 0.f, 0.f};
            store_float4(&b_tile[b_r][b_c], v);
        }
        __syncthreads();

        // ---------- inner K loop, register-tile FMA ----------
        // Each thread reads its 4-wide a_frag and b_frag from SMEM.
#pragma unroll
        for (int inner = 0; inner < kBlockTileK; ++inner) {
            float a_frag[kThreadTileM];
            float b_frag[kThreadTileN];
#pragma unroll
            for (int i = 0; i < kThreadTileM; ++i) {
                a_frag[i] = a_tile[ty * kThreadTileM + i][inner];
            }
#pragma unroll
            for (int j = 0; j < kThreadTileN; ++j) {
                b_frag[j] = b_tile[inner][tx * kThreadTileN + j];
            }
#pragma unroll
            for (int i = 0; i < kThreadTileM; ++i) {
#pragma unroll
                for (int j = 0; j < kThreadTileN; ++j) {
                    acc[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }
        __syncthreads();
    }

    // ---------- store C: one float4 per row of the 4×4 acc block ----------
#pragma unroll
    for (int i = 0; i < kThreadTileM; ++i) {
        const int row = row_base + i;
        if (row < m && col_base + 3 < n) {
            float4 out{acc[i][0], acc[i][1], acc[i][2], acc[i][3]};
            store_float4(&c[row * n + col_base], out);
        } else {
            // Fallback scalar store for the tail.
#pragma unroll
            for (int j = 0; j < kThreadTileN; ++j) {
                const int col = col_base + j;
                if (row < m && col < n) c[row * n + col] = acc[i][j];
            }
        }
    }
}

void launch(const float* a, const float* b, float* c, int m, int n, int k) {
    // float4 requires 16 B alignment; enforce n and k are multiples of 4.
    if ((n & 3) || (k & 3)) {
        std::cerr << "vectorized kernel requires n,k % 4 == 0 (got n=" << n << " k=" << k << ")\n";
        std::abort();
    }
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
    std::cout << "matmul [vectorized] PASS  m=" << m << " n=" << n << " k=" << k << '\n';
    return EXIT_SUCCESS;
}
