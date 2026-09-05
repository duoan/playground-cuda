// practice: matmul version = mma_ptx (K7)
//
// FP16 inputs, FP32 accumulator. One warp cooperates on one
// `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` instruction,
// which computes a 16 × 8 output tile per HMMA.
//
// Register loading is done via `ldmatrix.sync.aligned.x4.m8n8` for A
// and `ldmatrix.sync.aligned.x2.trans.m8n8` for B — these are the
// PTX instructions designed to produce exactly the per-lane layout
// mma.sync consumes.
//
// This is the version *underneath* the WMMA API (K8). Writing it by
// hand shows what a tensor-core-native kernel looks like at the PTX
// level.

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <vector>

#include <cuda_fp16.h>

#include "../../common/cuda_utils.cuh"

namespace {

using half_t = __half;

constexpr int kWarpsM = 2, kWarpsN = 2;
constexpr int kWarpsPerBlock = kWarpsM * kWarpsN;
constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;
constexpr int kMmaM = 16, kMmaN = 8, kMmaK = 16;
constexpr int kBlockTileM = kWarpsM * kMmaM;   // 32
constexpr int kBlockTileN = kWarpsN * kMmaN;   // 16
constexpr int kBlockTileK = kMmaK;             // 16

// TODO:
// - stage A(32×16) and B(16×16) halfs into SMEM (cooperative load)
// - use ldmatrix.x4 to pull A into 4 uint32_t regs per lane, in the
//   layout mma.sync wants
// - use ldmatrix.x2.trans to pull B into 2 uint32_t regs per lane
// - issue mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
// - store 4 float acc values per lane to C using the standard mma
//   output layout (group = laneid/4, tig = laneid%4)
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
    std::cout << "matmul [mma_ptx] practice PASS  m=" << m << " n=" << n << " k=" << k << '\n';
    return EXIT_SUCCESS;
}
