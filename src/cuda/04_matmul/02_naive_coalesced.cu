// version: naive with coalesced global loads
//
// diff vs 01_naive_miscoalesced:
// - (col, row) is mapped to (threadIdx.x, threadIdx.y) so that
//   consecutive lanes of a warp step through consecutive `col`s,
//   which makes their read of b[k, col] a 128 B coalesced access
//
// Point: coalescing is *not* about how many bytes you fetch — it's
// about whether a warp's 32 loads collapse into a single memory
// transaction. Swapping x/y in the thread mapping is enough to move
// from 32 transactions/warp to 1.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../common/cuda_utils.cuh"

namespace {

constexpr int kBlockDim = 16;

// naive kernel: one thread = one C element.
__global__ void matmul_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) {
        return;
    }
    float acc = 0.0f;
    for (int inner = 0; inner < k; ++inner) {
        acc += a[row * k + inner] * b[inner * n + col];
    }
    c[row * n + col] = acc;
}

void launch(const float* a, const float* b, float* c, int m, int n, int k) {
    dim3 block(kBlockDim, kBlockDim);
    dim3 grid(cuda_utils::ceil_div(n, kBlockDim), cuda_utils::ceil_div(m, kBlockDim));
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
    std::cout << "matmul [naive_coalesced] PASS  m=" << m << " n=" << n << " k=" << k << '\n';
    return EXIT_SUCCESS;
}
