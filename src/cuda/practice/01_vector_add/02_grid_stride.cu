// practice: vector_add version = grid-stride
//
// Goal: decouple grid size from data size. Each thread walks the array
// with stride = blockDim.x * gridDim.x.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../../common/cuda_utils.cuh"

namespace {

constexpr int kThreadsPerBlock = 256;

// TODO: grid-stride kernel.
// - index = blockIdx.x * blockDim.x + threadIdx.x
// - stride = blockDim.x * gridDim.x
// - for i in [index, count) step stride: c[i] = a[i] + b[i]
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int count) {
    (void)a;
    (void)b;
    (void)c;
    (void)count;
}

// TODO: launch. Simplest choice: blocks = ceil(count / kThreadsPerBlock).
// Extension: pick blocks from hardware (numSMs * blocksPerSM) instead.
void launch(const float* device_a, const float* device_b, float* device_c, int count) {
    (void)device_a;
    (void)device_b;
    (void)device_c;
    (void)count;
}

// ---- host boilerplate: identical across every practice version ----

void fill_inputs(std::vector<float>& a, std::vector<float>& b) {
    for (int i = 0; i < static_cast<int>(a.size()); ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }
}

void cpu_reference(const std::vector<float>& a, const std::vector<float>& b,
                   std::vector<float>& c) {
    for (int i = 0; i < static_cast<int>(c.size()); ++i) {
        c[i] = a[i] + b[i];
    }
}

bool check_output(const std::vector<float>& got, const std::vector<float>& expected) {
    for (int i = 0; i < static_cast<int>(got.size()); ++i) {
        if (std::fabs(got[i] - expected[i]) > 1e-5f) {
            std::cerr << "Mismatch at " << i << ": got " << got[i]
                      << ", expected " << expected[i] << '\n';
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    int log2n = 20;
    if (argc >= 2) {
        log2n = std::atoi(argv[1]);
    }
    const int count = 1 << log2n;
    const size_t bytes = count * sizeof(float);

    std::vector<float> host_a(count);
    std::vector<float> host_b(count);
    std::vector<float> host_c(count, 0.0f);
    std::vector<float> reference(count, 0.0f);

    fill_inputs(host_a, host_b);
    cpu_reference(host_a, host_b, reference);

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;
    CHECK_CUDA(cudaMalloc(&device_a, bytes));
    CHECK_CUDA(cudaMalloc(&device_b, bytes));
    CHECK_CUDA(cudaMalloc(&device_c, bytes));
    CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), bytes, cudaMemcpyHostToDevice));

    launch(device_a, device_b, device_c, count);

    CHECK_CUDA(cudaMemcpy(host_c.data(), device_c, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(device_a));
    CHECK_CUDA(cudaFree(device_b));
    CHECK_CUDA(cudaFree(device_c));

    if (!check_output(host_c, reference)) {
        return EXIT_FAILURE;
    }
    std::cout << "vector_add [grid_stride] practice PASS  count=" << count << '\n';
    return EXIT_SUCCESS;
}
