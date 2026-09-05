// version: grid-stride
//
// diff vs tiled:
// - grid size is decoupled from data size; each thread strides across
//   the array with stride = blockDim.x * gridDim.x
// - one launch, no boundary tail: works for any count
//
// The point is not speed (still memory-bound). The point is a more
// general CUDA skeleton:
// - grid can be sized to hardware (e.g. numSMs * blocksPerSM) instead
//   of ceil(count / block_size)
// - launch overhead is fixed regardless of data size
// - bypasses the 2^31-1 gridDim.x limit for very large arrays
// - warp accesses still coalesced: round-N reads a[stride*N + 0..31]

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../common/cuda_utils.cuh"

namespace {

constexpr int kThreadsPerBlock = 256;

// grid-stride kernel: one thread walks the array with a fixed stride.
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = index; i < count; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// launch: grid size chosen from the data (kept simple here; production
// code often uses numSMs * blocksPerSM instead).
void launch(const float* device_a, const float* device_b, float* device_c, int count) {
    const int blocks = cuda_utils::ceil_div(count, kThreadsPerBlock);
    vector_add_kernel<<<blocks, kThreadsPerBlock>>>(device_a, device_b, device_c, count);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
}

// ---- host boilerplate: identical across every version in this folder ----

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
    std::cout << "vector_add [grid_stride] PASS  count=" << count << '\n';
    return EXIT_SUCCESS;
}
