// version: sequential-addressing tree reduction (shared memory)
//
// diff vs interleaved:
// - the active-lane pattern is now `tid < offset`, offset halving each
//   round; contiguous active lanes remove the warp-divergence problem
// - shared[tid] += shared[tid + offset] also removes the strided-access
//   bank-conflict problem from the interleaved kernel
//
// Only the kernel body changes vs 02_interleaved.cu; the single-kernel
// atomicAdd finalize is identical.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../common/cuda_utils.cuh"

namespace {

constexpr int kThreadsPerBlock = 256;

// sequential kernel: classic tree reduction with `tid < offset` guard.
// finalize: block 0 thread 0 atomicAdd's the partial into `output`.
__global__ void reduce_sum_kernel(const float* input, float* output, int count) {
    extern __shared__ float shared[];

    const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    shared[threadIdx.x] = (global_index < count) ? input[global_index] : 0.0f;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            shared[threadIdx.x] += shared[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(output, shared[0]);
    }
}

// launch: single kernel, atomicAdd finalize.
float launch(const float* device_input, int count) {
    float* device_output = nullptr;
    CHECK_CUDA(cudaMalloc(&device_output, sizeof(float)));
    CHECK_CUDA(cudaMemset(device_output, 0, sizeof(float)));

    const int blocks = cuda_utils::ceil_div(count, kThreadsPerBlock);
    reduce_sum_kernel<<<blocks, kThreadsPerBlock, kThreadsPerBlock * sizeof(float)>>>(
        device_input, device_output, count);
    CHECK_LAST_CUDA_ERROR();

    float host_output = 0.0f;
    CHECK_CUDA(cudaMemcpy(&host_output, device_output, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(device_output));
    return host_output;
}

// ---- host boilerplate: identical across every version in this folder ----

void fill_input(std::vector<float>& values) {
    for (int i = 0; i < static_cast<int>(values.size()); ++i) {
        values[i] = static_cast<float>(i % 7);
    }
}

double reduce_sum_cpu(const std::vector<float>& values) {
    double total = 0.0;
    for (float v : values) {
        total += static_cast<double>(v);
    }
    return total;
}

bool check(float got, double expected) {
    const double diff = std::fabs(static_cast<double>(got) - expected);
    if (diff > 1e-3) {
        std::cerr << "Mismatch: got " << got << ", expected " << expected
                  << ", diff " << diff << '\n';
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    int log2n = 20;
    if (argc >= 2) {
        log2n = std::atoi(argv[1]);
    }
    const int count = (1 << log2n) + 37;
    const size_t bytes = count * sizeof(float);

    std::vector<float> host_input(count);
    fill_input(host_input);
    const double expected = reduce_sum_cpu(host_input);

    float* device_input = nullptr;
    CHECK_CUDA(cudaMalloc(&device_input, bytes));
    CHECK_CUDA(cudaMemcpy(device_input, host_input.data(), bytes, cudaMemcpyHostToDevice));

    const float got = launch(device_input, count);

    CHECK_CUDA(cudaFree(device_input));

    if (!check(got, expected)) {
        return EXIT_FAILURE;
    }
    std::cout << "reduce_sum [sequential] PASS  count=" << count << "  sum=" << got << '\n';
    return EXIT_SUCCESS;
}
