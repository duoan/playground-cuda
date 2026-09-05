// version: atomic
//
// Every thread reads one input and atomicAdd's it into a single output
// address. Simplest reduction. Contention on that address is heavy;
// bandwidth on the atomic path is the bottleneck. Baseline for the
// ladder.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../common/cuda_utils.cuh"

namespace {

constexpr int kThreadsPerBlock = 256;

// atomic kernel: every thread contributes one atomicAdd.
__global__ void reduce_sum_kernel(const float* input, float* output, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = index; i < count; i += stride) {
        atomicAdd(output, input[i]);
    }
}

// launch: one kernel, one output. `output` must be zeroed by the caller.
float launch(const float* device_input, int count) {
    float* device_output = nullptr;
    CHECK_CUDA(cudaMalloc(&device_output, sizeof(float)));
    CHECK_CUDA(cudaMemset(device_output, 0, sizeof(float)));

    const int blocks = cuda_utils::ceil_div(count, kThreadsPerBlock);
    reduce_sum_kernel<<<blocks, kThreadsPerBlock>>>(device_input, device_output, count);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    float host_output = 0.0f;
    CHECK_CUDA(cudaMemcpy(&host_output, device_output, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(device_output));
    return host_output;
}

// ---- host boilerplate: identical across every version in this folder ----

void fill_input(std::vector<float>& values) {
    // small integer pattern keeps CPU/GPU float diffs bounded; we compare
    // against a double-precision reference.
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
    // log2n default 20; bench uses `27` (~128 M elements, >> L2).
    int log2n = 20;
    if (argc >= 2) {
        log2n = std::atoi(argv[1]);
    }
    // +37 so the input is not a nice power of two; forces boundary handling.
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
    std::cout << "reduce_sum [atomic] PASS  count=" << count << "  sum=" << got << '\n';
    return EXIT_SUCCESS;
}
