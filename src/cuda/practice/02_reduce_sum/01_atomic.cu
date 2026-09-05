// practice: reduce_sum version = atomic
//
// Goal: every thread `atomicAdd`s its input into a single output float.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../../common/cuda_utils.cuh"

namespace {

constexpr int kThreadsPerBlock = 256;

// TODO: atomic kernel.
// - grid-stride loop over `count`
// - each thread: atomicAdd(output, input[i])
__global__ void reduce_sum_kernel(const float* input, float* output, int count) {
    (void)input;
    (void)output;
    (void)count;
}

// TODO: launch.
// - allocate a single-float output, memset to 0
// - grid = ceil(count / block); launch reduce_sum_kernel
// - copy result back, free output
float launch(const float* device_input, int count) {
    (void)device_input;
    (void)count;
    return 0.0f;
}

// ---- host boilerplate: identical across every practice version ----

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
    std::cout << "reduce_sum [atomic] practice PASS  count=" << count << "  sum=" << got << '\n';
    return EXIT_SUCCESS;
}
