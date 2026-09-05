// practice: reduce_sum version = warp-shuffle
//
// Goal: replace the shared-memory tree with __shfl_down_sync inside the
// warp. Each warp's lane 0 writes into a small shared array; warp 0
// then reduces those partials with another shuffle.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../../common/cuda_utils.cuh"

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kWarpSize = 32;

// TODO: warp reduce helper.
// - butterfly with __shfl_down_sync(0xffffffff, value, offset), offset from 16 down to 1
__device__ float warp_reduce_sum(float value) {
    return value;
}

// TODO: warp_shuffle kernel.
// - value = (idx < count) ? input[idx] : 0
// - value = warp_reduce_sum(value)
// - lane = tid % 32, warp_id = tid / 32
// - if lane == 0: warp_sums[warp_id] = value
// - __syncthreads()
// - if warp_id == 0: value = (lane < blockDim.x/32) ? warp_sums[lane] : 0;
//     value = warp_reduce_sum(value); if lane == 0: atomicAdd(output, value)
__global__ void reduce_sum_kernel(const float* input, float* output, int count) {
    (void)input;
    (void)output;
    (void)count;
}

// TODO: launch (single kernel, atomicAdd finalize; no dynamic smem).
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
    std::cout << "reduce_sum [warp_shuffle] practice PASS  count=" << count << "  sum=" << got << '\n';
    return EXIT_SUCCESS;
}
