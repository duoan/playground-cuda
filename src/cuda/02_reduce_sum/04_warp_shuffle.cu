// version: warp-shuffle reduction
//
// diff vs sequential:
// - intra-warp reduction is done in registers via `__shfl_down_sync`
//   instead of shared memory
// - each warp's lane 0 writes its partial into a small shared-memory
//   array; warp 0 then reduces those 8 values with another shuffle
// - no more block-wide tree in shared memory
//
// Point: shared memory has bank-conflict / bandwidth costs; the shuffle
// path is register-only and needs no `__syncthreads` between rounds.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../common/cuda_utils.cuh"

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kWarpSize = 32;

__device__ float warp_reduce_sum(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

// warp-shuffle kernel: warp-level shuffle -> one lane per warp writes
// partial -> warp 0 reduces the warp partials -> atomicAdd to output.
__global__ void reduce_sum_kernel(const float* input, float* output, int count) {
    __shared__ float warp_sums[kThreadsPerBlock / kWarpSize];

    const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    float value = (global_index < count) ? input[global_index] : 0.0f;

    value = warp_reduce_sum(value);

    const int lane = threadIdx.x % kWarpSize;
    const int warp_id = threadIdx.x / kWarpSize;
    if (lane == 0) {
        warp_sums[warp_id] = value;
    }
    __syncthreads();

    if (warp_id == 0) {
        value = (lane < (blockDim.x / kWarpSize)) ? warp_sums[lane] : 0.0f;
        value = warp_reduce_sum(value);
        if (lane == 0) {
            atomicAdd(output, value);
        }
    }
}

// launch: single kernel, atomicAdd finalize.
float launch(const float* device_input, int count) {
    float* device_output = nullptr;
    CHECK_CUDA(cudaMalloc(&device_output, sizeof(float)));
    CHECK_CUDA(cudaMemset(device_output, 0, sizeof(float)));

    const int blocks = cuda_utils::ceil_div(count, kThreadsPerBlock);
    reduce_sum_kernel<<<blocks, kThreadsPerBlock>>>(device_input, device_output, count);
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
    std::cout << "reduce_sum [warp_shuffle] PASS  count=" << count << "  sum=" << got << '\n';
    return EXIT_SUCCESS;
}
