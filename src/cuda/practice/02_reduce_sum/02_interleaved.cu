// practice: reduce_sum version = interleaved-addressing tree
//
// Goal: block reduces to one partial sum in shared memory, then adds
// that partial into `output` with a single atomicAdd per block. Uses
// the naive `tid % (2*s) == 0` pattern (which is intentionally
// warp-divergent — that's the pedagogical point).

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../../common/cuda_utils.cuh"

namespace {

constexpr int kThreadsPerBlock = 256;

// TODO: interleaved kernel.
// - extern __shared__ float shared[]
// - load: shared[tid] = (idx < count) ? input[idx] : 0; __syncthreads()
// - for (stride = 1; stride < blockDim.x; stride *= 2):
//     if (tid % (2*stride) == 0) shared[tid] += shared[tid + stride];
//     __syncthreads();
// - thread 0 atomicAdd's shared[0] into `output`
__global__ void reduce_sum_kernel(const float* input, float* output, int count) {
    (void)input;
    (void)output;
    (void)count;
}

// TODO: launch (single kernel, atomicAdd finalize).
// - cudaMalloc + cudaMemset output to zero
// - blocks = ceil(count / kThreadsPerBlock)
// - launch kernel with kThreadsPerBlock * sizeof(float) dynamic smem
// - cudaMemcpy final single float back to host
// - cudaFree output
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
    std::cout << "reduce_sum [interleaved] practice PASS  count=" << count << "  sum=" << got << '\n';
    return EXIT_SUCCESS;
}
