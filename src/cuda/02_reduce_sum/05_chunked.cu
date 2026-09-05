// version: chunked / hierarchical reduction (final ladder step)
//
// diff vs warp_shuffle:
// - each thread accumulates kChunkItemsPerThread values from global
//   memory into a register before entering the block-level reduction
// - grid is sized by chunks, not raw elements: one block spans
//   kThreadsPerBlock * kChunkItemsPerThread elements
//
// Point: (a) fewer blocks (less launch/schedule cost, less atomic
// contention on `output`); (b) 8 loads per thread pipeline through the
// memory subsystem, keeping HBM utilization high while each block's
// scalar work stays small.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../common/cuda_utils.cuh"

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kChunkItemsPerThread = 8;
constexpr int kBlockSpan = kThreadsPerBlock * kChunkItemsPerThread;

// chunked kernel: each thread accumulates kChunkItemsPerThread inputs
// into a register, then the block reduces those with a shared-memory
// tree; block 0 thread 0 atomicAdd's the partial into `output`.
__global__ void reduce_sum_kernel(const float* input, float* output, int count) {
    extern __shared__ float shared[];

    const int tid = threadIdx.x;
    const int block_start = blockIdx.x * kBlockSpan;
    const int thread_start = block_start + tid;

    float local_sum = 0.0f;
#pragma unroll
    for (int item = 0; item < kChunkItemsPerThread; ++item) {
        const int index = thread_start + item * blockDim.x;
        if (index < count) {
            local_sum += input[index];
        }
    }

    shared[tid] = local_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, shared[0]);
    }
}

// launch: single kernel, atomicAdd finalize. Grid = count / kBlockSpan,
// so ~ 1/kChunkItemsPerThread as many atomics as a 1-thread-1-element
// version.
float launch(const float* device_input, int count) {
    float* device_output = nullptr;
    CHECK_CUDA(cudaMalloc(&device_output, sizeof(float)));
    CHECK_CUDA(cudaMemset(device_output, 0, sizeof(float)));

    const int blocks = cuda_utils::ceil_div(count, kBlockSpan);
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
    std::cout << "reduce_sum [chunked] PASS  count=" << count << "  sum=" << got << '\n';
    return EXIT_SUCCESS;
}
