// bench harness: runs every reduce_sum version once, single binary,
// for ncu to profile them together.
//
// Kernels are byte-for-byte copies of those in the per-version files;
// only the kernel names carry a per-version suffix so ncu's kernel
// regex ("reduce_sum") groups them properly.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../common/cuda_utils.cuh"

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kWarpSize = 32;
constexpr int kChunkItemsPerThread = 8;

// ---- v1: atomic (per-element) — the pedagogical worst case ----
__global__ void reduce_sum_atomic_kernel(const float* input, float* output, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = index; i < count; i += stride) {
        atomicAdd(output, input[i]);
    }
}

// ---- v2..v5 all finalize with a single atomicAdd per block ----
__global__ void reduce_sum_interleaved_kernel(const float* input, float* output, int count) {
    extern __shared__ float shared[];
    const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    shared[threadIdx.x] = (global_index < count) ? input[global_index] : 0.0f;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (threadIdx.x % (2 * stride) == 0) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(output, shared[0]);
    }
}

__global__ void reduce_sum_sequential_kernel(const float* input, float* output, int count) {
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

__device__ float warp_reduce_sum(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void reduce_sum_warp_shuffle_kernel(const float* input, float* output, int count) {
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

__global__ void reduce_sum_chunked_kernel(const float* input, float* output, int count) {
    extern __shared__ float shared[];
    const int tid = threadIdx.x;
    const int block_start = blockIdx.x * blockDim.x * kChunkItemsPerThread;
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

void fill_input(std::vector<float>& values) {
    for (int i = 0; i < static_cast<int>(values.size()); ++i) {
        values[i] = static_cast<float>(i % 7);
    }
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

    float* device_input = nullptr;
    CHECK_CUDA(cudaMalloc(&device_input, bytes));
    CHECK_CUDA(cudaMemcpy(device_input, host_input.data(), bytes, cudaMemcpyHostToDevice));

    // Every variant finalizes into a single-float `output` via atomicAdd
    // (except v1_atomic which touches this word once per element).
    float* device_output = nullptr;
    CHECK_CUDA(cudaMalloc(&device_output, sizeof(float)));

    // v1: atomic (per-element).
    CHECK_CUDA(cudaMemset(device_output, 0, sizeof(float)));
    {
        const int blocks = cuda_utils::ceil_div(count, kThreadsPerBlock);
        reduce_sum_atomic_kernel<<<blocks, kThreadsPerBlock>>>(device_input, device_output, count);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // v2: interleaved.
    CHECK_CUDA(cudaMemset(device_output, 0, sizeof(float)));
    {
        const int blocks = cuda_utils::ceil_div(count, kThreadsPerBlock);
        reduce_sum_interleaved_kernel<<<blocks, kThreadsPerBlock,
                                        kThreadsPerBlock * sizeof(float)>>>(
            device_input, device_output, count);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // v3: sequential.
    CHECK_CUDA(cudaMemset(device_output, 0, sizeof(float)));
    {
        const int blocks = cuda_utils::ceil_div(count, kThreadsPerBlock);
        reduce_sum_sequential_kernel<<<blocks, kThreadsPerBlock,
                                       kThreadsPerBlock * sizeof(float)>>>(
            device_input, device_output, count);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // v4: warp_shuffle.
    CHECK_CUDA(cudaMemset(device_output, 0, sizeof(float)));
    {
        const int blocks = cuda_utils::ceil_div(count, kThreadsPerBlock);
        reduce_sum_warp_shuffle_kernel<<<blocks, kThreadsPerBlock>>>(device_input, device_output,
                                                                    count);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // v5: chunked.
    CHECK_CUDA(cudaMemset(device_output, 0, sizeof(float)));
    {
        const int block_span = kThreadsPerBlock * kChunkItemsPerThread;
        const int blocks = cuda_utils::ceil_div(count, block_span);
        reduce_sum_chunked_kernel<<<blocks, kThreadsPerBlock, kThreadsPerBlock * sizeof(float)>>>(
            device_input, device_output, count);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    CHECK_CUDA(cudaFree(device_input));
    CHECK_CUDA(cudaFree(device_output));

    std::cout << "reduce_sum bench harness done  count=" << count << '\n';
    return EXIT_SUCCESS;
}
