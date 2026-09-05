// bench harness: runs every layernorm version once for ncu.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../common/cuda_utils.cuh"

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kWarpSize = 32;

__global__ void layernorm_naive_kernel(const float* input, const float* gamma, const float* beta,
                                       float* output, int rows, int cols, float eps) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* row_ptr = input + row * cols;
    float* out_ptr = output + row * cols;
    float mean = 0.0f;
    for (int col = 0; col < cols; ++col) mean += row_ptr[col];
    mean /= static_cast<float>(cols);
    float var = 0.0f;
    for (int col = 0; col < cols; ++col) {
        const float d = row_ptr[col] - mean;
        var += d * d;
    }
    var /= static_cast<float>(cols);
    const float inv_std = 1.0f / sqrtf(var + eps);
    for (int col = 0; col < cols; ++col) {
        const float n = (row_ptr[col] - mean) * inv_std;
        out_ptr[col] = n * gamma[col] + beta[col];
    }
}

__global__ void layernorm_block_kernel(const float* input, const float* gamma, const float* beta,
                                       float* output, int rows, int cols, float eps) {
    __shared__ float shared_sum[kThreadsPerBlock];
    __shared__ float shared_var[kThreadsPerBlock];
    const int row = blockIdx.x;
    if (row >= rows) return;
    const float* row_ptr = input + row * cols;
    float* out_ptr = output + row * cols;

    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) local_sum += row_ptr[col];
    shared_sum[threadIdx.x] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) shared_sum[threadIdx.x] += shared_sum[threadIdx.x + offset];
        __syncthreads();
    }
    const float mean = shared_sum[0] / static_cast<float>(cols);

    float local_var = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const float d = row_ptr[col] - mean;
        local_var += d * d;
    }
    shared_var[threadIdx.x] = local_var;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) shared_var[threadIdx.x] += shared_var[threadIdx.x + offset];
        __syncthreads();
    }
    const float inv_std = 1.0f / sqrtf(shared_var[0] / static_cast<float>(cols) + eps);
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const float n = (row_ptr[col] - mean) * inv_std;
        out_ptr[col] = n * gamma[col] + beta[col];
    }
}

__device__ float warp_reduce_sum_bench(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2)
        value += __shfl_down_sync(0xffffffff, value, offset);
    return value;
}

__global__ void layernorm_warp_shuffle_kernel(const float* input, const float* gamma,
                                              const float* beta, float* output, int rows,
                                              int cols, float eps) {
    __shared__ float warp_sums[kThreadsPerBlock / kWarpSize];
    __shared__ float warp_sumsq[kThreadsPerBlock / kWarpSize];
    __shared__ float stats[2];

    const int row = blockIdx.x;
    if (row >= rows) return;
    const float* row_ptr = input + row * cols;
    float* out_ptr = output + row * cols;

    float local_sum = 0.0f;
    float local_sumsq = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const float v = row_ptr[col];
        local_sum += v;
        local_sumsq += v * v;
    }
    local_sum = warp_reduce_sum_bench(local_sum);
    local_sumsq = warp_reduce_sum_bench(local_sumsq);

    const int lane = threadIdx.x % kWarpSize;
    const int warp_id = threadIdx.x / kWarpSize;
    if (lane == 0) {
        warp_sums[warp_id] = local_sum;
        warp_sumsq[warp_id] = local_sumsq;
    }
    __syncthreads();

    if (warp_id == 0) {
        const int num_warps = blockDim.x / kWarpSize;
        float sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        float sumsq = (lane < num_warps) ? warp_sumsq[lane] : 0.0f;
        sum = warp_reduce_sum_bench(sum);
        sumsq = warp_reduce_sum_bench(sumsq);
        if (lane == 0) {
            const float mean = sum / static_cast<float>(cols);
            const float variance = sumsq / static_cast<float>(cols) - mean * mean;
            stats[0] = mean;
            stats[1] = 1.0f / sqrtf(variance + eps);
        }
    }
    __syncthreads();

    const float mean = stats[0];
    const float inv_std = stats[1];
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const float n = (row_ptr[col] - mean) * inv_std;
        out_ptr[col] = n * gamma[col] + beta[col];
    }
}

void fill_input(std::vector<float>& input, std::vector<float>& gamma, std::vector<float>& beta,
                int rows, int cols) {
    for (int row = 0; row < rows; ++row)
        for (int col = 0; col < cols; ++col)
            input[row * cols + col] =
                static_cast<float>((row * 11 + col * 5) % 23) * 0.2f - 1.0f;
    for (int col = 0; col < cols; ++col) {
        gamma[col] = 1.0f + 0.01f * static_cast<float>(col % 7);
        beta[col] = 0.05f * static_cast<float>((col % 5) - 2);
    }
}

}  // namespace

int main(int argc, char** argv) {
    int rows = 64;
    int cols = 256;
    if (argc >= 3) {
        rows = std::atoi(argv[1]);
        cols = std::atoi(argv[2]);
    }
    const float eps = 1e-5f;
    std::vector<float> host_input(rows * cols);
    std::vector<float> host_gamma(cols);
    std::vector<float> host_beta(cols);
    fill_input(host_input, host_gamma, host_beta, rows, cols);

    float* device_input = nullptr;
    float* device_gamma = nullptr;
    float* device_beta = nullptr;
    float* device_output = nullptr;
    CHECK_CUDA(cudaMalloc(&device_input, host_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_gamma, host_gamma.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_beta, host_beta.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_output, host_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(device_input, host_input.data(), host_input.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_gamma, host_gamma.data(), host_gamma.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_beta, host_beta.data(), host_beta.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    {
        const int blocks = cuda_utils::ceil_div(rows, kThreadsPerBlock);
        layernorm_naive_kernel<<<blocks, kThreadsPerBlock>>>(device_input, device_gamma,
                                                             device_beta, device_output, rows,
                                                             cols, eps);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    {
        layernorm_block_kernel<<<rows, kThreadsPerBlock>>>(device_input, device_gamma,
                                                           device_beta, device_output, rows, cols,
                                                           eps);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    {
        layernorm_warp_shuffle_kernel<<<rows, kThreadsPerBlock>>>(device_input, device_gamma,
                                                                  device_beta, device_output,
                                                                  rows, cols, eps);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    CHECK_CUDA(cudaFree(device_input));
    CHECK_CUDA(cudaFree(device_gamma));
    CHECK_CUDA(cudaFree(device_beta));
    CHECK_CUDA(cudaFree(device_output));
    std::cout << "layernorm bench harness done  rows=" << rows << " cols=" << cols << '\n';
    return EXIT_SUCCESS;
}
