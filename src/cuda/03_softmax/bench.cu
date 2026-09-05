// bench harness: launches every softmax version once, for ncu.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "../common/cuda_utils.cuh"

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

__global__ void softmax_naive_kernel(const float* input, float* output, int rows, int cols) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* row_ptr = input + row * cols;
    float* out_ptr = output + row * cols;
    float max_value = row_ptr[0];
    for (int col = 1; col < cols; ++col) max_value = fmaxf(max_value, row_ptr[col]);
    double sum = 0.0;
    for (int col = 0; col < cols; ++col)
        sum += std::exp(static_cast<double>(row_ptr[col] - max_value));
    for (int col = 0; col < cols; ++col)
        out_ptr[col] = static_cast<float>(
            std::exp(static_cast<double>(row_ptr[col] - max_value)) / sum);
}

__global__ void softmax_block_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float shared_max[kThreadsPerBlock];
    __shared__ float shared_sum[kThreadsPerBlock];
    const int row = blockIdx.x;
    if (row >= rows) return;
    const float* row_ptr = input + row * cols;
    float* out_ptr = output + row * cols;

    float local_max = kNegInf;
    for (int col = threadIdx.x; col < cols; col += blockDim.x)
        local_max = fmaxf(local_max, row_ptr[col]);
    shared_max[threadIdx.x] = local_max;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset)
            shared_max[threadIdx.x] =
                fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + offset]);
        __syncthreads();
    }
    const float row_max = shared_max[0];

    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x)
        local_sum += expf(row_ptr[col] - row_max);
    shared_sum[threadIdx.x] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) shared_sum[threadIdx.x] += shared_sum[threadIdx.x + offset];
        __syncthreads();
    }
    const float row_sum = shared_sum[0];

    for (int col = threadIdx.x; col < cols; col += blockDim.x)
        out_ptr[col] = expf(row_ptr[col] - row_max) / row_sum;
}

__global__ void softmax_online_kernel(const float* input, float* output, int rows, int cols) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* row_ptr = input + row * cols;
    float* out_ptr = output + row * cols;
    float row_max = kNegInf;
    double row_sum = 0.0;
    for (int col = 0; col < cols; ++col) {
        const float x = row_ptr[col];
        const float new_row_max = fmaxf(row_max, x);
        row_sum = row_sum * std::exp(static_cast<double>(row_max - new_row_max)) +
                  std::exp(static_cast<double>(x - new_row_max));
        row_max = new_row_max;
    }
    for (int col = 0; col < cols; ++col)
        out_ptr[col] = static_cast<float>(
            std::exp(static_cast<double>(row_ptr[col] - row_max)) / row_sum);
}

void fill_input(std::vector<float>& input, int rows, int cols) {
    for (int row = 0; row < rows; ++row)
        for (int col = 0; col < cols; ++col)
            input[row * cols + col] = static_cast<float>((row * 13 + col * 7) % 31) * 0.1f;
}

}  // namespace

int main(int argc, char** argv) {
    int rows = 64;
    int cols = 257;
    if (argc >= 3) {
        rows = std::atoi(argv[1]);
        cols = std::atoi(argv[2]);
    }
    const size_t bytes = static_cast<size_t>(rows) * cols * sizeof(float);

    std::vector<float> host_input(rows * cols);
    fill_input(host_input, rows, cols);

    float* device_input = nullptr;
    float* device_output = nullptr;
    CHECK_CUDA(cudaMalloc(&device_input, bytes));
    CHECK_CUDA(cudaMalloc(&device_output, bytes));
    CHECK_CUDA(cudaMemcpy(device_input, host_input.data(), bytes, cudaMemcpyHostToDevice));

    {
        const int blocks = cuda_utils::ceil_div(rows, kThreadsPerBlock);
        softmax_naive_kernel<<<blocks, kThreadsPerBlock>>>(device_input, device_output, rows, cols);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    {
        softmax_block_kernel<<<rows, kThreadsPerBlock>>>(device_input, device_output, rows, cols);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    {
        const int blocks = cuda_utils::ceil_div(rows, kThreadsPerBlock);
        softmax_online_kernel<<<blocks, kThreadsPerBlock>>>(device_input, device_output, rows,
                                                           cols);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    CHECK_CUDA(cudaFree(device_input));
    CHECK_CUDA(cudaFree(device_output));
    std::cout << "softmax bench harness done  rows=" << rows << " cols=" << cols << '\n';
    return EXIT_SUCCESS;
}
