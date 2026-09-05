// version: naive (one thread per row)
//
// A single thread walks a whole row: find max, compute sum of
// exp(x - max), then normalize. Logically the same as CPU code.
//
// Great for illustrating what softmax *is*. Bad for performance:
// each thread does O(cols) work, and the block/warp is under-used
// because thread-parallelism only spans rows, not the inner reduction.

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

// naive kernel: one thread per row.
__global__ void softmax_kernel(const float* input, float* output, int rows, int cols) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const float* row_ptr = input + row * cols;
    float* out_ptr = output + row * cols;

    float max_value = row_ptr[0];
    for (int col = 1; col < cols; ++col) {
        max_value = fmaxf(max_value, row_ptr[col]);
    }
    double sum = 0.0;
    for (int col = 0; col < cols; ++col) {
        sum += std::exp(static_cast<double>(row_ptr[col] - max_value));
    }
    for (int col = 0; col < cols; ++col) {
        out_ptr[col] = static_cast<float>(
            std::exp(static_cast<double>(row_ptr[col] - max_value)) / sum);
    }
}

// launch: grid = ceil(rows / threads_per_block).
void launch(const float* device_input, float* device_output, int rows, int cols) {
    const int blocks = cuda_utils::ceil_div(rows, kThreadsPerBlock);
    softmax_kernel<<<blocks, kThreadsPerBlock>>>(device_input, device_output, rows, cols);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
}

// ---- host boilerplate: identical across every version in this folder ----

void fill_input(std::vector<float>& input, int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            input[row * cols + col] = static_cast<float>((row * 13 + col * 7) % 31) * 0.1f;
        }
    }
}

void softmax_cpu(const std::vector<float>& input, std::vector<float>& output, int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        const float* row_ptr = input.data() + row * cols;
        float* out_ptr = output.data() + row * cols;
        float max_value = row_ptr[0];
        for (int col = 1; col < cols; ++col) {
            max_value = std::max(max_value, row_ptr[col]);
        }
        double sum = 0.0;
        for (int col = 0; col < cols; ++col) {
            sum += std::exp(static_cast<double>(row_ptr[col] - max_value));
        }
        for (int col = 0; col < cols; ++col) {
            out_ptr[col] = static_cast<float>(
                std::exp(static_cast<double>(row_ptr[col] - max_value)) / sum);
        }
    }
}

bool check_output(const std::vector<float>& got, const std::vector<float>& expected, int n) {
    for (int i = 0; i < n; ++i) {
        if (std::fabs(got[i] - expected[i]) > 1e-4f) {
            std::cerr << "Mismatch at " << i << ": got " << got[i] << ", expected " << expected[i]
                      << '\n';
            return false;
        }
    }
    return true;
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
    std::vector<float> host_output(rows * cols, 0.0f);
    std::vector<float> reference(rows * cols, 0.0f);

    fill_input(host_input, rows, cols);
    softmax_cpu(host_input, reference, rows, cols);

    float* device_input = nullptr;
    float* device_output = nullptr;
    CHECK_CUDA(cudaMalloc(&device_input, bytes));
    CHECK_CUDA(cudaMalloc(&device_output, bytes));
    CHECK_CUDA(cudaMemcpy(device_input, host_input.data(), bytes, cudaMemcpyHostToDevice));

    launch(device_input, device_output, rows, cols);

    CHECK_CUDA(cudaMemcpy(host_output.data(), device_output, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(device_input));
    CHECK_CUDA(cudaFree(device_output));

    if (!check_output(host_output, reference, rows * cols)) {
        return EXIT_FAILURE;
    }
    std::cout << "softmax [naive] PASS  rows=" << rows << " cols=" << cols << '\n';
    return EXIT_SUCCESS;
}
