// version: block (one block per row)
//
// diff vs naive:
// - one *block* now owns a row; threads in the block collaborate on
//   the two reductions (max, then sum-of-exp) and finally the normalize
// - two shared-memory tree reductions per row (sequential-addressing)
//
// Point: softmax = "two reductions + a normalize". This version makes
// that structure explicit and uses all threads in a block to shorten
// the inner loop from O(cols) to O(cols / block).

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

// block kernel: one block per row.
__global__ void softmax_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float shared_max[kThreadsPerBlock];
    __shared__ float shared_sum[kThreadsPerBlock];

    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const float* row_ptr = input + row * cols;
    float* out_ptr = output + row * cols;

    // per-thread partial max over strided cols.
    float local_max = kNegInf;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, row_ptr[col]);
    }
    shared_max[threadIdx.x] = local_max;
    __syncthreads();

    // tree reduction for row max.
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            shared_max[threadIdx.x] =
                fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    const float row_max = shared_max[0];

    // per-thread partial sum of exp(x - max).
    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        local_sum += expf(row_ptr[col] - row_max);
    }
    shared_sum[threadIdx.x] = local_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + offset];
        }
        __syncthreads();
    }
    const float row_sum = shared_sum[0];

    // normalize.
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        out_ptr[col] = expf(row_ptr[col] - row_max) / row_sum;
    }
}

// launch: grid = rows (one block per row).
void launch(const float* device_input, float* device_output, int rows, int cols) {
    softmax_kernel<<<rows, kThreadsPerBlock>>>(device_input, device_output, rows, cols);
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
    std::cout << "softmax [block] PASS  rows=" << rows << " cols=" << cols << '\n';
    return EXIT_SUCCESS;
}
