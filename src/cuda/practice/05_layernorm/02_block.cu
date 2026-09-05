// practice: layernorm version = block

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../../common/cuda_utils.cuh"

namespace {

constexpr int kThreadsPerBlock = 256;

// TODO: block kernel — one block per row.
// - two shared arrays of size kThreadsPerBlock: shared_sum, shared_var
// - each thread accumulates local_sum over stride blockDim.x
// - tree reduction -> mean = shared_sum[0] / cols
// - each thread accumulates local_var = (x - mean)^2 over stride
// - tree reduction -> inv_std = 1 / sqrt(shared_var[0] / cols + eps)
// - normalize + affine: out[col] = (x - mean) * inv_std * gamma[col] + beta[col]
__global__ void layernorm_kernel(const float* input, const float* gamma, const float* beta,
                                 float* output, int rows, int cols, float eps) {
    (void)input; (void)gamma; (void)beta; (void)output;
    (void)rows; (void)cols; (void)eps;
}

// TODO: launch. grid = rows; block = kThreadsPerBlock.
void launch(const float* device_input, const float* device_gamma, const float* device_beta,
            float* device_output, int rows, int cols, float eps) {
    (void)device_input; (void)device_gamma; (void)device_beta; (void)device_output;
    (void)rows; (void)cols; (void)eps;
}

// ---- host boilerplate: identical across every version in this folder ----

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

void layernorm_cpu(const std::vector<float>& input, const std::vector<float>& gamma,
                   const std::vector<float>& beta, std::vector<float>& output, int rows, int cols,
                   float eps) {
    for (int row = 0; row < rows; ++row) {
        const float* row_ptr = input.data() + row * cols;
        float* out_ptr = output.data() + row * cols;
        double mean = 0.0;
        for (int col = 0; col < cols; ++col) mean += static_cast<double>(row_ptr[col]);
        mean /= static_cast<double>(cols);
        double var = 0.0;
        for (int col = 0; col < cols; ++col) {
            const double d = static_cast<double>(row_ptr[col]) - mean;
            var += d * d;
        }
        var /= static_cast<double>(cols);
        const double inv_std = 1.0 / std::sqrt(var + static_cast<double>(eps));
        for (int col = 0; col < cols; ++col) {
            const double normalized = (static_cast<double>(row_ptr[col]) - mean) * inv_std;
            out_ptr[col] = static_cast<float>(normalized * gamma[col] + beta[col]);
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
    int cols = 256;
    if (argc >= 3) {
        rows = std::atoi(argv[1]);
        cols = std::atoi(argv[2]);
    }
    const float eps = 1e-5f;

    std::vector<float> host_input(rows * cols);
    std::vector<float> host_gamma(cols);
    std::vector<float> host_beta(cols);
    std::vector<float> host_output(rows * cols, 0.0f);
    std::vector<float> reference(rows * cols, 0.0f);
    fill_input(host_input, host_gamma, host_beta, rows, cols);
    layernorm_cpu(host_input, host_gamma, host_beta, reference, rows, cols, eps);

    float* device_input = nullptr;
    float* device_gamma = nullptr;
    float* device_beta = nullptr;
    float* device_output = nullptr;
    CHECK_CUDA(cudaMalloc(&device_input, host_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_gamma, host_gamma.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_beta, host_beta.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_output, host_output.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(device_input, host_input.data(), host_input.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_gamma, host_gamma.data(), host_gamma.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_beta, host_beta.data(), host_beta.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    launch(device_input, device_gamma, device_beta, device_output, rows, cols, eps);

    CHECK_CUDA(cudaMemcpy(host_output.data(), device_output, host_output.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(device_input));
    CHECK_CUDA(cudaFree(device_gamma));
    CHECK_CUDA(cudaFree(device_beta));
    CHECK_CUDA(cudaFree(device_output));

    if (!check_output(host_output, reference, rows * cols)) return EXIT_FAILURE;
    std::cout << "layernorm [block] practice PASS  rows=" << rows << " cols=" << cols << '\n';
    return EXIT_SUCCESS;
}
