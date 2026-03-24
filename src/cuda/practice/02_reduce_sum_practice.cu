#include "../common/cuda_utils.cuh"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

constexpr int kThreadsPerBlock = 256;

// TODO:
// Start with the atomic reduction version.
// Each thread can load one value and atomicAdd into output[0].
__global__ void reduce_sum_atomic_kernel(
    const float* input,
    float* output,
    int count) {
  // TODO
}

double reduce_sum_cpu(const std::vector<float>& values) {
  double total = 0.0;
  for (float value : values) {
    total += value;
  }
  return total;
}

}  // namespace

int main() {
  constexpr int count = 4096;
  std::vector<float> host_input(count);
  for (int i = 0; i < count; ++i) {
    host_input[i] = static_cast<float>(i % 7);
  }

  float* device_input = nullptr;
  float* device_output = nullptr;
  CHECK_CUDA(cudaMalloc(&device_input, host_input.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_output, sizeof(float)));
  CHECK_CUDA(cudaMemcpy(
      device_input,
      host_input.data(),
      host_input.size() * sizeof(float),
      cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(device_output, 0, sizeof(float)));

  const int blocks = cuda_utils::ceil_div(count, kThreadsPerBlock);
  reduce_sum_atomic_kernel<<<blocks, kThreadsPerBlock>>>(device_input, device_output, count);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());

  float gpu_sum = 0.0f;
  CHECK_CUDA(cudaMemcpy(&gpu_sum, device_output, sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(device_input));
  CHECK_CUDA(cudaFree(device_output));

  const double cpu_sum = reduce_sum_cpu(host_input);
  if (std::fabs(cpu_sum - gpu_sum) > 1e-3) {
    std::cerr << "reduce_sum practice failed\n";
    return EXIT_FAILURE;
  }

  std::cout << "reduce_sum practice passed\n";
  return EXIT_SUCCESS;
}
