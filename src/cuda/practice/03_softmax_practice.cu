#include "../common/cuda_utils.cuh"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

// TODO:
// Practice target:
// 1. write a row-wise softmax kernel for one row
// 2. compute max first for numerical stability
// 3. compute sum(exp(x - max))
// 4. normalize each element
__global__ void softmax_row_kernel(
    const float* input,
    float* output,
    int cols) {
  // TODO
}

}  // namespace

int main() {
  constexpr int cols = 8;
  std::vector<float> host_input = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f, 0.0f, -1.0f, 4.0f};
  std::vector<float> host_output(cols, 0.0f);

  float* device_input = nullptr;
  float* device_output = nullptr;
  CHECK_CUDA(cudaMalloc(&device_input, cols * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_output, cols * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(device_input, host_input.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

  softmax_row_kernel<<<1, cols>>>(device_input, device_output, cols);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(host_output.data(), device_output, cols * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(device_input));
  CHECK_CUDA(cudaFree(device_output));

  std::cout << "softmax practice placeholder ran\n";
  return EXIT_SUCCESS;
}
