#include "../common/cuda_utils.cuh"

#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

// TODO:
// Write the naive matmul kernel.
// Each thread should compute one C[row, col].
__global__ void matmul_naive_kernel(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k) {
  // TODO
}

}  // namespace

int main() {
  constexpr int m = 4;
  constexpr int n = 4;
  constexpr int k = 4;

  std::vector<float> host_a(m * k, 1.0f);
  std::vector<float> host_b(k * n, 2.0f);
  std::vector<float> host_c(m * n, 0.0f);

  float* device_a = nullptr;
  float* device_b = nullptr;
  float* device_c = nullptr;
  CHECK_CUDA(cudaMalloc(&device_a, host_a.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_b, host_b.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_c, host_c.size() * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), host_a.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), host_b.size() * sizeof(float), cudaMemcpyHostToDevice));

  dim3 block(16, 16);
  dim3 grid(1, 1);
  matmul_naive_kernel<<<grid, block>>>(device_a, device_b, device_c, m, n, k);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(host_c.data(), device_c, host_c.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(device_a));
  CHECK_CUDA(cudaFree(device_b));
  CHECK_CUDA(cudaFree(device_c));

  std::cout << "matmul practice placeholder ran\n";
  return EXIT_SUCCESS;
}
