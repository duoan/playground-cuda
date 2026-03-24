#include "../common/cuda_utils.cuh"

#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

constexpr int kThreadsPerBlock = 256;

// TODO:
// Write the naive vector add kernel.
// Goal:
// - one thread computes one output element
// - guard against out-of-bounds access
__global__ void vector_add_naive_kernel(
    const float* a,
    const float* b,
    float* c,
    int count) {
  // TODO
}

void fill_inputs(std::vector<float>& a, std::vector<float>& b) {
  for (int i = 0; i < static_cast<int>(a.size()); ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(2 * i);
  }
}

bool check_output(
    const std::vector<float>& a,
    const std::vector<float>& b,
    const std::vector<float>& c) {
  for (int i = 0; i < static_cast<int>(c.size()); ++i) {
    if (c[i] != a[i] + b[i]) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  constexpr int count = 1024;

  std::vector<float> host_a(count);
  std::vector<float> host_b(count);
  std::vector<float> host_c(count, 0.0f);
  fill_inputs(host_a, host_b);

  float* device_a = nullptr;
  float* device_b = nullptr;
  float* device_c = nullptr;
  const size_t bytes = host_a.size() * sizeof(float);

  CHECK_CUDA(cudaMalloc(&device_a, bytes));
  CHECK_CUDA(cudaMalloc(&device_b, bytes));
  CHECK_CUDA(cudaMalloc(&device_c, bytes));

  CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), bytes, cudaMemcpyHostToDevice));

  const int blocks = cuda_utils::ceil_div(count, kThreadsPerBlock);
  vector_add_naive_kernel<<<blocks, kThreadsPerBlock>>>(device_a, device_b, device_c, count);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(host_c.data(), device_c, bytes, cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(device_a));
  CHECK_CUDA(cudaFree(device_b));
  CHECK_CUDA(cudaFree(device_c));

  if (!check_output(host_a, host_b, host_c)) {
    std::cerr << "vector_add practice failed\n";
    return EXIT_FAILURE;
  }

  std::cout << "vector_add practice passed\n";
  return EXIT_SUCCESS;
}
