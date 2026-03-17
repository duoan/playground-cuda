#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <torch/torch.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {

__global__ void add_scalar_kernel(float* data, int64_t count, float value) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    data[index] += value;
  }
}

}  // namespace

int main() {
  try {
    const auto device_count = c10::cuda::device_count();
    if (device_count <= 0 || !torch::cuda::is_available()) {
      throw std::runtime_error("CUDA is not available to libtorch.");
    }

    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    auto tensor = torch::zeros({16}, options);

    constexpr int threads_per_block = 256;
    const int64_t count = tensor.numel();
    const int blocks = static_cast<int>((count + threads_per_block - 1) / threads_per_block);

    add_scalar_kernel<<<blocks, threads_per_block>>>(tensor.data_ptr<float>(), count, 3.0f);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "App: vector_add" << '\n';
    std::cout << "CUDA devices: " << static_cast<int>(device_count) << '\n';
    std::cout << "Tensor device: " << tensor.device() << '\n';
    std::cout << "Tensor values: " << tensor.cpu() << std::endl;

  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
