#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <torch/torch.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>

int main() {
  try {
    const auto device_count = c10::cuda::device_count();

    std::cout << "App: device_query" << '\n';
    std::cout << "torch::cuda::is_available(): " << torch::cuda::is_available() << '\n';
    std::cout << "CUDA devices: " << static_cast<int>(device_count) << '\n';

    if (device_count <= 0) {
      return EXIT_SUCCESS;
    }

    auto tensor = torch::rand({4, 4}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "Tensor device: " << tensor.device() << '\n';
    std::cout << "Tensor mean: " << tensor.mean().item<float>() << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
