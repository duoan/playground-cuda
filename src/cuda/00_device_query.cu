#include "common/cuda_utils.cuh"

#include <iostream>

int main() {
  int device_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&device_count));

  std::cout << "App: device_query" << '\n';
  std::cout << "CUDA devices: " << device_count << '\n';

  for (int device = 0; device < device_count; ++device) {
    cudaDeviceProp props{};
    CHECK_CUDA(cudaGetDeviceProperties(&props, device));

    std::cout << '\n';
    std::cout << "Device " << device << ": " << props.name << '\n';
    std::cout << "  Compute capability: " << props.major << '.' << props.minor << '\n';
    std::cout << "  Global memory (MB): " << (props.totalGlobalMem / (1024 * 1024)) << '\n';
    std::cout << "  Multiprocessors: " << props.multiProcessorCount << '\n';
    std::cout << "  Max threads per block: " << props.maxThreadsPerBlock << '\n';
    std::cout << "  Warp size: " << props.warpSize << '\n';
  }

  return EXIT_SUCCESS;
}
