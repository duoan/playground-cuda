#include "../common/cuda_utils.cuh"

#include <cstdlib>
#include <iostream>

namespace {

// TODO:
// Practice target:
// - write one dense layer kernel, or
// - call your matmul helper and then apply ReLU / GELU
__global__ void relu_kernel(float* values, int count) {
  // TODO
}

}  // namespace

int main() {
  std::cout << "Open this file and implement the MLP building blocks.\n";
  return EXIT_SUCCESS;
}
