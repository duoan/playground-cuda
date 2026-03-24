#include "../common/cuda_utils.cuh"

#include <cstdlib>
#include <iostream>

namespace {

// TODO:
// Practice target:
// - compute row mean
// - compute row variance
// - normalize each element
__global__ void layernorm_row_kernel(
    const float* input,
    float* output,
    int cols,
    float epsilon) {
  // TODO
}

}  // namespace

int main() {
  std::cout << "Open this file and implement layernorm_row_kernel.\n";
  return EXIT_SUCCESS;
}
