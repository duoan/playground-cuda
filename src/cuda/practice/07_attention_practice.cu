#include "../common/cuda_utils.cuh"

#include <cstdlib>
#include <iostream>

namespace {

// TODO:
// Break attention into three steps:
// 1. score = QK^T
// 2. prob = softmax(score)
// 3. output = prob * V
//
// Start by implementing only the score kernel.
__global__ void attention_score_kernel(
    const float* q,
    const float* k,
    float* scores,
    int seq_len,
    int head_dim) {
  // TODO
}

}  // namespace

int main() {
  std::cout << "Open this file and implement attention step by step.\n";
  return EXIT_SUCCESS;
}
