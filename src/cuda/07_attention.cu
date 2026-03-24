#include "common/cuda_utils.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

// This file follows one teaching rule:
// keep the whole optimization ladder in one place.
//
// Current ladder:
// 1. CPU reference
// 2. naive QK^T -> softmax -> PV decomposition
// 3. more structured tiled/online-softmax attention
// 4. causal support on top of the tiled path

// 这个例子是单头 self-attention 的教学版。
// 维度故意很小，目的是让你看懂数据流，而不是追求性能。
constexpr int kSeqLen = 8;
constexpr int kHeadDim = 8;
constexpr int kTileTokens = 4;
constexpr float kScale = 1.0f / 2.8284271247461903f;  // 1 / sqrt(8)

void fill_inputs(std::vector<float>& q, std::vector<float>& k, std::vector<float>& v) {
  for (int i = 0; i < static_cast<int>(q.size()); ++i) {
    q[i] = static_cast<float>((i % 5) - 2) * 0.2f;
  }
  for (int i = 0; i < static_cast<int>(k.size()); ++i) {
    k[i] = static_cast<float>((i % 7) - 3) * 0.15f;
  }
  for (int i = 0; i < static_cast<int>(v.size()); ++i) {
    v[i] = static_cast<float>((i % 6) - 2) * 0.1f;
  }
}

double dot_row(
    const std::vector<float>& a,
    int a_row,
    const std::vector<float>& b,
    int b_row) {
  double acc = 0.0;
  for (int d = 0; d < kHeadDim; ++d) {
    acc += static_cast<double>(a[a_row * kHeadDim + d]) *
           static_cast<double>(b[b_row * kHeadDim + d]);
  }
  return acc;
}

void attention_cpu_reference(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    std::vector<float>& out,
    bool causal) {
  std::vector<double> scores(kSeqLen * kSeqLen, 0.0);
  std::vector<double> probs(kSeqLen * kSeqLen, 0.0);

  for (int row = 0; row < kSeqLen; ++row) {
    double row_max = -1e30;
    for (int col = 0; col < kSeqLen; ++col) {
      const bool allowed = !causal || col <= row;
      const double score = allowed
          ? dot_row(q, row, k, col) * static_cast<double>(kScale)
          : -1e30;
      scores[row * kSeqLen + col] = score;
      row_max = std::max(row_max, score);
    }

    double row_sum = 0.0;
    for (int col = 0; col < kSeqLen; ++col) {
      const double value = std::exp(scores[row * kSeqLen + col] - row_max);
      probs[row * kSeqLen + col] = value;
      row_sum += value;
    }

    for (int d = 0; d < kHeadDim; ++d) {
      double acc = 0.0;
      for (int col = 0; col < kSeqLen; ++col) {
        const double p = probs[row * kSeqLen + col] / row_sum;
        acc += p * static_cast<double>(v[col * kHeadDim + d]);
      }
      out[row * kHeadDim + d] = static_cast<float>(acc);
    }
  }
}

bool check_output(const std::vector<float>& got, const std::vector<float>& expected, const char* label) {
  for (size_t i = 0; i < got.size(); ++i) {
    if (std::fabs(got[i] - expected[i]) > 1e-4f) {
      std::cerr << label << " mismatch at " << i
                << ": got " << got[i]
                << ", expected " << expected[i] << '\n';
      return false;
    }
  }
  return true;
}

// naive attention:
// 1. 先算 score = Q * K^T
// 2. 再对 score 的每一行做 softmax
// 3. 最后算 out = softmax(score) * V
//
// 这个拆法最容易学，但会把很多中间矩阵都写到 global memory。
__global__ void attention_scores_kernel(
    const float* q,
    const float* k,
    float* scores,
    bool causal) {
  const int row = blockIdx.x;
  const int col = threadIdx.x;
  if (row >= kSeqLen || col >= kSeqLen) {
    return;
  }

  if (!causal || col <= row) {
    float acc = 0.0f;
    for (int d = 0; d < kHeadDim; ++d) {
      acc += q[row * kHeadDim + d] * k[col * kHeadDim + d];
    }
    scores[row * kSeqLen + col] = acc * kScale;
  } else {
    scores[row * kSeqLen + col] = -1e30f;
  }
}

__global__ void attention_softmax_kernel(float* scores) {
  const int row = blockIdx.x;
  if (row >= kSeqLen) {
    return;
  }

  __shared__ float shared[kSeqLen];
  const int col = threadIdx.x;

  shared[col] = scores[row * kSeqLen + col];
  __syncthreads();

  // 先找这一行的最大值，避免 exp 之后数值爆掉。
  float row_max = shared[0];
  for (int i = 1; i < kSeqLen; ++i) {
    row_max = fmaxf(row_max, shared[i]);
  }

  float row_sum = 0.0f;
  for (int i = 0; i < kSeqLen; ++i) {
    shared[i] = expf(shared[i] - row_max);
    row_sum += shared[i];
  }

  scores[row * kSeqLen + col] = shared[col] / row_sum;
}

__global__ void attention_value_kernel(
    const float* probs,
    const float* v,
    float* out) {
  const int row = blockIdx.x;
  const int d = threadIdx.x;
  if (row >= kSeqLen || d >= kHeadDim) {
    return;
  }

  float acc = 0.0f;
  for (int col = 0; col < kSeqLen; ++col) {
    acc += probs[row * kSeqLen + col] * v[col * kHeadDim + d];
  }
  out[row * kHeadDim + d] = acc;
}

// tiled / fused attention:
// 这个版本强调的是“一个 query row 和 K/V tiles 协作”。
// 我们不再把 score matrix 完整写出来，而是沿着 sequence 维度一块块地处理。
//
// 为了适合零基础学习，这里保留了在线 softmax 的核心思想：
// - 每次看到一个新 tile，都更新当前 row 的 max 和 sum
// - 最后直接得到输出，不需要存完整 attention matrix
__global__ void attention_tiled_kernel(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    bool causal) {
  const int row = blockIdx.x;
  const int d = threadIdx.x;
  if (row >= kSeqLen || d >= kHeadDim) {
    return;
  }

  __shared__ float q_shared[kHeadDim];
  __shared__ float k_tile[kTileTokens][kHeadDim];
  __shared__ float v_tile[kTileTokens][kHeadDim];
  __shared__ float scores[kTileTokens];
  __shared__ float weights[kTileTokens];
  __shared__ float running_max_shared;
  __shared__ float running_sum_shared;
  __shared__ float tile_max_shared;
  __shared__ float tile_sum_shared;
  __shared__ float old_scale_shared;

  if (d < kHeadDim) {
    q_shared[d] = q[row * kHeadDim + d];
  }
  __syncthreads();

  // 在线 softmax 的核心状态。
  // 每个 thread 只负责自己那一维的输出累加，
  // 但整块 thread 共享同一个 running max / sum。
  if (d == 0) {
    running_max_shared = -1e30f;
    running_sum_shared = 0.0f;
  }
  __syncthreads();

  float acc = 0.0f;

  // 这里把 sequence 维度按 tile 来扫。
  // 一个 tile 只有 4 个 token，这样“块内协作搬数据 -> 算 score -> 更新 softmax”
  // 的逻辑会很清楚。
  for (int tile_start = 0; tile_start < kSeqLen; tile_start += kTileTokens) {
    for (int token = 0; token < kTileTokens; ++token) {
      const int seq_idx = tile_start + token;
      if (seq_idx < kSeqLen) {
        k_tile[token][d] = k[seq_idx * kHeadDim + d];
        v_tile[token][d] = v[seq_idx * kHeadDim + d];
      } else {
        k_tile[token][d] = 0.0f;
        v_tile[token][d] = 0.0f;
      }
    }
    __syncthreads();

    // 只有一个 thread 负责在 tile 内做“数学总结”：
    // 算出这一块的 score、max、sum，然后把这些共享状态广播给其他 thread。
    if (d == 0) {
      tile_max_shared = -1e30f;
      for (int token = 0; token < kTileTokens; ++token) {
        const int seq_idx = tile_start + token;
        const bool allowed = seq_idx < kSeqLen && (!causal || seq_idx <= row);
        if (allowed) {
          float score = 0.0f;
          for (int i = 0; i < kHeadDim; ++i) {
            score += q_shared[i] * k_tile[token][i];
          }
          score *= kScale;
          scores[token] = score;
          tile_max_shared = fmaxf(tile_max_shared, score);
        } else {
          scores[token] = -1e30f;
        }
      }

      tile_sum_shared = 0.0f;
      for (int token = 0; token < kTileTokens; ++token) {
        if (tile_start + token < kSeqLen) {
          tile_sum_shared += expf(scores[token] - tile_max_shared);
        }
      }

      const float new_max = fmaxf(running_max_shared, tile_max_shared);
      old_scale_shared = expf(running_max_shared - new_max);
      running_sum_shared = running_sum_shared * old_scale_shared +
                           tile_sum_shared * expf(tile_max_shared - new_max);
      running_max_shared = new_max;

      for (int token = 0; token < kTileTokens; ++token) {
        if (tile_start + token < kSeqLen) {
          weights[token] = expf(scores[token] - new_max);
        } else {
          weights[token] = 0.0f;
        }
      }
    }
    __syncthreads();

    // 每个 thread 只负责输出向量中的一个维度。
    // 这里的 acc 是“未归一化的分子”，最后再除以 running_sum。
    acc = acc * old_scale_shared;
    for (int token = 0; token < kTileTokens; ++token) {
      acc += weights[token] * v_tile[token][d];
    }

    __syncthreads();
  }

  out[row * kHeadDim + d] = acc / running_sum_shared;
}

void run_naive_attention(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    std::vector<float>& out,
    bool causal) {
  float* device_q = nullptr;
  float* device_k = nullptr;
  float* device_v = nullptr;
  float* device_scores = nullptr;
  float* device_out = nullptr;

  CHECK_CUDA(cudaMalloc(&device_q, q.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_k, k.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_v, v.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_scores, kSeqLen * kSeqLen * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_out, out.size() * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(device_q, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_k, k.data(), k.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_v, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));

  attention_scores_kernel<<<kSeqLen, kSeqLen>>>(device_q, device_k, device_scores, causal);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());

  attention_softmax_kernel<<<kSeqLen, kSeqLen>>>(device_scores);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());

  attention_value_kernel<<<kSeqLen, kHeadDim>>>(device_scores, device_v, device_out);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(out.data(), device_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(device_q));
  CHECK_CUDA(cudaFree(device_k));
  CHECK_CUDA(cudaFree(device_v));
  CHECK_CUDA(cudaFree(device_scores));
  CHECK_CUDA(cudaFree(device_out));
}

void run_tiled_attention(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    std::vector<float>& out,
    bool causal) {
  float* device_q = nullptr;
  float* device_k = nullptr;
  float* device_v = nullptr;
  float* device_out = nullptr;

  CHECK_CUDA(cudaMalloc(&device_q, q.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_k, k.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_v, v.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_out, out.size() * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(device_q, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_k, k.data(), k.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_v, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));

  attention_tiled_kernel<<<kSeqLen, kHeadDim>>>(device_q, device_k, device_v, device_out, causal);
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(out.data(), device_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(device_q));
  CHECK_CUDA(cudaFree(device_k));
  CHECK_CUDA(cudaFree(device_v));
  CHECK_CUDA(cudaFree(device_out));
}

void print_sample(const std::vector<float>& values, const char* label) {
  std::cout << label << ":";
  for (int i = 0; i < static_cast<int>(values.size()) && i < 8; ++i) {
    std::cout << ' ' << values[i];
  }
  std::cout << '\n';
}

}  // namespace

int main() {
  std::vector<float> q(kSeqLen * kHeadDim);
  std::vector<float> k(kSeqLen * kHeadDim);
  std::vector<float> v(kSeqLen * kHeadDim);
  std::vector<float> naive_out(kSeqLen * kHeadDim, 0.0f);
  std::vector<float> tiled_out(kSeqLen * kHeadDim, 0.0f);
  std::vector<float> causal_naive_out(kSeqLen * kHeadDim, 0.0f);
  std::vector<float> causal_tiled_out(kSeqLen * kHeadDim, 0.0f);
  std::vector<float> reference_out(kSeqLen * kHeadDim, 0.0f);
  std::vector<float> causal_reference_out(kSeqLen * kHeadDim, 0.0f);

  fill_inputs(q, k, v);
  attention_cpu_reference(q, k, v, reference_out, false);
  attention_cpu_reference(q, k, v, causal_reference_out, true);
  run_naive_attention(q, k, v, naive_out, false);
  run_tiled_attention(q, k, v, tiled_out, false);
  run_naive_attention(q, k, v, causal_naive_out, true);
  run_tiled_attention(q, k, v, causal_tiled_out, true);

  const bool naive_ok = check_output(naive_out, reference_out, "naive");
  const bool tiled_ok = check_output(tiled_out, reference_out, "tiled");
  const bool causal_naive_ok = check_output(causal_naive_out, causal_reference_out, "causal naive");
  const bool causal_tiled_ok = check_output(causal_tiled_out, causal_reference_out, "causal tiled");
  if (!naive_ok || !tiled_ok || !causal_naive_ok || !causal_tiled_ok) {
    return EXIT_FAILURE;
  }

  std::cout << "App: attention" << '\n';
  std::cout << "Seq len: " << kSeqLen << '\n';
  std::cout << "Head dim: " << kHeadDim << '\n';
  std::cout << '\n';
  std::cout << "[naive version] PASS (non-causal)" << '\n';
  print_sample(naive_out, "  sample output");
  print_sample(reference_out, "  reference output");
  std::cout << "[tiled version] PASS (non-causal)" << '\n';
  print_sample(tiled_out, "  sample output");
  std::cout << "[causal naive version] PASS" << '\n';
  print_sample(causal_naive_out, "  sample output");
  print_sample(causal_reference_out, "  reference output");
  std::cout << "[causal tiled version] PASS" << '\n';
  print_sample(causal_tiled_out, "  sample output");

  return EXIT_SUCCESS;
}
