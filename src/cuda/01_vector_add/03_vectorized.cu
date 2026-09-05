// version: vectorized (float4 body + scalar tail, one kernel)
//
// diff vs grid_stride:
// - use float4 load/store: one instruction moves 16 bytes instead of 4
// - kernel signature stays float* and does the reinterpret_cast internally
// - grid is sized only for the n_vec body; the first n_tail (< 4) threads
//   also handle the leftover scalar elements in the same launch
//
// This is the production shape (see many PyTorch elementwise kernels).
// Two ways to split body and tail were possible:
//   (A) body kernel takes float4*, host reinterpret_casts before launch,
//       then a separate scalar tail kernel — two launches, more code.
//   (B) one kernel takes float*, does the reinterpret internally, and
//       reuses the first n_tail threads for the tail — one launch.
// (B) is what real code ships. We go straight to it.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../common/cuda_utils.cuh"

namespace {

constexpr int kThreadsPerBlock = 256;

// fused kernel: body uses float4, first n_tail threads also do a scalar
// add for the trailing elements.
//
// For count = 10:
//   n_vec = 2, n_tail = 2, tail_offset = 8
//   tid=0 -> c[0..3] via float4, then c[8] via scalar
//   tid=1 -> c[4..7] via float4, then c[9] via scalar
//   tid>=2 -> body no-op, tail no-op
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int count) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    const int n_vec = count / 4;
    const int n_tail = count - n_vec * 4;
    const int tail_offset = n_vec * 4;

    if (tid < n_vec) {
        const auto* a4 = reinterpret_cast<const float4*>(a);
        const auto* b4 = reinterpret_cast<const float4*>(b);
        auto* c4 = reinterpret_cast<float4*>(c);
        const float4 lhs = a4[tid];
        const float4 rhs = b4[tid];
        c4[tid] = make_float4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w);
    }
    if (tid < n_tail) {
        const int i = tail_offset + tid;
        c[i] = a[i] + b[i];
    }
}

// launch: grid sized to n_vec (or count when count < 4).
void launch(const float* device_a, const float* device_b, float* device_c, int count) {
    const int vector_count = count / 4;
    const int launch_count = vector_count > 0 ? vector_count : count;
    const int blocks = cuda_utils::ceil_div(launch_count, kThreadsPerBlock);
    vector_add_kernel<<<blocks, kThreadsPerBlock>>>(device_a, device_b, device_c, count);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
}

// ---- host boilerplate: identical across every version in this folder ----

void fill_inputs(std::vector<float>& a, std::vector<float>& b) {
    for (int i = 0; i < static_cast<int>(a.size()); ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }
}

void cpu_reference(const std::vector<float>& a, const std::vector<float>& b,
                   std::vector<float>& c) {
    for (int i = 0; i < static_cast<int>(c.size()); ++i) {
        c[i] = a[i] + b[i];
    }
}

bool check_output(const std::vector<float>& got, const std::vector<float>& expected) {
    for (int i = 0; i < static_cast<int>(got.size()); ++i) {
        if (std::fabs(got[i] - expected[i]) > 1e-5f) {
            std::cerr << "Mismatch at " << i << ": got " << got[i]
                      << ", expected " << expected[i] << '\n';
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    int log2n = 20;
    if (argc >= 2) {
        log2n = std::atoi(argv[1]);
    }
    const int count = 1 << log2n;
    const size_t bytes = count * sizeof(float);

    std::vector<float> host_a(count);
    std::vector<float> host_b(count);
    std::vector<float> host_c(count, 0.0f);
    std::vector<float> reference(count, 0.0f);

    fill_inputs(host_a, host_b);
    cpu_reference(host_a, host_b, reference);

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;
    CHECK_CUDA(cudaMalloc(&device_a, bytes));
    CHECK_CUDA(cudaMalloc(&device_b, bytes));
    CHECK_CUDA(cudaMalloc(&device_c, bytes));
    CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), bytes, cudaMemcpyHostToDevice));

    launch(device_a, device_b, device_c, count);

    CHECK_CUDA(cudaMemcpy(host_c.data(), device_c, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(device_a));
    CHECK_CUDA(cudaFree(device_b));
    CHECK_CUDA(cudaFree(device_c));

    if (!check_output(host_c, reference)) {
        return EXIT_FAILURE;
    }
    std::cout << "vector_add [vectorized] PASS  count=" << count << '\n';
    return EXIT_SUCCESS;
}
