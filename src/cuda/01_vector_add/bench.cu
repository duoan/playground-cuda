// bench harness: launches every vector_add version once so ncu can
// profile them all in a single run.
//
// This binary is *only* used by books/cuda/bench/*. It duplicates the
// kernel bodies from the per-version files (rather than including them)
// on purpose:
//   - each version file compiles as its own translation unit with its
//     own `namespace {}` (see the per-version *.cu files);
//   - keeping the bench harness independent means changing a version
//     file does not silently break bench numbers unless we also update
//     this harness intentionally.
//
// The kernels below are byte-for-byte copies of the ones in:
//   01_naive.cu, 02_grid_stride.cu, 03_vectorized.cu.
// Only the kernel names get a per-version suffix so ncu's regex can
// tell them apart.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../common/cuda_utils.cuh"

namespace {

constexpr int kThreadsPerBlock = 256;

// ---- v1: naive ----
__global__ void vector_add_naive_kernel(const float* a, const float* b, float* c, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        c[index] = a[index] + b[index];
    }
}

// ---- v2: grid-stride ----
__global__ void vector_add_grid_stride_kernel(const float* a, const float* b, float* c,
                                              int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = index; i < count; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// ---- v3: vectorized (fused body + tail) ----
__global__ void vector_add_vectorized_kernel(const float* a, const float* b, float* c, int count) {
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

void fill_inputs(std::vector<float>& a, std::vector<float>& b) {
    for (int i = 0; i < static_cast<int>(a.size()); ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }
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
    fill_inputs(host_a, host_b);

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;
    CHECK_CUDA(cudaMalloc(&device_a, bytes));
    CHECK_CUDA(cudaMalloc(&device_b, bytes));
    CHECK_CUDA(cudaMalloc(&device_c, bytes));
    CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), bytes, cudaMemcpyHostToDevice));

    // v1: naive — grid = ceil(count / block).
    {
        const int blocks = cuda_utils::ceil_div(count, kThreadsPerBlock);
        vector_add_naive_kernel<<<blocks, kThreadsPerBlock>>>(device_a, device_b, device_c, count);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // v2: grid-stride.
    {
        const int blocks = cuda_utils::ceil_div(count, kThreadsPerBlock);
        vector_add_grid_stride_kernel<<<blocks, kThreadsPerBlock>>>(device_a, device_b, device_c,
                                                                   count);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // v3: vectorized (fused).
    {
        const int vector_count = count / 4;
        const int launch_count = vector_count > 0 ? vector_count : count;
        const int blocks = cuda_utils::ceil_div(launch_count, kThreadsPerBlock);
        vector_add_vectorized_kernel<<<blocks, kThreadsPerBlock>>>(device_a, device_b, device_c,
                                                                   count);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    CHECK_CUDA(cudaFree(device_a));
    CHECK_CUDA(cudaFree(device_b));
    CHECK_CUDA(cudaFree(device_c));

    std::cout << "vector_add bench harness done  count=" << count << '\n';
    return EXIT_SUCCESS;
}
