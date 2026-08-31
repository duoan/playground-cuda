#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

__global__ void vectorAdd_cuda(const float* a, const float* b, float* c, int N) {
    // each thread process exactly one element of C
    // 1D grid,
    // multi blocks (the limit is 1024 blocks), each of block has blockDim.x theads.
    const unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void vector_add_f32x4_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                        float* __restrict__ c, const int N) {
    int idx = 4 * (blockDim.x * blockIdx.x + threadIdx.x);
    if ((idx + 3) < N) {
        float4 reg_a = *(reinterpret_cast<const float4*>(&a[idx]));
        float4 reg_b = *(reinterpret_cast<const float4*>(&b[idx]));
        float4 reg_c;

        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;

        *reinterpret_cast<float4*>(&c[idx]) = reg_c;
    }
    // handle remaining elements if N is not a multiple of 4
    else {
        for (int i = 0; i < 4; ++i) {
            if ((idx + i) < N) {
                c[idx + i] = a[idx + i] + b[idx + i];
            }
        }
    }
}

void vectorAdd_cpu(const float* a, const float* b, float* c, int N) {
    for (unsigned int idx = 0; idx < N; ++idx) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    unsigned int N = 1 << 20;

    size_t size = N * sizeof(float);

    float* h_a = (float*)std::malloc(size);  // input a
    float* h_b = (float*)std::malloc(size);  // input b
    float* h_c = (float*)std::malloc(size);  // output

    // init the inputs
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    float *d_a, *d_b, *d_c;  // device memory pointers

    // allocate GPU (VRAM) memory
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // copy input a and b from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // launch
    int threadPerBlocks = 1024;
    // int blocks = (N + threadPerBlocks - 1) / threadPerBlocks;
    // vectorAdd_cuda<<<blocks, threadPerBlocks>>>(d_a, d_b, d_c, N);

    int totalVectors = (N + 3) / 4;  // Ceiling division for vector chunks
    int blocksPerGrid = (totalVectors + threadPerBlocks - 1) / threadPerBlocks;
    vector_add_f32x4_kernel<<<blocksPerGrid, threadPerBlocks>>>(d_a, d_b, d_c, N);

    // wait GPU finish the async job
    cudaDeviceSynchronize();

    // copy output from device to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    bool success = true;
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        float diff = std::abs(expected - h_c[i]);
        if (diff > std::numeric_limits<float>::epsilon()) {
            std::cout << "Error at index " << i << ": Got " << h_c[i] << ", expected " << expected
                      << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Success! All elements are correct." << std::endl;
    }

    // ============================================================
    // STEP 8: Clean up allocated memory
    // ============================================================
    // Free host memory (CPU)
    free(h_a);
    free(h_b);
    free(h_c);

    // Free device memory (GPU)
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return success ? 0 : 1;
}
