#include "../common.h"

/**
 * CPU implementation of General Matrix Multiply (GEMM)
 * Time Complexity: O(M * N * K)
 * Space Complexity: O(1) excluding input/output matrices
 */
void gemm_cpu(const float* A,  // a matrix (M x K)
              const float* B,  // a matrix (K x N)
              float* C,        // a matrix (M x N)
              int M, int N, int K) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float acc = 0.0f;  // accumulator
            for (int k = 0; k < K; ++k) {
                int A_idx = row * K + k;  // row-major access (cache-friendly)
                int B_idx = k * N + col;  // column-major access (cache-unfriendly)
                acc += A[A_idx] * B[B_idx];
            }
            int C_idx = row * N + col;  // row-major access (cache-friendly)
            C[C_idx] = acc;
        }
    }
}

__global__ void gemm_kernel(const float* A,  // a matrix (M x K)
                            const float* B,  // a matrix (K x N)
                            float* C,        // a matrix (M x N)
                            int M, int N, int K) {
    // Each thread computes one output element at (row, column).
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            int A_idx = row * K + k;
            int B_idx = k * N + col;
            sum += A[A_idx] * B[B_idx];
        }
        int C_idx = row * N + col;
        C[C_idx] = sum;
    }
}

/**
 * Main function demonstrating GEMM on both CPU and GPU
 *
 * This function:
 * 1. Allocates memory for input and output matrices
 * 2. Initializes test data
 * 3. Runs CPU reference implementation with timing
 * 4. Runs GPU kernel implementation with timing
 * 5. Verifies correctness by comparing CPU and GPU results
 * 6. Cleans up allocated memory
 */
int main() {
    // Matrix dimensions: C[M×N] = A[M×K] * B[K×N]
    // M=512, N=512, K=256 results in:
    // - A: 512×256 = 131,072 elements (~512 KB)
    // - B: 256×512 = 131,072 elements (~512 KB)
    // - C: 512×512 = 262,144 elements (~1 MB)
    // - Total operations: 2 * M * N * K = 134,217,728 FLOPs (~134 MFLOPS)
    const int M_rows = 512, N_cols = 512, K_shared_dim = 256;
    const int size_A = M_rows * K_shared_dim;  // Size of matrix A
    const int size_B = K_shared_dim * N_cols;  // Size of matrix B
    const int size_C = M_rows * N_cols;        // Size of matrix C

    std::cout << "GEMM: C[" << M_rows << "x" << N_cols << "] = A[" << M_rows << "x" << K_shared_dim
              << "] * B[" << K_shared_dim << "x" << N_cols << "]" << std::endl;
    std::cout << "Total operations: " << (long long)M_rows * N_cols * K_shared_dim * 2 << " FLOPs"
              << std::endl;

    // Allocate host (CPU) memory for matrices
    // h_ prefix denotes host memory pointers
    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    allocate_host(&h_A, size_A);      // Input matrix A (M×K)
    allocate_host(&h_B, size_B);      // Input matrix B (K×N)
    allocate_host(&h_C_cpu, size_C);  // CPU output (for verification)
    allocate_host(&h_C_gpu, size_C);  // GPU output (for comparison)

    // Initialize input matrices with test data
    // Pattern chosen for easy verification and debugging
    for (int i = 0; i < size_A; ++i) {
        h_A[i] = static_cast<float>(i % 10);  // Matrix A: values 0-9 repeating
    }
    for (int i = 0; i < size_B; ++i) {
        h_B[i] = static_cast<float>((i * 2) % 10);  // Matrix B: values 0,2,4,6,8 repeating
    }

    // Run CPU version with timing
    // Timer uses RAII pattern: timing starts at construction, ends at destruction
    {
        CpuTimer cpu_timer("CPU GEMM");
        gemm_cpu(h_A, h_B, h_C_cpu, M_rows, N_cols, K_shared_dim);
    }  // Timer prints elapsed time here

    // Allocate device (GPU) memory
    // d_ prefix denotes device memory pointers
    // GPU memory allocation is separate from host memory
    float *d_A, *d_B, *d_C;
    allocate_device(&d_A, size_A);  // Input matrix A on GPU
    allocate_device(&d_B, size_B);  // Input matrix B on GPU
    allocate_device(&d_C, size_C);  // Output matrix C on GPU

    // Copy data from host to device
    // This is a synchronous operation (blocks until copy completes)
    // GPU kernels require data to be in device memory
    copy_to_device(d_A, h_A, size_A);
    copy_to_device(d_B, h_B, size_B);

    // Configure 2D kernel launch parameters
    // threadsPerBlock: 2D block dimensions (16x16 = 256 threads per block)
    // 16x16 is a good default: balances occupancy and warp efficiency
    // Each warp has 32 threads, so 16x16 = 256 = 8 warps per block
    dim3 threadsPerBlock(16, 16);  // 16x16 = 256 threads per block

    // blocksPerGrid: 2D grid dimensions covering output matrix C
    // Formula: ceil(N_cols / threadsPerBlock.x) for x-dimension
    //          ceil(M_rows / threadsPerBlock.y) for y-dimension
    // Ceiling division ensures all output elements are covered
    dim3 blocksPerGrid(
        (N_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,  // Blocks in x-dimension (columns)
        (M_rows + threadsPerBlock.y - 1) / threadsPerBlock.y   // Blocks in y-dimension (rows)
    );

    std::cout << "GPU: Launching " << blocksPerGrid.x << "x" << blocksPerGrid.y << " blocks with "
              << threadsPerBlock.x << "x" << threadsPerBlock.y << " threads per block" << std::endl;
    std::cout << "Total threads: "
              << blocksPerGrid.x * blocksPerGrid.y * threadsPerBlock.x * threadsPerBlock.y
              << std::endl;

    // Run GPU version with timing
    // Kernel launch syntax: kernel_name<<<grid_size, block_size>>>(parameters)
    // dim3 types allow 2D/3D grid and block configurations
    // Kernel launch is asynchronous (returns immediately)
    // cudaDeviceSynchronize() waits for kernel completion
    // Warmup: first launch pays JIT / context init cost.
    gemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M_rows, N_cols, K_shared_dim);
    CUDA_CHECK(cudaGetLastError());

    {
        GpuTimer gpu_timer("GPU GEMM");
        gemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M_rows, N_cols,
                                                        K_shared_dim);
    }

    // Copy result back from device to host
    // Synchronous operation: blocks until copy completes
    copy_to_host(h_C_gpu, d_C, size_C);

    // Verify GPU results against CPU results
    // Uses floating-point tolerance comparison (default tolerance: 1e-5)
    // This ensures numerical differences don't cause false failures
    if (verify_results(h_C_gpu, h_C_cpu, size_C)) {
        std::cout << "✓ GEMM results match!" << std::endl;

        // Display sample results for verification
        std::cout << "\nSample results (first 3x3 of output matrix):" << std::endl;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                std::cout << h_C_gpu[i * N_cols + j] << " ";
            }
            std::cout << std::endl;
        }
    } else {
        std::cerr << "✗ Error: GPU and CPU results do not match!" << std::endl;
    }

    // Clean up memory
    // Free all allocated memory to prevent leaks
    // Important: free device memory before host memory (good practice)
    free_host(h_A);
    free_host(h_B);
    free_host(h_C_cpu);
    free_host(h_C_gpu);
    free_device(d_A);
    free_device(d_B);
    free_device(d_C);

    return 0;
}
