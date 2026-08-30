#include "../common.h"

void matrix_add_cpu(const float* A, const float* B, float* C, int n_rows, int n_cols) {
    for (int r = 0; r < n_rows; ++r) {
        for (int c = 0; c < n_cols; ++c) {
            int i = r * n_cols + c;
            C[i] = A[i] + B[i];
        }
    }
}

// Map .x to column (not row) for coalesced access:
//   - A warp = 32 threads with consecutive threadIdx.x (same .y/.z).
//   - Row-major layout: A[r * n_cols + c], so c-adjacent elements are
//     memory-adjacent.
//   - To let one warp read 32 contiguous floats in a single 128B
//     transaction, threadIdx.x must vary along c (column), not r (row).
//   - Mapping .x to r instead spreads a warp across 32 rows, breaking
//     coalescing and cutting bandwidth by ~32x.
// Host launch must match: dim3 block(32, 8); grid = ((n_cols+31)/32, (n_rows+7)/8).
__global__ void matrix_add_kernel(const float* A, const float* B, float* C, int n_rows,
                                  int n_cols) {
    int c = blockDim.x * blockIdx.x + threadIdx.x;
    int r = blockDim.y * blockIdx.y + threadIdx.y;

    if (r < n_rows && c < n_cols) {
        int i = r * n_cols + c;
        C[i] = A[i] + B[i];
    }
}

/**
 * Main function demonstrating matrix addition on both CPU and GPU
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
    // Matrix dimensions: 1024x1024 = 1,048,576 elements
    // Each matrix is ~4 MB (assuming float = 4 bytes)
    // This size is large enough to demonstrate GPU parallelism benefits
    const int num_rows = 1024;
    const int num_cols = 1024;
    const int n = num_rows * num_cols;  // Total number of elements

    std::cout << "Matrix Addition: " << num_rows << "x" << num_cols << " = " << n << " elements"
              << std::endl;

    // Allocate host (CPU) memory for input and output matrices
    // h_ prefix denotes host memory pointers
    // All matrices have same dimensions: num_rows × num_cols
    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    allocate_host(&h_A, n);      // Input matrix A
    allocate_host(&h_B, n);      // Input matrix B
    allocate_host(&h_C_cpu, n);  // CPU output (for verification)
    allocate_host(&h_C_gpu, n);  // GPU output (for comparison)

    // Initialize input matrices with test data
    // Pattern chosen for easy verification: C[i] = A[i] + B[i] = i + 2*i = 3*i
    // Matrices stored in row-major order: element at (row, col) is at index row*num_cols + col
    for (int i = 0; i < n; ++i) {
        h_A[i] = static_cast<float>(i);      // Matrix A: sequential values [0, 1, 2, ..., n-1]
        h_B[i] = static_cast<float>(i * 2);  // Matrix B: doubled values [0, 2, 4, ..., 2*(n-1)]
    }

    // Run CPU version with timing
    // Timer uses RAII pattern: timing starts at construction, ends at destruction
    {
        Timer cpu_timer("CPU Matrix Addition");
        matrix_add_cpu(h_A, h_B, h_C_cpu, num_rows, num_cols);
    }  // Timer prints elapsed time here

    // Allocate device (GPU) memory
    // d_ prefix denotes device memory pointers
    // GPU memory allocation is separate from host memory
    float *d_A, *d_B, *d_C;
    allocate_device(&d_A, n);  // Input matrix A on GPU
    allocate_device(&d_B, n);  // Input matrix B on GPU
    allocate_device(&d_C, n);  // Output matrix C on GPU

    // Copy data from host to device
    // This is a synchronous operation (blocks until copy completes)
    // GPU kernels require data to be in device memory
    copy_to_device(d_A, h_A, n);
    copy_to_device(d_B, h_B, n);

    // Configure 2D kernel launch parameters
    // threadsPerBlock: 2D block dimensions (16x16 = 256 threads per block)
    // 16x16 is a good default: balances occupancy and warp efficiency
    // Each warp has 32 threads, so 16x16 = 256 = 8 warps per block
    dim3 threadsPerBlock(16, 16);  // 16x16 = 256 threads per block

    // blocksPerGrid: 2D grid dimensions
    // Formula: ceil(num_cols / threadsPerBlock.x) for x-dimension
    //          ceil(num_rows / threadsPerBlock.y) for y-dimension
    // Ceiling division ensures all matrix elements are covered
    dim3 blocksPerGrid(
        (num_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,  // Blocks in x-dimension (columns)
        (num_rows + threadsPerBlock.y - 1) / threadsPerBlock.y   // Blocks in y-dimension (rows)
    );

    std::cout << "GPU: Launching " << blocksPerGrid.x << "x" << blocksPerGrid.y << " blocks with "
              << threadsPerBlock.x << "x" << threadsPerBlock.y << " threads per block" << std::endl;
    std::cout << "Total threads: "
              << blocksPerGrid.x * blocksPerGrid.y * threadsPerBlock.x * threadsPerBlock.y
              << std::endl;

    // Warmup: first launch pays JIT / context init cost.
    matrix_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, num_rows, num_cols);
    CUDA_CHECK(cudaGetLastError());

    {
        GpuTimer gpu_timer("GPU Matrix Addition");
        matrix_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, num_rows, num_cols);
    }

    // Copy result back from device to host
    // Synchronous operation: blocks until copy completes
    copy_to_host(h_C_gpu, d_C, n);

    // Verify GPU results against CPU results
    // Uses floating-point tolerance comparison (default tolerance: 1e-5)
    // This ensures numerical differences don't cause false failures
    if (verify_results(h_C_gpu, h_C_cpu, n)) {
        std::cout << "✓ Matrix addition results match!" << std::endl;
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