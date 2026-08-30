#include "../common.h"

void transpose_cpu(const float* in, float* out, int num_rows, int num_cols) {
    for (int row = 0; row < num_rows; ++row) {
        for (int col = 0; col < num_cols; ++col) {
            out[col * num_rows + row] = in[row * num_cols + col];
        }
    }
}

__global__ void transpose_kernel(const float* in, float* out, int num_rows, int num_cols) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < num_cols && row < num_rows) {
        int in_idx = row * num_cols + col;
        int out_idx = col * num_rows + row;
        out[out_idx] = in[in_idx];
    }
}

/**
 * Main function demonstrating matrix transpose on both CPU and GPU
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

    std::cout << "Matrix Transpose: " << num_rows << "x" << num_cols << " -> " << num_cols << "x"
              << num_rows << std::endl;

    // Allocate host (CPU) memory for input and output matrices
    // h_ prefix denotes host memory pointers
    float *h_in, *h_out_cpu, *h_out_gpu;
    allocate_host(&h_in, n);       // Input matrix (num_rows × num_cols)
    allocate_host(&h_out_cpu, n);  // CPU output (for verification)
    allocate_host(&h_out_gpu, n);  // GPU output (for comparison)

    // Initialize input matrix with test data
    // Pattern: element at (row, col) = row * num_cols + col
    // This makes verification easy: transpose should swap row and column indices
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            h_in[i * num_cols + j] = static_cast<float>(i * num_cols + j);
        }
    }

    // Run CPU version with timing
    // Timer uses RAII pattern: timing starts at construction, ends at destruction
    {
        CpuTimer cpu_timer("CPU Matrix Transpose");
        transpose_cpu(h_in, h_out_cpu, num_rows, num_cols);
    }  // Timer prints elapsed time here

    // Allocate device (GPU) memory
    // d_ prefix denotes device memory pointers
    // GPU memory allocation is separate from host memory
    float *d_in, *d_out;
    allocate_device(&d_in, n);   // Input matrix on GPU
    allocate_device(&d_out, n);  // Output matrix on GPU

    // Copy data from host to device
    // This is a synchronous operation (blocks until copy completes)
    // GPU kernels require data to be in device memory
    copy_to_device(d_in, h_in, n);

    // Configure 2D kernel launch parameters
    // threadsPerBlock: 2D block dimensions (16x16 = 256 threads per block)
    // 16x16 is a good default: balances occupancy and warp efficiency
    // Each warp has 32 threads, so 16x16 = 256 = 8 warps per block
    dim3 threadsPerBlock(16, 16);  // 16x16 = 256 threads per block

    // blocksPerGrid: 2D grid dimensions covering input matrix
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
    transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, num_rows, num_cols);
    CUDA_CHECK(cudaGetLastError());

    {
        GpuTimer gpu_timer("GPU Matrix Transpose");
        transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, num_rows, num_cols);
    }

    // Copy result back from device to host
    // Synchronous operation: blocks until copy completes
    copy_to_host(h_out_gpu, d_out, n);

    // Verify GPU results against CPU results
    // Uses floating-point tolerance comparison (default tolerance: 1e-5)
    // This ensures numerical differences don't cause false failures
    if (verify_results(h_out_gpu, h_out_cpu, n)) {
        std::cout << "✓ Matrix transpose results match!" << std::endl;

        // Display sample results for verification
        // Shows original and transposed matrices side-by-side
        std::cout << "\nOriginal matrix (first 3x3):" << std::endl;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                std::cout << h_in[i * num_cols + j] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "\nTransposed matrix (first 3x3):" << std::endl;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                std::cout << h_out_gpu[i * num_rows + j] << " ";
            }
            std::cout << std::endl;
        }
    } else {
        std::cerr << "✗ Error: GPU and CPU results do not match!" << std::endl;
    }

    // Clean up memory
    // Free all allocated memory to prevent leaks
    // Important: free device memory before host memory (good practice)
    free_host(h_in);
    free_host(h_out_cpu);
    free_host(h_out_gpu);
    free_device(d_in);
    free_device(d_out);

    return 0;
}
