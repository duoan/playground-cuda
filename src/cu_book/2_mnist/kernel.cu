#include <cuda_runtime.h>

/**
 * C = A @ B
 *
 * Shapes (all row-major, contiguous):
 *   A [m, k]   A[row][i]   at A[row * k + i]
 *   B [k, n]   B[i][col]   at B[i * n + col]
 *   C [m, n]   C[row][col] at C[row * n + col]
 *
 * Formula:  C[row][col] = sum_{i in [0, k)} A[row][i] * B[i][col]
 *
 * Grid convention: one thread per output element of C.
 *   threadIdx.x -> col  (contiguous -> coalesced global loads/stores)
 *   threadIdx.y -> row
 *
 * Typical launch:
 *   dim3 block(16, 16);
 *   dim3 grid((n + 15) / 16, (m + 15) / 16);
 *   matmul_a_b_kernel<<<grid, block>>>(A, B, C, m, n, k);
 */
__global__ void matmul_a_b_kernel(const float* __restrict__ A,  // input,  [m, k]
                                  const float* __restrict__ B,  // input,  [k, n]
                                  float* __restrict__ C,        // output, [m, n]
                                  const int m,                  // rows of A and C
                                  const int n,                  // cols of B and C
                                  const int k  // cols of A, rows of B  (reduction axis)
) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < m && col < n) {
        float acc = 0.0f;
        // dot product of A[row, :] and B[:, col]
        for (int i = 0; i < k; ++i) {
            acc += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = acc;
    }
}

/**
 * C = A^T @ B      (A is stored un-transposed; the kernel reads it transposed)
 *
 * Shapes (all row-major, contiguous):
 *   A [k, m]   A[i][row]   at A[i * m + row]     -- reduction axis is the leading dim
 *   B [k, n]   B[i][col]   at B[i * n + col]
 *   C [m, n]   C[row][col] at C[row * n + col]
 *
 * Formula:  C[row][col] = sum_{i in [0, k)} A[i][row] * B[i][col]
 *
 * Typical use in backward pass, weight gradient:
 *   dW = X^T @ dY
 *   with A = X [batch, D_in], B = dY [batch, H]
 *   so   k = batch, m = D_in, n = H, output dW is [D_in, H].
 *
 * Grid convention: one thread per output element of C.
 *   threadIdx.x -> col
 *   threadIdx.y -> row
 *
 * Typical launch:
 *   dim3 block(16, 16);
 *   dim3 grid((n + 15) / 16, (m + 15) / 16);
 *   matmul_at_b_kernel<<<grid, block>>>(A, B, C, m, n, k);
 */
__global__ void matmul_at_b_kernel(
    const float* __restrict__ A,  // input,  [k, m]  (accessed as A^T)
    const float* __restrict__ B,  // input,  [k, n]
    float* __restrict__ C,        // output, [m, n]
    const int m,                  // cols of A, rows of C
    const int n,                  // cols of B and C
    const int k                   // rows of A and B  (reduction axis)
) {
    //   C[row][col] = sum_i A[i * m + row] * B[i * n + col]
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < m && col < n) {
        float acc = 0.0f;

        // dot product on A[:, row] and B[:, col]
        for (int i = 0; i < k; ++i) {
            const int A_stride = m;
            const int B_stride = n;
            acc += A[i * A_stride + row] * B[i * B_stride + col];
        }
        C[row * n + col] = acc;
    }
}

/**
 * C = A @ B^T      (B is stored un-transposed; the kernel reads it transposed)
 *
 * Shapes (all row-major, contiguous):
 *   A [m, k]   A[row][i]   at A[row * k + i]
 *   B [n, k]   B[col][i]   at B[col * k + i]     -- rows of B are the "columns" of B^T
 *   C [m, n]   C[row][col] at C[row * n + col]
 *
 * Formula:  C[row][col] = sum_{i in [0, k)} A[row][i] * B[col][i]
 *
 * Typical use in backward pass, input gradient:
 *   dX = dY @ W^T
 *   with A = dY [batch, H], B = W [D_in, H]
 *   so   m = batch, n = D_in, k = H, output dX is [batch, D_in].
 *
 * Grid convention: one thread per output element of C.
 *   threadIdx.x -> col
 *   threadIdx.y -> row
 *
 * Typical launch:
 *   dim3 block(16, 16);
 *   dim3 grid((n + 15) / 16, (m + 15) / 16);
 *   matmul_a_bt_kernel<<<grid, block>>>(A, B, C, m, n, k);
 */
__global__ void matmul_a_bt_kernel(
    const float* __restrict__ A,  // input,  [m, k]
    const float* __restrict__ B,  // input,  [n, k]  (accessed as B^T)
    float* __restrict__ C,        // output, [m, n]
    const int m,                  // rows of A and C
    const int n,                  // rows of B, cols of C
    const int k                   // cols of A and B  (reduction axis)
) {
    //   C[row][col] = sum_i A[row * k + i] * B[col * k + i]
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < m && col < n) {
        float acc = 0.0f;

        // dot product on A[row, :] and B[col, :]
        for (int i = 0; i < k; ++i) {
            const int A_stride = k;
            const int B_stride = k;
            acc += A[row * A_stride + i] * B[col * B_stride + i];
        }
        C[row * n + col] = acc;
    }
}

/**
 * Y = X + bias      (bias broadcast along the batch dimension)
 *
 * Shapes (all row-major, contiguous):
 *   X    [batch_size, hidden_dim]   X[row][col] at X[row * hidden_dim + col]
 *   bias [hidden_dim]               bias[col] added to every row of X
 *   Y    [batch_size, hidden_dim]   same layout as X (can alias X for in-place)
 *
 * Formula:  Y[row][col] = X[row][col] + bias[col]
 *
 * Why bias is indexed by col (not row):
 *   In an MLP layer Y = X @ W + b, each of the hidden_dim output neurons has
 *   its own bias, so bias.shape = [hidden_dim]. The same bias vector is added
 *   to every sample in the batch -- i.e. bias broadcasts over the batch (row)
 *   dimension. Indexing bias[col] means "which neuron", not "which sample".
 *
 * Grid convention: one thread per output element of Y.
 *   threadIdx.x -> col  (contiguous -> coalesced global loads/stores;
 *                        also makes bias[col] a coalesced load reused across rows)
 *   threadIdx.y -> row
 *
 * Typical launch:
 *   dim3 block(32, 8);
 *   dim3 grid((hidden_dim + 31) / 32, (batch_size + 7) / 8);
 *   bias_forward_kernel<<<grid, block>>>(X, bias, Y, batch_size, hidden_dim);
 *
 * Note: this kernel is memory-bound and cheap. In practice it is usually
 * fused into the preceding matmul (see matmul_bias_relu_forward_kernel) to
 * avoid a separate kernel launch and a round trip through global memory.
 */
__global__ void bias_forward_kernel(
    const float* __restrict__ X,     // input,  [batch_size, hidden_dim]
    const float* __restrict__ bias,  // input,  [hidden_dim]  (broadcast over batch)
    float* __restrict__ Y,           // output, [batch_size, hidden_dim]
    const int batch_size,            // rows of X and Y
    const int hidden_dim             // cols of X and Y, length of bias
) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < batch_size && col < hidden_dim) {
        Y[row * hidden_dim + col] = X[row * hidden_dim + col] + bias[col];
    }
}
