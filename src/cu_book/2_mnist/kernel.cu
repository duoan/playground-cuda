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

/**
 * grad_bias = sum over batch of grad_output    (backward of bias_forward)
 *
 * Shapes (all row-major, contiguous):
 *   grad_output [batch_size, hidden_dim]   upstream gradient at this layer's output
 *   grad_bias   [hidden_dim]               gradient wrt bias
 *
 * Formula:  grad_bias[col] = sum_{b in [0, batch_size)} grad_output[b][col]
 *
 * Why sum over batch:
 *   Forward is  Y[b][col] = X[b][col] + bias[col],  so
 *   dL/dbias[col] = sum_b (dL/dY[b][col]) * (dY[b][col]/dbias[col])
 *                 = sum_b  grad_output[b][col] * 1
 *   The batch dim is what gets reduced because the same bias vector was
 *   broadcast over the batch in forward.
 *
 * Grid convention: 1D grid over the hidden_dim output. One thread per bias
 * element. Each thread does an independent serial reduction along batch.
 *   threadIdx.x -> col
 *
 * Typical launch:
 *   int block = 256;
 *   int grid  = (hidden_dim + block - 1) / block;
 *   bias_backward_kernel<<<grid, block>>>(grad_output, grad_bias, batch_size, hidden_dim);
 *
 * Coalescing:
 *   For a fixed b, the 32 threads of a warp read
 *     grad_output[b * hidden_dim + 0], [+ 1], ..., [+ 31]
 *   -- 32 contiguous floats, one 128B transaction. Ideal.
 *
 * Note: with tiny batch (e.g. 8) the serial reduction is cheap and this kernel
 * is fine. For large batch, a tree/warp-shuffle reduction can help, but that's
 * only worth it once bias-backward shows up in the profile.
 */
__global__ void bias_backward_kernel(
    const float* __restrict__ grad_output,  // input,  [batch_size, hidden_dim]
    float* __restrict__ grad_bias,          // output, [hidden_dim]
    const int batch_size,                   // rows of grad_output (reduction axis)
    const int hidden_dim                    // cols of grad_output, length of grad_bias
) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col < hidden_dim) {
        float acc = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            acc += grad_output[b * hidden_dim + col];
        }
        grad_bias[col] = acc;
    }
}

/**
 * x = max(x, 0)     (ReLU forward, in-place)
 *
 * Element-wise, no cross-thread dependency. Modifies x in place; no separate
 * output buffer needed. Shape of x is anything flattened -- caller passes the
 * total element count.
 *
 * Formula:  x[i] = max(x[i], 0)
 *
 * Grid convention: 1D grid over the flat element count. One thread per element.
 *   threadIdx.x -> flat index
 *
 * Typical launch:
 *   int block = 256;
 *   int grid  = (size + block - 1) / block;
 *   relu_forward_kernel<<<grid, block>>>(x, size);
 *
 * Note: memory-bound (1 load + 1 store per element, one FMA-cheap op). Usually
 * fused into the preceding matmul + bias (see matmul_bias_relu_forward_kernel).
 */
__global__ void relu_forward_kernel(float* x, const int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        x[idx] = fmaxf(0.0f, x[idx]);
    }
}

/**
 * grad *= (x > 0)     (ReLU backward, in-place mask on grad)
 *
 * Forward was  y = max(x, 0),  so
 *   dy/dx = 1  if pre-activation > 0,  else 0
 * We accept the ReLU output (post-activation) as `x` here, which works because
 * for ReLU  (x > 0)  is equivalent to  (pre > 0)  -- positive values pass
 * through unchanged, non-positive values become 0.
 *
 * Formula:  grad[i] = grad[i] * (x[i] > 0 ? 1 : 0)
 *
 * Grid convention: 1D grid over the flat element count. One thread per element.
 *   threadIdx.x -> flat index
 *
 * Typical launch:
 *   int block = 256;
 *   int grid  = (size + block - 1) / block;
 *   relu_backward_kernel<<<grid, block>>>(grad, x, size);
 *
 * `grad` is modified in place -- callers should not assume it survives.
 */
__global__ void relu_backward_kernel(float* grad, const float* __restrict__ x, const int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        grad[idx] *= (x[idx] > 0.0f ? 1.0f : 0.0f);
    }
}

/**
 * Row-wise softmax (in-place), with a numerical-stability shift by row max
 * and a small floor to avoid log(0) in the downstream cross-entropy.
 *
 * Shape (row-major, contiguous):
 *   x [batch_size, hidden_dim]   softmax computed independently per row
 *
 * Formula (per row b):
 *   m       = max_j x[b][j]
 *   e[b][j] = exp(x[b][j] - m)
 *   s       = sum_j e[b][j]
 *   x[b][j] = max(e[b][j] / s, 1e-7)     -- floor keeps log() safe in CE
 *
 * Grid convention: 1 block = 1 row, 1 thread per block. Each row is reduced
 * serially by a single thread (3 passes over the row).
 *   blockIdx.x -> row
 *
 * Typical launch:
 *   softmax_kernel<<<batch_size, 1>>>(x, batch_size, hidden_dim);
 *
 * Performance note:
 *   1 thread / block is *very* under-utilized -- a warp is 32 threads, so 31
 *   lanes sit idle, and the row loop is fully serial. For batch_size = 8 you
 *   only occupy 8 warps out of hundreds available on the GPU. Fine for a
 *   naive baseline; the standard optimization is 1 block per row with
 *   `hidden_dim` threads doing a warp-shuffle reduction for max/sum. Swap
 *   in later if this kernel shows up in profiling.
 *
 * Correctness note:
 *   The max-finding loop redundantly compares x[b][0] against itself once
 *   (harmless, fmaxf(a, a) == a). Could start at col = 1 to save one op.
 */
__global__ void softmax_kernel(float* x, const int batch_size, const int hidden_dim) {
    int row = blockIdx.x;
    if (row < batch_size) {
        // 1. Row max for numerical stability.
        float x_max = x[row * hidden_dim];
        for (int col = 0; col < hidden_dim; ++col) {
            x_max = fmaxf(x_max, x[row * hidden_dim + col]);
        }

        // 2. Exponentiate the shifted values and accumulate the sum.
        float exp_sum = 0.0f;
        for (int col = 0; col < hidden_dim; ++col) {
            int idx = row * hidden_dim + col;
            x[idx] = expf(x[idx] - x_max);
            exp_sum += x[idx];
        }

        // 3. Normalize; floor at 1e-7 so downstream log() is safe.
        for (int col = 0; col < hidden_dim; ++col) {
            int idx = row * hidden_dim + col;
            x[idx] = fmaxf(x[idx] / exp_sum, 1e-7f);
        }
    }
}
