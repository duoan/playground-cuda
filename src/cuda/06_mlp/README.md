# 06_mlp

Kernel: tiny two-layer MLP `y = W2 * relu(W1 * x + b1) + b2` for a batch of rows.

## Layout

Each file is one **version**; self-contained; binaries at `build/06_mlp/<version>`.

## Ladder

| # | File               | Idea                                              | Point                                                                                       |
|---|--------------------|---------------------------------------------------|---------------------------------------------------------------------------------------------|
| 1 | `01_naive.cu`      | 3 kernels: linear1 → relu → linear2               | Baseline. Everything spelled out; two full global-memory round-trips through hidden state.  |
| 2 | `02_fused.cu`      | Fuse relu into linear1 (2 kernels)                | Classic epilogue fusion: skip one launch and one write pass over the hidden activations.    |
| 3 | `03_tiled_fused.cu`| Single kernel, input tile in smem, hidden in smem | Both matmuls in one launch. Hidden state never touches global memory.                       |

## Build / run

```
make build/06_mlp/02_fused
./build/06_mlp/03_tiled_fused
```

## Bench

`bench.cu` launches all three versions once for `ncu`.

## Practice

`src/cuda/practice/06_mlp/` mirrors this folder with stubs.
