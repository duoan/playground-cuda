# 07_attention

Kernel: single-head self-attention `out = softmax(Q * K^T / sqrt(d)) * V`.
Both versions optionally support causal masking.

## Layout

Each file is one **version**; self-contained; binaries at `build/07_attention/<version>`.

## Ladder

| # | File          | Idea                                            | Point                                                                                              |
|---|---------------|-------------------------------------------------|----------------------------------------------------------------------------------------------------|
| 1 | `01_naive.cu` | 3 kernels: scores → softmax → weighted-sum      | Baseline. Materialises the full [seq × seq] score matrix in global memory.                          |
| 2 | `02_tiled.cu` | 1 kernel: sweep K/V tiles, online softmax       | Never writes the score matrix.  Classic Flash-Attention-flavoured single-pass fused inner loop.    |

## Build / run

```
make build/07_attention/02_tiled
./build/07_attention/02_tiled
```

Each binary self-checks against a CPU reference (both non-causal and causal).

## Bench

`bench.cu` launches both versions once for `ncu`.

## Practice

`src/cuda/practice/07_attention/` mirrors this folder with stubs.
