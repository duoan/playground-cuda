# 03_softmax

Row-wise softmax: `y[row] = softmax(x[row])` on the GPU.

## Layout

Each file is one **version** on the optimization ladder. Each file is
self-contained. Binaries land at `build/03_softmax/<version>`.

## Ladder

| # | File            | Idea                                          | Point                                                                                                    |
|---|-----------------|-----------------------------------------------|----------------------------------------------------------------------------------------------------------|
| 1 | `01_naive.cu`   | One thread per row (3 passes: max, sum, norm) | Baseline. Directly mirrors the CPU version. Bad occupancy on wide rows.                                   |
| 2 | `02_block.cu`   | One block per row, cooperative reduction      | Makes the "two reductions + a normalize" structure of softmax explicit; uses all threads in the block.    |
| 3 | `03_online.cu`  | One thread per row, single-pass online update | Introduces the online (streaming) max/sum trick — the exact one FlashAttention uses to fuse with matmul.   |

### Why only 3

Earlier variants of this file also carried `masked` and `causal`
kernels. Both are semantic variants (different math), not optimization
steps. `causal` in particular is fully covered by the attention chapter
(chapter 7 onwards). They were removed to keep this folder purely an
optimization ladder.

## Build / run

```
make build/03_softmax/02_block
./build/03_softmax/03_online 128 512    # rows=128, cols=512
```

Every binary takes optional `rows` and `cols` (defaults 64×257).

## Bench

`bench.cu` runs all three versions once in a single binary for `ncu`.

## Practice

`src/cuda/practice/03_softmax/` mirrors this folder with the kernel and
launch stubbed out.
