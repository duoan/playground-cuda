# 05_layernorm

Kernel: row-wise LayerNorm `y = (x - mean) / sqrt(var + eps) * gamma + beta`.

## Layout

Each file is one **version**; self-contained; binaries at
`build/05_layernorm/<version>`.

## Ladder

| # | File                    | Idea                                                | Point                                                                                             |
|---|-------------------------|-----------------------------------------------------|---------------------------------------------------------------------------------------------------|
| 1 | `01_naive.cu`           | One thread per row, three passes over the row       | Baseline. Directly mirrors the CPU version.                                                        |
| 2 | `02_block.cu`           | One block per row, sequential-addressing tree reductions | Same "two reductions + one elementwise pass" skeleton as softmax block.                              |
| 3 | `03_warp_shuffle.cu`    | Fused sum+sumsq, warp-shuffle intra-warp + smem cross-warp | Fewer barriers, no smem tree; also folds the two reductions into a single pass over the row.       |

## Build / run

```
make build/05_layernorm/02_block
./build/05_layernorm/03_warp_shuffle 128 1024
```

Every binary takes optional `rows` and `cols` (defaults 64 × 256).

## Bench

`bench.cu` launches all three versions once for `ncu`.

## Practice

`src/cuda/practice/05_layernorm/` mirrors this folder with stubs.
