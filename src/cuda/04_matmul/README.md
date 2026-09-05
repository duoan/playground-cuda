# 04_matmul

Kernel: `C = A × B` on the GPU (single precision).

## Layout

Each file is one **version**. Each is self-contained. Binaries land at
`build/04_matmul/<version>`.

## Ladder

| # | File                     | Idea                                                    | Point                                                                                                              |
|---|--------------------------|---------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| 1 | `01_naive.cu`            | One thread = one C element, dot product from HBM        | Baseline. All reuse is left on the table.                                                                           |
| 2 | `02_tiled.cu`            | Block loads `kTile × kTile` tiles of A, B into smem     | O(kTile) reuse of each loaded value; the first version that is not HBM-throughput bound.                            |
| 3 | `03_warp_tiled.cu`       | 32 × 8 warp-shaped block, one warp per row × 64 cols    | Introduces "block tile → warp tile → thread tile" hierarchy.                                                        |
| 4 | `04_register_blocked.cu` | Each thread computes a 2 × 2 register block of outputs  | Multiplies register reuse. Every (a_frag, b_frag) drives 4 FMAs.                                                    |
| 5 | `05_pipeline.cu`         | Double-buffered shared memory (ping-pong K tiles)       | Teaching pipeline: preload next K tile while consuming current. Plain FMA (no `cp.async` / `mma.sync`) for clarity. |

## Build / run

```
make build/04_matmul/03_warp_tiled
./build/04_matmul/04_register_blocked 512 512 512
```

Every binary takes optional `m n k` (defaults 128 × 128 × 128).

## Bench

`bench.cu` runs all five versions once in a single binary for `ncu`.

## Practice

`src/cuda/practice/04_matmul/` mirrors this folder with the kernel and
launch stubbed out.
