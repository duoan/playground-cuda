# 10_flash_attention_v3

Kernel: warp-shuffle-based scoring, then pipelined K/V staging.

## Ladder

| # | File                    | Idea                                                     | Point                                                                                         |
|---|-------------------------|----------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| 1 | `01_warp_streaming.cu`  | Warp reduces the dot product; lane 0 holds online state | Replaces SMEM scoring with `__shfl_down_sync`.                                                 |
| 2 | `02_pipeline.cu`        | Two-slot SMEM double buffer for K/V (preload/consume)   | Adds a conceptual pipeline: prefetch key i+1 while consuming key i.                            |

## Build / run

```
make build/10_flash_attention_v3/01_warp_streaming
./build/10_flash_attention_v3/01_warp_streaming
```

Every binary checks against a shared CPU reference (Q_count={4}, K_count={16}, head_dim={8}).

## Bench

`bench.cu` launches all versions once for `ncu`.

## Practice

`src/cuda/practice/10_flash_attention_v3/` mirrors this folder with stubs.
