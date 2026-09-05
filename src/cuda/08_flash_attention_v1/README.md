# 08_flash_attention_v1

Kernel: single-head streaming attention with online softmax.

## Ladder

| # | File           | Idea                                                  | Point                                                             |
|---|----------------|-------------------------------------------------------|-------------------------------------------------------------------|
| 1 | `01_online.cu` | 1 thread per query row, register-resident state      | Baseline. Online softmax with no shared memory.                    |
| 2 | `02_shared.cu` | 1 block per query row, K/V tiles staged in SMEM     | Same math, block cooperates to load each K/V tile into SRAM.       |

## Build / run

```
make build/08_flash_attention_v1/01_online
./build/08_flash_attention_v1/01_online
```

Every binary checks against a shared CPU reference (Q_count={4}, K_count={16}, head_dim={8}).

## Bench

`bench.cu` launches all versions once for `ncu`.

## Practice

`src/cuda/practice/08_flash_attention_v1/` mirrors this folder with stubs.
