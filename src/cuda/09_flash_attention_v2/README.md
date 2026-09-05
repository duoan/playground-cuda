# 09_flash_attention_v2

Kernel: shared-memory-staged online attention, then warp-role specialisation.

## Ladder

| # | File                     | Idea                                                    | Point                                                                                        |
|---|--------------------------|---------------------------------------------------------|----------------------------------------------------------------------------------------------|
| 1 | `01_tile_staged.cu`      | kTileKeys threads load K/V tile; thread 0 updates state | Introduces the cooperative-load + serial-state pattern.                                       |
| 2 | `02_warp_specialised.cu` | Lane roles: load q / load KV / score / state / accum    | Splits the FlashAttention v2 responsibilities across warp lanes without warp primitives.       |

## Build / run

```
make build/09_flash_attention_v2/01_tile_staged
./build/09_flash_attention_v2/01_tile_staged
```

Every binary checks against a shared CPU reference (Q_count={4}, K_count={16}, head_dim={8}).

## Bench

`bench.cu` launches all versions once for `ncu`.

## Practice

`src/cuda/practice/09_flash_attention_v2/` mirrors this folder with stubs.
