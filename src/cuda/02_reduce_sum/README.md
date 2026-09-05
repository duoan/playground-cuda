# 02_reduce_sum

Kernel: `sum(input[0..N))` on the GPU.

## Layout

Each file is one **version** on the optimization ladder. Each file is
self-contained (kernel + multi-stage launch + main + CPU reference +
check) and produces its own binary at `build/02_reduce_sum/<version>`.

## Ladder

| # | File                | Idea                                                    | Point                                                                           |
|---|---------------------|---------------------------------------------------------|---------------------------------------------------------------------------------|
| 1 | `01_atomic.cu`      | Every thread `atomicAdd`s into one output.              | Baseline. Simplest correct reduction. Contention-bound.                          |
| 2 | `02_interleaved.cu` | Block reduces via `tid % (2*s) == 0` shared-memory tree.| Introduces multi-stage launch and shared-memory reduction. Warp-divergent + bank-conflict pattern (this is on purpose — the next step fixes both). |
| 3 | `03_sequential.cu`  | Same tree, guard is `tid < offset`.                     | Contiguous active lanes → no intra-warp divergence, no strided-access bank conflicts. |
| 4 | `04_warp_shuffle.cu`| Intra-warp reduce with `__shfl_down_sync` in registers. | Removes the shared-memory tree; keeps only warp partials.                        |
| 5 | `05_chunked.cu`     | Each thread accumulates `kChunkItemsPerThread` inputs first. | Fewer blocks per stage → fewer kernel launches, higher work-per-thread.          |

`03 → 04` is where the shift from "shared memory as scratchpad" to
"registers + shuffle" happens. `04 → 05` is a common trick you'll see
in many reduction / softmax / layernorm kernels.

## Build / run

```
make build/02_reduce_sum/03_sequential
make run APP=02_reduce_sum/04_warp_shuffle
```

Every binary takes an optional `log2n` (default 20). Bench-size:

```
./build/02_reduce_sum/05_chunked 27
```

Each prints `reduce_sum [<name>] PASS  count=<N>  sum=<value>`.

## Bench

`bench.cu` launches every version once in a single binary so
`books/cuda/bench/` can profile them together with `ncu`.

## Practice

`src/cuda/practice/02_reduce_sum/` mirrors this folder with kernel/launch
stubbed out.
