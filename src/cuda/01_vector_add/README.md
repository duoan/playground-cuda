# 01_vector_add

Kernel: `c[i] = a[i] + b[i]` on the GPU.

## Layout

Each file is one **version** on the optimization ladder. Each file is
self-contained (kernel + launch + main + CPU reference + check) and
produces its own binary at `build/01_vector_add/<version>`.

The host boilerplate (`fill_inputs`, `cpu_reference`, `check_output`,
`main`) is intentionally identical across versions. The only lines that
change between two versions are the kernel and the launch.

That means `diff 01_naive.cu 02_grid_stride.cu` shows exactly the
optimization step — nothing else.

## Ladder

| # | File                   | What changes vs. previous                              | Point                                                                 |
|---|------------------------|--------------------------------------------------------|-----------------------------------------------------------------------|
| 1 | `01_naive.cu`          | —                                                      | Baseline. One thread = one element.                                   |
| 2 | `02_grid_stride.cu`    | Grid size decoupled from data size via stride loop.    | General CUDA skeleton. No 2^31 gridDim.x limit. Works for any count.  |
| 3 | `03_vectorized.cu`     | `float4` load/store; body + tail fused into one kernel.| Wider memory transactions (`LDG.E.128`). Production shape.            |

Two earlier versions were dropped as not carrying enough teaching weight:

- **tiled (multi-items-per-thread)**: subsumed by `grid_stride`, which
  already has each thread walk multiple elements. Same coalescing story.
- **vectorized (two kernels)**: the "external `reinterpret_cast` +
  separate tail kernel" split is easier to read but nobody ships it.
  Skipped in favor of the fused single-kernel form directly.

## Build / run

```
make build/01_vector_add/01_naive          # build one version
make all                                   # build every version of every kernel
make run APP=01_vector_add/02_grid_stride  # build + run
```

Every binary takes an optional `log2n` (default 20). Bench-size run:

```
./build/01_vector_add/03_vectorized 27     # 128M floats
```

Each version prints `vector_add [<name>] PASS  count=<N>` on success.

## Adding a new version

1. Copy the closest existing version file (usually the previous one).
2. Rename to `NN_<short_name>.cu` where `NN` is the next number.
3. Change only the kernel, the launch, and the header comment describing
   the diff vs the previous version.
4. Keep the host boilerplate byte-identical to the other versions in this
   folder — that's what makes the diff readable.
5. Only add a new version if it carries a distinct teaching point or
   measurable perf improvement. Otherwise put the idea in a comment.

## Practice

`src/cuda/practice/01_vector_add/` mirrors this folder. Each practice
file has the kernel and launch bodies stubbed out with TODOs. The
skeletons compile but the produced binaries will fail the check until
you fill them in.
