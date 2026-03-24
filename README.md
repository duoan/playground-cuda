# playground-cuda

A CUDA playground for learning native CUDA, plus a separate Python area for Triton and PyTorch experiments.

## Layout

- `src/cuda/`: native CUDA teaching kernels, organized by learning order
- `src/cuda/common/`: shared CUDA helpers such as error checking
- `src/triton/`: future Triton learning kernels and notes
- `src/cutile/`: future CuTile DSL learning kernels and notes
- `python/playground_cuda/`: existing Python package for PyTorch experiments
- `docs/`: notes, writeups, and profiling references
- `reports/`: captured experiment outputs
- `traces/`: profiler traces

## Learning Roadmap

Use the same operator progression across `CUDA -> Triton -> CuTile DSL`, so we keep learning one idea in three forms instead of learning three disconnected stacks.

## Teaching Principles

For each operator, follow one consistent rule:

- Keep the full learning path in one file for that operator.
- Start from the most direct baseline implementation.
- Add optimized versions step by step in the same file.
- Explain what each optimization changes in the execution model.
- Keep a CPU reference and correctness check in the same file.
- Add a separate practice file with key functions left as `TODO`.

For CUDA kernels, the target progression is usually:

1. `naive`
2. `shared-memory tiled`
3. `warp-aware`
4. `register-blocked`
5. `tensor-core / pipeline-oriented`

Important note:

- The final stage should move toward modern high-performance structure.
- But we only claim “library-level performance” after real benchmarking against reference libraries such as cuBLAS, CUTLASS, or FlashAttention implementations.

| Stage | Topic                  | Core Operators                                                     | CUDA Focus                                               | Triton Focus                                                | CuTile DSL Focus                                      |
| ----- | ---------------------- | ------------------------------------------------------------------ | -------------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------- |
| 0     | Execution model basics | `vector_add`, `relu`, `broadcast add`                              | thread / block / grid, indexing, memory hierarchy        | program model, program ids, block-level elementwise mapping | Python DSL basics, tiling vocabulary                  |
| 1     | Basic reductions       | `reduce_sum`, `reduce_max`                                         | shared memory reduction, warp reduction, synchronization | block reductions, reduction axes                            | reduction structure and tile partitioning             |
| 2     | Softmax                | `softmax`, `masked softmax`, `causal softmax`                      | stable softmax, row reduction, online softmax            | row-wise kernels, fusion-friendly reductions                | tiled reduction structure for attention-style kernels |
| 3     | GEMM                   | `matmul`, `sgemm`                                                  | naive matmul, shared-memory tiling, register blocking    | block matmul, pointer arithmetic, autotuned tile sizes      | tiled matmul structure and pipeline ideas             |
| 4     | Norm and MLP           | `layernorm`, `rmsnorm`, `mlp`, `swiglu`                            | fused reductions, activation fusion, matmul pipelines    | fused kernels, row/block mapping                            | Python DSL fusion structure                           |
| 5     | Attention              | `attention`, `qk^t`, `pv`                                          | decomposition into score, softmax, value projection      | block-wise attention kernels                                | q/k/v tile movement and fused attention structure     |
| 6     | FlashAttention         | `flash_attention_v1`, `flash_attention_v2`, `flash_attention_v3`   | online softmax, SRAM reuse, work partition, pipeline     | fused attention kernels                                     | advanced tiled attention pipelines                    |
| 7     | Advanced fused ops     | `bias+gelu`, `dropout+residual`, `rope`, `topk`, `paged attention` | fusion, bandwidth optimization, specialized kernels      | fusion patterns, persistent-style kernels                   | higher-level fused DSL kernels                        |

## Suggested Order

| Order | Operator             | Why It Comes Here                                                        |
| ----- | -------------------- | ------------------------------------------------------------------------ |
| 1     | `vector_add`         | learn indexing, launch configuration, and the basic CUDA execution model |
| 2     | `reduce_sum`         | learn synchronization, shared memory, and warp-level thinking            |
| 3     | `softmax`            | combine reduction, numerical stability, and row-wise parallelism         |
| 4     | `matmul`             | learn tiling, data reuse, and the foundation of deep learning kernels    |
| 5     | `layernorm`          | practice reduction + normalization + fusion                              |
| 6     | `mlp`                | combine GEMM and activation patterns into a common DL block              |
| 7     | `attention`          | connect GEMM-like work, softmax, and fusion                              |
| 8     | `flash_attention_v1` | learn online softmax and SRAM-aware tiling                               |
| 9     | `flash_attention_v2` | learn better work partition and parallelism                              |
| 10    | `flash_attention_v3` | study modern high-performance pipeline ideas                             |

## File Plan

| Track | Planned Files |
| --- | --- |
| CUDA | `src/cuda/01_vector_add.cu`, `02_reduce_sum.cu`, `03_softmax.cu`, `04_matmul.cu`, `05_layernorm.cu`, `06_mlp.cu`, `07_attention.cu`, `08_flash_attention_v1.cu`, `09_flash_attention_v2.cu`, `10_flash_attention_v3.cu` |
| Triton | `src/triton/01_vector_add.py`, `02_reduce_sum.py`, `03_softmax.py`, `04_matmul.py`, `05_layernorm.py`, `06_mlp.py`, `07_attention.py`, `08_flash_attention_v1.py`, `09_flash_attention_v2.py`, `10_flash_attention_v3.py` |
| CuTile DSL | `src/cutile/01_vector_add.py`, `02_reduce_sum.py`, `03_softmax.py`, `04_matmul.py`, `05_layernorm.py`, `06_mlp.py`, `07_attention.py`, `08_flash_attention_v1.py`, `09_flash_attention_v2.py`, `10_flash_attention_v3.py` |

## Current CUDA Files

| File | Purpose |
| --- | --- |
| `src/cuda/00_device_query.cu` | inspect the current GPU and CUDA runtime environment |
| `src/cuda/01_vector_add.cu` | first teaching kernel with `naive -> tiled` progression |
| `src/cuda/02_reduce_sum.cu` | reduction teaching kernel with `atomic -> shared memory -> warp-aware` progression |
| `src/cuda/03_softmax.cu` | row-wise softmax teaching kernel with `naive -> block/shared-memory` progression |
| `src/cuda/04_matmul.cu` | teaching GEMM kernel with `naive -> shared-memory tiled` progression |
| `src/cuda/05_layernorm.cu` | row-wise layernorm teaching kernel with `naive -> block/shared-memory` progression |
| `src/cuda/06_mlp.cu` | small two-layer MLP teaching kernel with `naive -> fused first-stage` progression |
| `src/cuda/07_attention.cu` | single-head self-attention teaching kernel with `naive -> more structured tiled` progression |
| `src/cuda/08_flash_attention_v1.cu` | simplified FlashAttention v1 teaching kernel focused on online softmax |
| `src/cuda/09_flash_attention_v2.cu` | simplified FlashAttention v2 teaching kernel with shared-memory tile staging |
| `src/cuda/10_flash_attention_v3.cu` | simplified FlashAttention v3 teaching kernel with warp-specialized roles |
| `src/cuda/common/cuda_utils.cuh` | shared helper utilities such as `CHECK_CUDA(...)` |
| `src/cuda/practice/` | exercise files with `TODO` kernels left for you to implement |

## Optimization Ladder Status

| File | Current Stages | Next Target |
| --- | --- | --- |
| `01_vector_add.cu` | `naive -> block-tiled -> grid-stride -> vectorized` | `wider vectorized access / memory-oriented tuning` |
| `02_reduce_sum.cu` | `atomic -> shared-memory -> warp-aware -> hierarchical chunked reduction` | `more aggressive hierarchical / shuffle-heavy reduction` |
| `03_softmax.cu` | `CPU -> naive -> block/shared-memory -> online -> masked -> causal` | `FlashAttention-ready fused softmax variants` |
| `04_matmul.cu` | `naive -> shared-memory tiled -> warp/register tiled -> register-blocked -> pipeline-oriented` | `benchmark and compare against real reference libraries` |
| `05_layernorm.cu` | `CPU -> naive -> block/shared-memory -> warp-aware / fused` | `RMSNorm sibling and stronger row-wise fusion` |
| `06_mlp.cu` | `CPU -> naive -> partially fused -> tiled GEMM-style fused` | `larger tile shapes and stronger pipeline overlap` |
| `07_attention.cu` | `CPU -> naive decomposition -> tiled/online-softmax -> causal support` | `FlashAttention-style next hand-off` |
| `08_flash_attention_v1.cu` | `CPU/reference -> simplified v1 -> shared-memory staged` | `stronger block-level tiling and SRAM reuse` |
| `09_flash_attention_v2.cu` | `CPU/reference -> simplified v2 -> warp-specialized` | `more aggressive work partition and pipeline staging` |
| `10_flash_attention_v3.cu` | `CPU/reference -> simplified v3 -> pipeline-oriented` | `tensor-core / Hopper-style evolution` |

## Next-Level Plans

### `01_vector_add.cu`

- Add a `grid-stride loop` version so one kernel can cover arbitrarily large vectors in a standard CUDA style.
- Add a `vectorized load/store` version using `float2` or `float4` so you can see what “wider memory transactions” look like in simple code.
- Compare the mental model:
  `one thread -> one element`
  then
  `one thread -> many elements over stride`
  then
  `one thread -> many elements with vectorized memory access`

### `02_reduce_sum.cu`

- Add a `multi-elements-per-thread` version so each thread accumulates several values before entering the block reduction.
- Add a more explicit `hierarchical reduction` version that shows:
  input -> block partial sums -> second-stage reduction -> final sum.
- Add notes on why reduction performance usually depends on:
  memory bandwidth, synchronization cost, and how much work each thread does before communicating.

### `03_softmax.cu`

- Add an `online softmax` version to connect this file directly to FlashAttention.
- Add `masked softmax` and `causal softmax` variants so the file starts to reflect transformer workloads.
- Make the progression explicit:
  stable row softmax
  then
  block-cooperative row softmax
  then
  online / masked / causal softmax.

### `04_matmul.cu`

- Add a `tensor-core / pipeline-oriented` teaching version.
- Explain the next concepts in order:
  `mma-friendly tile shapes`, `staging`, `double buffering`, and `pipeline overlap`.
- Add a benchmark section later if you want to honestly compare:
  naive vs tiled vs warp/register vs tensor-core-oriented.

### `05_layernorm.cu`

- Add a `warp-aware row layernorm` version for smaller row widths.
- Add a more fused version that keeps intermediate row statistics in registers/shared memory and reduces unnecessary traffic.
- Optionally add `RMSNorm` afterward, because it is a simpler relative and very common in modern models.

### `06_mlp.cu`

- Add a more `GEMM-like tiled MLP` version where the linear layers start borrowing ideas from the matmul file.
- Add a stronger `epilogue fusion` version such as:
  linear + bias + activation
  or
  gated activation style.
- After that, split the file into conceptual stages:
  baseline MLP
  partially fused MLP
  tiled/fused MLP.

### `07_attention.cu`

- Add a more explicit `tile-by-tile attention` version that makes the transition into FlashAttention feel natural.
- Make the decomposition clearer:
  score computation
  softmax update
  value accumulation
  all inside a structured tiled loop.
- Add optional `causal` support later so it matches autoregressive transformer behavior.

### `08_flash_attention_v1.cu`

- Add stronger `SRAM reuse` and more explicit block-level tile staging.
- Make the `online max / online sum / output rescaling` steps more visible in comments and output summaries.
- Add a comparison note showing exactly what v1 removes relative to naive attention:
  materialized score matrix and large intermediate memory traffic.

### `09_flash_attention_v2.cu`

- Add clearer `warp specialization` and block work partition.
- Show how v2 improves the execution plan over v1 without changing the math.
- Add a compact summary block in code/comments:
  what data each thread group loads, computes, and writes.

### `10_flash_attention_v3.cu`

- Add a more explicit `pipeline-oriented` teaching layer that prepares you for tensor-core-era kernels.
- Explain what changes in emphasis at this stage:
  less about the equation, more about pipeline structure and role specialization.
- Later, if you want, add a “conceptual Hopper path” note for:
  warp groups, async staging, and tensor-core-friendly execution.

## Native CUDA

```bash
make
make run APP=00_device_query
make run APP=01_vector_add
make run APP=02_reduce_sum
make run APP=03_softmax
make run APP=04_matmul
make run APP=05_layernorm
make run APP=06_mlp
make run APP=07_attention
make run APP=08_flash_attention_v1
make run APP=09_flash_attention_v2
make run APP=10_flash_attention_v3
make practice
```

Build output goes to `build/`.

Practice builds go to `build/practice/`.

## Tomorrow Study Path

For reading:

1. `src/cuda/01_vector_add.cu`
2. `src/cuda/02_reduce_sum.cu`
3. `src/cuda/03_softmax.cu`
4. `src/cuda/04_matmul.cu`
5. `src/cuda/05_layernorm.cu`
6. `src/cuda/06_mlp.cu`
7. `src/cuda/07_attention.cu`
8. `src/cuda/08_flash_attention_v1.cu`
9. `src/cuda/09_flash_attention_v2.cu`
10. `src/cuda/10_flash_attention_v3.cu`

For hands-on practice:

1. `src/cuda/practice/01_vector_add_practice.cu`
2. `src/cuda/practice/02_reduce_sum_practice.cu`
3. `src/cuda/practice/03_softmax_practice.cu`
4. `src/cuda/practice/04_matmul_practice.cu`
5. `src/cuda/practice/05_layernorm_practice.cu`
6. `src/cuda/practice/06_mlp_practice.cu`
7. `src/cuda/practice/07_attention_practice.cu`

## Review Slides

Open the slide index in a browser:

- `docs/slides/index.html`
- `docs/slides/next_levels.html`

Each kernel also has its own HTML deck:

- `docs/slides/00_device_query.html`
- `docs/slides/01_vector_add.html`
- `docs/slides/02_reduce_sum.html`
- `docs/slides/03_softmax.html`
- `docs/slides/04_matmul.html`
- `docs/slides/05_layernorm.html`
- `docs/slides/06_mlp.html`
- `docs/slides/07_attention.html`
- `docs/slides/08_flash_attention_v1.html`
- `docs/slides/09_flash_attention_v2.html`
- `docs/slides/10_flash_attention_v3.html`

Use left/right arrow keys to move like a PPT.

## Python

```bash
uv sync
uv run python -m playground_cuda.device_query
uv run python -m playground_cuda.vector_add
uv run python -m playground_cuda.training_lab --mode baseline --steps 20
```

## Notes

- The native CUDA examples in `src/cuda/` are plain CUDA Runtime API code and do not depend on `libtorch`.
- Set `CUDA_HOME` or `NVCC` if your CUDA toolkit is not installed under `/usr/local/cuda`.
- Use `uv sync` after pulling changes so the local `.venv/` stays in sync with `pyproject.toml` and `uv.lock`.
