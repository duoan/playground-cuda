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
| `src/cuda/04_matmul.cu` | teaching GEMM kernel with `naive -> shared-memory tiled` progression |
| `src/cuda/common/cuda_utils.cuh` | shared helper utilities such as `CHECK_CUDA(...)` |

## Native CUDA

```bash
make
make run APP=00_device_query
make run APP=01_vector_add
make run APP=02_reduce_sum
make run APP=04_matmul
```

Build output goes to `build/`.

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
