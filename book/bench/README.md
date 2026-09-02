# Benchmarks

For each chapter, we run every kernel variant and capture two things:

1. **Wall-time & memory throughput** via a small "bench" binary (or just re-run the app once per variant) — computed from problem size and CUDA-event elapsed time.
2. **Nsight Compute metrics** via `ncu`, for a curated set of metrics per chapter.

`sudo` is required on this host because the driver has `RmProfilingAdminOnly=1`.

## Layout

```
bench/
  README.md          (this file)
  common.sh          (helpers: sudo ncu wrapper, csv extract)
  01_vector_add.sh   (per-chapter bench script)
  01_vector_add.csv  (raw metric dump)
  01_vector_add.md   (human-readable summary, embedded into typst)
  ...
```

## Metric picks (rationale)

For memory-bound kernels (vector_add, reduce, softmax, layernorm):

- `gpu__time_duration.sum`   — kernel wall time
- `dram__bytes.sum.pct_of_peak_sustained_elapsed` — % of HBM peak
- `sm__cycles_active.avg.pct_of_peak_sustained_elapsed` — SM utilization
- `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second` — L1 global load BW

For compute-bound kernels (matmul, MLP):

- `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` — tensor pipe util
- `smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed` — FMA rate
- `sm__warps_active.avg.pct_of_peak_sustained_elapsed` — occupancy

For reduction / warp-heavy kernels:

- `smsp__inst_executed_op_shared_st.sum` / `_op_shared_ld.sum` — smem traffic
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` — bank conflicts
- `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` — sync barrier stall
