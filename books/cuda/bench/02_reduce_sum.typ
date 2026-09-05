// Auto-generated perf table.  Hardware: A100 80GB SXM4.
#figure(
table(
  columns: (auto, auto, auto, auto, auto, auto, auto, auto),
  stroke: 0.5pt + gray, inset: 5pt,
  align: (left, right, right, right, right, right, right, right),
  [*version*], [*regs*], [*smem (B)*], [*time (μs)*], [*HBM %*], [*SM %*], [*HBM GB/s*], [*% peak*],
  [atomic], [16], [0], [367782.40], [0.1], [100.0], [1 / 1], [0.1],
  [interleaved], [16], [0], [2622.43], [10.1], [99.9], [206 / 205], [10.1],
  [sequential], [16], [0], [1444.19], [18.3], [99.7], [373 / 372], [18.3],
  [warp-shuffle], [16], [32], [1524.13], [17.4], [99.3], [354 / 352], [17.4],
  [chunked], [19], [0], [287.39], [92.0], [98.8], [1876 / 1868], [92.0],
),
  caption: [*Table (perf):* kernel-level performance metrics measured by Nsight Compute (`ncu`, CUDA 13.0) on a single A100 80GB SXM4 (HBM peak 2039 GB/s; FP32 peak 19.5 TFLOPS). Columns: *regs* = `launch__registers_per_thread`; *smem* = `launch__shared_mem_per_block_static` (bytes); *time* = `gpu__time_duration.sum` for the first kernel launch (multi-stage kernels report only the stage acting on the largest input); *HBM %* = `dram__bytes.sum.pct_of_peak_sustained_elapsed` (uses elapsed wall time as denominator, so short kernels look artificially low); *SM %* = `sm__cycles_active.avg.pct_of_peak_sustained_elapsed`; *HBM GB/s* is written `measured / logical`—`measured` = `dram__bytes.sum / time`, `logical` = `bytes-per-launch / time`; when the working set exceeds L2 (40 MB) these two should agree, and any gap quantifies L2 hit rate; *% peak* = `measured HBM GB/s ÷ 2039` GB/s. Rows are ordered as they appear on the optimization ladder in the chapter.],
  kind: table,
) <perf-table>

// Auto-generated diagnostic table.
#figure(
table(
  columns: (auto, auto, auto, auto, auto, auto),
  stroke: 0.5pt + gray, inset: 5pt,
  align: (left, right, right, right, right, right),
  [*version*], [*issued/32*], [*pred_on/32*], [*smem conf.*], [*barrier stall*], [*mem stall*],
  [atomic], [32.0], [28.6], [0], [0.00], [6045.10],
  [interleaved], [31.9], [24.0], [424,592], [2.80], [2.11],
  [sequential], [31.8], [20.0], [296,900], [4.33], [9.58],
  [warp-shuffle], [31.5], [29.3], [40,044], [2.99], [62.60],
  [chunked], [31.9], [23.7], [164,836], [2.62], [16.00],
),
  caption: [*Table (diag):* diagnostic metrics used as evidence for warp-lane utilization, shared-memory bank conflict, and warp-stall claims in the chapter. All from `ncu` on the same launch as the perf table. *issued/32* = `smsp__thread_inst_executed_per_inst_executed.ratio`, the average number of lanes that participated in each issued warp instruction (32 = every lane active; less than 32 means predication *or* branch divergence—`ncu` alone cannot distinguish, but on the CUDA kernels in this book it is almost always predication). *pred_on/32* = `smsp__average_thread_inst_executed_pred_on_per_inst_executed.ratio`, the average number of lanes that were predicated-on (did real work). The *gap* `issued − pred_on` counts lanes that occupied the issue slot but performed no work—this is the correct definition of "wasted warp cycles", not divergence. *smem conf.* = `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum + l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` (0 means no measurable bank conflicts, either because the access pattern is conflict-free or because the input is too small to accumulate them). *barrier stall* = `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio`, warps stalled at `__syncthreads` per issue-active cycle. *mem stall* = `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio`, warps waiting on a long-latency memory operation (global-memory load, L2 atomic response) per issue-active cycle.],
  kind: table,
) <diag-table>
