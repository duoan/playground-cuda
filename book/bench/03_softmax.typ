// Auto-generated perf table.  Hardware: A100 80GB SXM4.
#figure(
table(
  columns: (auto, auto, auto, auto, auto, auto, auto, auto),
  stroke: 0.5pt + gray, inset: 5pt,
  align: (left, right, right, right, right, right, right, right),
  [*version*], [*regs*], [*smem (B)*], [*time (μs)*], [*HBM %*], [*SM %*], [*HBM GB/s*], [*% peak*],
  [naive], [35], [0], [121.02], [0.0], [0.9], [1 / 104], [0.0],
  [block], [20], [2,048], [5.63], [0.6], [36.9], [13 / 2234], [0.6],
  [online], [32], [0], [252.80], [0.0], [0.9], [0 / 50], [0.0],
  [masked], [37], [0], [435.84], [0.0], [0.9], [0 / 29], [0.0],
  [causal], [36], [0], [46.75], [0.0], [0.9], [0 / 269], [0.0],
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
  [naive], [32.0], [32.0], [0], [0.00], [1.26],
  [block], [31.1], [21.4], [0], [2.44], [2.13],
  [online], [32.0], [32.0], [0], [0.00], [4.77],
  [masked], [26.3], [25.9], [0], [0.00], [9.82],
  [causal], [22.7], [22.4], [0], [0.00], [1.68],
),
  caption: [*Table (diag):* diagnostic metrics used as evidence for warp-lane utilization, shared-memory bank conflict, and warp-stall claims in the chapter. All from `ncu` on the same launch as the perf table. *issued/32* = `smsp__thread_inst_executed_per_inst_executed.ratio`, the average number of lanes that participated in each issued warp instruction (32 = every lane active; less than 32 means predication *or* branch divergence—`ncu` alone cannot distinguish, but on the CUDA kernels in this book it is almost always predication). *pred_on/32* = `smsp__average_thread_inst_executed_pred_on_per_inst_executed.ratio`, the average number of lanes that were predicated-on (did real work). The *gap* `issued − pred_on` counts lanes that occupied the issue slot but performed no work—this is the correct definition of "wasted warp cycles", not divergence. *smem conf.* = `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum + l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` (0 means no measurable bank conflicts, either because the access pattern is conflict-free or because the input is too small to accumulate them). *barrier stall* = `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio`, warps stalled at `__syncthreads` per issue-active cycle. *mem stall* = `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio`, warps waiting on a long-latency memory operation (global-memory load, L2 atomic response) per issue-active cycle.],
  kind: table,
) <diag-table>
