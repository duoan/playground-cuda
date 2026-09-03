// Auto-generated perf table.  Hardware: A100 80GB SXM4.
#figure(
table(
  columns: (auto, auto, auto, auto, auto, auto, auto),
  stroke: 0.5pt + gray, inset: 5pt,
  align: (left, right, right, right, right, right, right, right),
  [*version*], [*regs*], [*smem (B)*], [*time (μs)*], [*TC %*], [*warp %*], [*HBM %*],
  [naive], [32], [0], [8.35], [0.0], [1.4], [0.2],
  [tiled], [32], [2,048], [5.95], [0.0], [1.3], [0.3],
  [warp-tiled], [30], [4,608], [13.41], [0.0], [0.8], [0.1],
  [register-blocked], [40], [4,096], [7.55], [0.0], [0.3], [0.2],
  [pipeline-teaching], [40], [8,192], [7.14], [0.0], [0.3], [0.3],
),
  caption: [*Table (perf):* kernel-level performance metrics measured by Nsight Compute (`ncu`, CUDA 13.0) on a single A100 80GB SXM4 (FP32 peak 19.5 TFLOPS; Tensor-Core FP16 peak 312 TFLOPS; HBM peak 2039 GB/s). Columns: *regs* = `launch__registers_per_thread`; *smem* = `launch__shared_mem_per_block_static` (bytes); *time* = `gpu__time_duration.sum`; *TC %* = `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` (0 means the kernel does not invoke tensor cores—`wmma::mma_sync` / MMA PTX); *warp %* = `sm__warps_active.avg.pct_of_peak_sustained_elapsed` (fraction of the SM's warp slots occupied on average); *HBM %* = `dram__bytes.sum.pct_of_peak_sustained_elapsed` (low when working set fits in L2). Rows are ordered as they appear on the optimization ladder in the chapter.],
  kind: table,
) <perf-table>

// Auto-generated diagnostic table.
#figure(
table(
  columns: (auto, auto, auto, auto, auto, auto),
  stroke: 0.5pt + gray, inset: 5pt,
  align: (left, right, right, right, right, right),
  [*version*], [*issued/32*], [*pred_on/32*], [*smem conf.*], [*barrier stall*], [*mem stall*],
  [naive], [32.0], [31.4], [0], [0.00], [13.75],
  [tiled], [32.0], [31.6], [0], [1.05], [7.41],
  [warp-tiled], [32.0], [31.2], [0], [1.46], [10.40],
  [register-blocked], [32.0], [31.8], [256], [0.53], [4.18],
  [pipeline-teaching], [32.0], [31.8], [256], [0.44], [4.38],
),
  caption: [*Table (diag):* diagnostic metrics used as evidence for warp-lane utilization, shared-memory bank conflict, and warp-stall claims in the chapter. All from `ncu` on the same launch as the perf table. *issued/32* = `smsp__thread_inst_executed_per_inst_executed.ratio`, the average number of lanes that participated in each issued warp instruction (32 = every lane active; less than 32 means predication *or* branch divergence—`ncu` alone cannot distinguish, but on the CUDA kernels in this book it is almost always predication). *pred_on/32* = `smsp__average_thread_inst_executed_pred_on_per_inst_executed.ratio`, the average number of lanes that were predicated-on (did real work). The *gap* `issued − pred_on` counts lanes that occupied the issue slot but performed no work—this is the correct definition of "wasted warp cycles", not divergence. *smem conf.* = `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum + l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` (0 means no measurable bank conflicts, either because the access pattern is conflict-free or because the input is too small to accumulate them). *barrier stall* = `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio`, warps stalled at `__syncthreads` per issue-active cycle. *mem stall* = `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio`, warps waiting on a long-latency memory operation (global-memory load, L2 atomic response) per issue-active cycle.],
  kind: table,
) <diag-table>
