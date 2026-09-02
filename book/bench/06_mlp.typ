// Auto-generated perf table.  Hardware: A100 80GB SXM4.
#figure(
table(
  columns: (auto, auto, auto, auto, auto, auto, auto),
  stroke: 0.5pt + gray, inset: 5pt,
  align: (left, right, right, right, right, right, right, right),
  [*version*], [*regs*], [*smem (B)*], [*time (μs)*], [*TC %*], [*warp %*], [*HBM %*],
  [naive-linear1], [27], [0], [3.14], [0.0], [0.0], [0.1],
  [naive-linear2], [31], [0], [3.55], [0.0], [0.0], [0.1],
  [fused-linear1-relu], [27], [0], [3.10], [0.0], [0.0], [0.1],
  [linear2], [31], [0], [3.55], [0.0], [0.0], [0.1],
  [tiled-fused], [30], [80], [4.51], [0.0], [0.0], [0.1],
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
  [naive-linear1], [16.0], [15.6], [0], [0.00], [8.39],
  [naive-linear2], [4.0], [3.9], [0], [0.00], [10.54],
  [fused-linear1-relu], [16.0], [15.6], [0], [0.00], [8.63],
  [linear2], [4.0], [3.9], [0], [0.00], [10.65],
  [tiled-fused], [18.3], [12.6], [0], [0.76], [16.69],
),
  caption: [*Table (diag):* diagnostic metrics used as evidence for warp-lane utilization, shared-memory bank conflict, and warp-stall claims in the chapter. All from `ncu` on the same launch as the perf table. *issued/32* = `smsp__thread_inst_executed_per_inst_executed.ratio`, the average number of lanes that participated in each issued warp instruction (32 = every lane active; less than 32 means predication *or* branch divergence—`ncu` alone cannot distinguish, but on the CUDA kernels in this book it is almost always predication). *pred_on/32* = `smsp__average_thread_inst_executed_pred_on_per_inst_executed.ratio`, the average number of lanes that were predicated-on (did real work). The *gap* `issued − pred_on` counts lanes that occupied the issue slot but performed no work—this is the correct definition of "wasted warp cycles", not divergence. *smem conf.* = `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum + l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` (0 means no measurable bank conflicts, either because the access pattern is conflict-free or because the input is too small to accumulate them). *barrier stall* = `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio`, warps stalled at `__syncthreads` per issue-active cycle. *mem stall* = `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio`, warps waiting on a long-latency memory operation (global-memory load, L2 atomic response) per issue-active cycle.],
  kind: table,
) <diag-table>
