#!/usr/bin/env python3
"""Generic ncu bench runner for a single chapter binary.

Usage:
    run_bench.py <binary> <kernel_regex> <out_prefix> <mode> [--n N]

<mode> selects the metric set:
    mem      memory-bound kernels (vector_add, reduce, softmax, layernorm)
    compute  compute-bound kernels (matmul, mlp, attention body)
    both     both sets

<out_prefix> is a path stem; we write <prefix>.csv, <prefix>.md, <prefix>.typ.

<--n N> passes N to a heuristic that computes an "effective bytes per launch"
for GB/s.  It's optional; if omitted we skip the GB/s column.
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import os
import re
import subprocess
import sys

NCU = os.environ.get("NCU", "/usr/local/cuda-13.0/bin/ncu")
HBM_PEAK_GBPS = 2039.0   # A100 80GB SXM4 (datasheet); PCIe variant is 1555
FP32_PEAK_TFLOPS = 19.5  # A100 FP32
TC_FP16_TFLOPS = 312.0   # A100 tensor core FP16

# Metrics we always capture (launch config + timing + divergence/conflict/stall).
# These let us evidence-driven talk about warp divergence, bank conflict, and
# barrier / memory stall in every chapter.
BASE_METRICS = [
    "gpu__time_duration.sum",
    "launch__grid_size",
    "launch__block_size",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_static",
    # Lane utilization: 32 = every lane participated in every issued instruction.
    # <32 means some lanes were predicated off (does not necessarily mean branch
    # divergence — predicated-off lanes still occupy the issue slot).
    "smsp__thread_inst_executed_per_inst_executed.ratio",
    "smsp__average_thread_inst_executed_pred_on_per_inst_executed.ratio",
    # Shared memory bank conflicts (0 = none; higher = worse).
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    # Stall reasons: how many warps stalled per issue-active cycle, broken by cause.
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
]

MEM_METRICS = BASE_METRICS + [
    "dram__bytes.sum",
    "dram__bytes.sum.pct_of_peak_sustained_elapsed",
    "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
]

COMPUTE_METRICS = BASE_METRICS + [
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes.sum.pct_of_peak_sustained_elapsed",
]


def run_ncu(binary: str, kernel_regex: str, metrics: list[str], out_csv: str,
            binary_args: list[str] | None = None) -> None:
    cmd = [
        "sudo", "-n", NCU,
        "-k", f"regex:{kernel_regex}",
        "--csv",
        "--log-file", out_csv,
        "--metrics", ",".join(metrics),
        binary,
    ]
    if binary_args:
        cmd += binary_args
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def parse_csv(path: str) -> tuple[list[str], dict]:
    raw = open(path).read().splitlines()
    for i, line in enumerate(raw):
        if line.startswith('"ID"'):
            raw = raw[i:]
            break
    reader = csv.DictReader(io.StringIO("\n".join(raw)))
    # Same kernel can launch many times (multi-stage reduction, etc.).
    # ncu emits one row per (launch, metric).  We keep the FIRST launch for
    # each kernel — that is the one operating on the largest input, and hence
    # the one whose metrics matter for the ladder comparison.
    by_kernel: dict[str, dict] = collections.defaultdict(dict)
    seen_launch: dict[str, str] = {}
    order: list[str] = []
    for r in reader:
        k = r["Kernel Name"]
        launch_id = r.get("ID", "")
        if k in seen_launch and seen_launch[k] != launch_id:
            # Different launch of the same kernel — skip (keep the first).
            continue
        if k not in by_kernel:
            order.append(k)
            seen_launch[k] = launch_id
            by_kernel[k]["_grid"] = r.get("Grid Size", "")
            by_kernel[k]["_block"] = r.get("Block Size", "")
        by_kernel[k][r["Metric Name"]] = (r["Metric Value"], r.get("Metric Unit", ""))
    return order, by_kernel


def short_name(k: str) -> str:
    """Turn '<unnamed>::foo_bar_kernel(...)' into 'foo-bar'."""
    m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)_kernel", k)
    if not m:
        return k.split("(")[0].split("::")[-1]
    name = m.group(1)
    # strip common prefixes
    for pref in ("vector_add_", "reduce_sum_", "softmax_", "matmul_", "layernorm_",
                 "mlp_", "attention_", "flash_attention_"):
        if name.startswith(pref):
            name = name[len(pref):]
    return name.replace("_", "-")


def num(v: str) -> float:
    return float(v.replace(",", ""))


def build_tables(order: list[str], by_kernel: dict, mode: str,
                 n_hint: int | None) -> tuple[str, str]:
    """Return (markdown, typst) strings.

    The typst output contains TWO tables:
      1. perf table:  time / throughput / occupancy (what we had before)
      2. diag table:  divergence / bank conflicts / stall  (evidence-driven)
    """
    have_time = lambda d: "gpu__time_duration.sum" in d

    md_lines = []
    if mode == "mem":
        headers = ["version", "regs", "smem", "time (μs)",
                   "HBM %", "SM %", "GB/s", "% peak",
                   "issued/32", "pred_on/32", "smem conf.", "barrier stall", "mem stall"]
    else:
        headers = ["version", "regs", "smem", "time (μs)",
                   "TC %", "warp %", "HBM %",
                   "issued/32", "pred_on/32", "smem conf.", "barrier stall", "mem stall"]
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    perf_rows: list[str] = []
    diag_rows: list[str] = []
    for k in order:
        d = by_kernel[k]
        if not have_time(d):
            continue
        time_ns = num(d["gpu__time_duration.sum"][0])
        time_us = time_ns / 1e3
        regs = d.get("launch__registers_per_thread", ("?", ""))[0]
        smem = d.get("launch__shared_mem_per_block_static", ("0", ""))[0]
        name = short_name(k)

        # -- diagnostic metrics: same for mem/compute mode --
        # issued/32: lanes that participated in each issued instruction (predication
        # included).  <32 means predication or branch divergence.
        issued_lanes = num(d.get(
            "smsp__thread_inst_executed_per_inst_executed.ratio", ("0", ""))[0])
        # pred_on/32: lanes that *actually did work* (predicated-on).
        # Big gap between issued and pred_on means predication is masking work.
        pred_on_lanes = num(d.get(
            "smsp__average_thread_inst_executed_pred_on_per_inst_executed.ratio",
            ("0", ""))[0])
        bc_ld = num(d.get(
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", ("0", ""))[0])
        bc_st = num(d.get(
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum", ("0", ""))[0])
        barrier_stall = num(d.get(
            "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
            ("0", ""))[0])
        mem_stall = num(d.get(
            "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
            ("0", ""))[0])
        bc_total = bc_ld + bc_st
        bc_str = f"{int(bc_total):,}"
        issued_str = f"{issued_lanes:.1f}"
        pred_on_str = f"{pred_on_lanes:.1f}"

        if mode == "mem":
            hbm_pct = num(d["dram__bytes.sum.pct_of_peak_sustained_elapsed"][0])
            sm_pct = num(d["sm__cycles_active.avg.pct_of_peak_sustained_elapsed"][0])
            dram_bytes = num(d["dram__bytes.sum"][0])
            gbps = dram_bytes / time_ns
            peak = gbps / HBM_PEAK_GBPS * 100
            gbps_s = f"{gbps:.0f}"
            peak_s = f"{peak:.1f}"
            if n_hint is not None:
                logical_gbps = n_hint / time_ns
                gbps_s = f"{gbps:.0f} / {logical_gbps:.0f}"
            md_lines.append(
                f"| {name} | {regs} | {smem} | {time_us:.2f} | "
                f"{hbm_pct:.1f} | {sm_pct:.1f} | {gbps_s} | {peak_s} | "
                f"{issued_str} | {pred_on_str} | {bc_str} | "
                f"{barrier_stall:.2f} | {mem_stall:.2f} |"
            )
            perf_rows.append(
                f"  [{name}], [{regs}], [{smem}], [{time_us:.2f}], "
                f"[{hbm_pct:.1f}], [{sm_pct:.1f}], [{gbps_s}], [{peak_s}],"
            )
        else:
            tc_pct = num(d.get(
                "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
                ("0", ""))[0])
            warp_pct = num(d.get(
                "sm__warps_active.avg.pct_of_peak_sustained_elapsed", ("0", ""))[0])
            hbm_pct = num(d.get(
                "dram__bytes.sum.pct_of_peak_sustained_elapsed", ("0", ""))[0])
            md_lines.append(
                f"| {name} | {regs} | {smem} | {time_us:.2f} | "
                f"{tc_pct:.1f} | {warp_pct:.1f} | {hbm_pct:.1f} | "
                f"{issued_str} | {pred_on_str} | {bc_str} | "
                f"{barrier_stall:.2f} | {mem_stall:.2f} |"
            )
            perf_rows.append(
                f"  [{name}], [{regs}], [{smem}], [{time_us:.2f}], "
                f"[{tc_pct:.1f}], [{warp_pct:.1f}], [{hbm_pct:.1f}],"
            )

        diag_rows.append(
            f"  [{name}], [{issued_str}], [{pred_on_str}], [{bc_str}], "
            f"[{barrier_stall:.2f}], [{mem_stall:.2f}],"
        )

    # ---- perf table ----
    if mode == "mem":
        perf_header = ("  [*version*], [*regs*], [*smem (B)*], [*time (μs)*], "
                       "[*HBM %*], [*SM %*], [*HBM GB/s*], [*% peak*],")
        perf_cols = "auto, auto, auto, auto, auto, auto, auto, auto"
        perf_caption = (
            "*Table (perf):* kernel-level performance metrics measured by "
            "Nsight Compute (`ncu`, CUDA 13.0) on a single A100 80GB SXM4 "
            "(HBM peak 2039 GB/s; FP32 peak 19.5 TFLOPS). "
            "Columns: *regs* = `launch__registers_per_thread`; "
            "*smem* = `launch__shared_mem_per_block_static` (bytes); "
            "*time* = `gpu__time_duration.sum` for the first kernel launch "
            "(multi-stage kernels report only the stage acting on the largest input); "
            "*HBM %* = `dram__bytes.sum.pct_of_peak_sustained_elapsed` "
            "(uses elapsed wall time as denominator, so short kernels look artificially low); "
            "*SM %* = `sm__cycles_active.avg.pct_of_peak_sustained_elapsed`; "
            "*HBM GB/s* is written `measured / logical`—`measured` = "
            "`dram__bytes.sum / time`, `logical` = `bytes-per-launch / time`; "
            "when the working set exceeds L2 (40 MB) these two should agree, "
            "and any gap quantifies L2 hit rate; *% peak* = "
            "`measured HBM GB/s ÷ 2039` GB/s. "
            "Rows are ordered as they appear on the optimization ladder in the chapter."
        )
    else:
        perf_header = ("  [*version*], [*regs*], [*smem (B)*], [*time (μs)*], "
                       "[*TC %*], [*warp %*], [*HBM %*],")
        perf_cols = "auto, auto, auto, auto, auto, auto, auto"
        perf_caption = (
            "*Table (perf):* kernel-level performance metrics measured by "
            "Nsight Compute (`ncu`, CUDA 13.0) on a single A100 80GB SXM4 "
            "(FP32 peak 19.5 TFLOPS; Tensor-Core FP16 peak 312 TFLOPS; HBM peak 2039 GB/s). "
            "Columns: *regs* = `launch__registers_per_thread`; "
            "*smem* = `launch__shared_mem_per_block_static` (bytes); "
            "*time* = `gpu__time_duration.sum`; "
            "*TC %* = `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` "
            "(0 means the kernel does not invoke tensor cores—`wmma::mma_sync` / MMA PTX); "
            "*warp %* = `sm__warps_active.avg.pct_of_peak_sustained_elapsed` "
            "(fraction of the SM's warp slots occupied on average); "
            "*HBM %* = `dram__bytes.sum.pct_of_peak_sustained_elapsed` "
            "(low when working set fits in L2). "
            "Rows are ordered as they appear on the optimization ladder in the chapter."
        )

    # NOTE: inside #figure(...) we are in code mode, so `table(...)` must NOT
    # have a leading `#`.  Same for `diag_tab_body` below.
    perf_tab_body = "\n".join([
        "table(",
        f"  columns: ({perf_cols}),",
        "  stroke: 0.5pt + gray, inset: 5pt,",
        "  align: (left, right, right, right, right, right, right, right),",
        perf_header,
        *perf_rows,
        ")",
    ])
    perf_tab = "\n".join([
        "// Auto-generated perf table.  Hardware: A100 80GB SXM4.",
        "#figure(",
        perf_tab_body + ",",
        f"  caption: [{perf_caption}],",
        "  kind: table,",
        ") <perf-table>",
    ])

    # ---- diagnostic table ----
    diag_caption = (
        "*Table (diag):* diagnostic metrics used as evidence for "
        "warp-lane utilization, shared-memory bank conflict, and warp-stall "
        "claims in the chapter. All from `ncu` on the same launch as the "
        "perf table. *issued/32* = "
        "`smsp__thread_inst_executed_per_inst_executed.ratio`, the average "
        "number of lanes that participated in each issued warp instruction "
        "(32 = every lane active; less than 32 means predication *or* branch "
        "divergence—`ncu` alone cannot distinguish, but on the CUDA kernels "
        "in this book it is almost always predication). "
        "*pred_on/32* = "
        "`smsp__average_thread_inst_executed_pred_on_per_inst_executed.ratio`, "
        "the average number of lanes that were predicated-on (did real work). "
        "The *gap* `issued − pred_on` counts lanes that occupied the issue "
        "slot but performed no work—this is the correct definition of "
        "\"wasted warp cycles\", not divergence. *smem conf.* = "
        "`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum + "
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` (0 means "
        "no measurable bank conflicts, either because the access pattern is "
        "conflict-free or because the input is too small to accumulate them). "
        "*barrier stall* = "
        "`smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio`, "
        "warps stalled at `__syncthreads` per issue-active cycle. "
        "*mem stall* = "
        "`smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio`, "
        "warps waiting on a long-latency memory operation (global-memory load, "
        "L2 atomic response) per issue-active cycle."
    )
    diag_tab_body = "\n".join([
        "table(",
        "  columns: (auto, auto, auto, auto, auto, auto),",
        "  stroke: 0.5pt + gray, inset: 5pt,",
        "  align: (left, right, right, right, right, right),",
        "  [*version*], [*issued/32*], [*pred_on/32*], [*smem conf.*], "
        "[*barrier stall*], [*mem stall*],",
        *diag_rows,
        ")",
    ])
    diag_tab = "\n".join([
        "// Auto-generated diagnostic table.",
        "#figure(",
        diag_tab_body + ",",
        f"  caption: [{diag_caption}],",
        "  kind: table,",
        ") <diag-table>",
    ])

    typ_out = perf_tab + "\n\n" + diag_tab + "\n"
    return "\n".join(md_lines) + "\n", typ_out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("kernel_regex")
    ap.add_argument("out_prefix")
    ap.add_argument("mode", choices=["mem", "compute"])
    ap.add_argument("--bytes-per-launch", type=int, default=None,
                    help="bytes moved per launch; used to compute GB/s")
    ap.add_argument("--binary-args", default="",
                    help="extra args to pass to the CUDA binary, space-separated")
    args = ap.parse_args()

    metrics = MEM_METRICS if args.mode == "mem" else COMPUTE_METRICS
    csv_path = args.out_prefix + ".csv"
    md_path = args.out_prefix + ".md"
    typ_path = args.out_prefix + ".typ"

    binary_args = args.binary_args.split() if args.binary_args else None
    run_ncu(args.binary, args.kernel_regex, metrics, csv_path, binary_args=binary_args)
    order, by_kernel = parse_csv(csv_path)
    if not order:
        print(f"no kernels matched regex {args.kernel_regex!r} in {args.binary}",
              file=sys.stderr)
        return 1

    md, typ = build_tables(order, by_kernel, args.mode, args.bytes_per_launch)
    open(md_path, "w").write(md)
    open(typ_path, "w").write(typ)
    print(md)
    print(f"wrote {md_path} and {typ_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
