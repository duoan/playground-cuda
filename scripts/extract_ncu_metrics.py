#!/usr/bin/env python3
"""Extract a compact set of ncu metrics from src/cuda/*/ncu/*.txt logs.

Each ncu log holds one or more kernel invocations, each with the full
`--set detailed` metric dump.  For every log we pull a fixed set of numbers
that we actually cite in the book:

    Duration               (us)
    Elapsed Cycles         (cycle)
    Memory Throughput      (%)          # SOL memory
    DRAM Throughput        (%)          # SOL DRAM
    Compute (SM) Throughput(%)          # SOL compute
    L1/TEX Hit Rate        (%)
    L2 Hit Rate            (%)
    Issue Slots Busy       (%)
    Achieved Occupancy     (%)
    Registers Per Thread   (register/thread)
    Static SMEM Per Block  (byte/block)
    Block Size
    Grid Size

Output: one JSON file per log, at src/cuda/<chapter>/ncu/<log>.json,
plus a chapter-level aggregate at src/cuda/<chapter>/ncu/summary.json.

Usage:
    python3 scripts/extract_ncu_metrics.py           # extract everything
    python3 scripts/extract_ncu_metrics.py --only 04_matmul
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Metrics we want.  Keyed by our short name.  Value = list of (regex, unit
# preference).  Regex captures the numeric value.  Some metrics have multiple
# rows across sections (e.g. Memory Throughput appears in SOL and in Memory
# Workload Analysis with different units); we take the *first* occurrence,
# which is the SOL section.
#
# Metric-name lines in ncu look like:
#     Memory Throughput           %          94.98
# Numbers use `,` as thousands separator.
METRIC_PATTERNS: dict[str, str] = {
    # SOL Throughput block
    "duration_us":         r"Duration\s+us\s+([\d.,]+)",
    "duration_ms":         r"Duration\s+ms\s+([\d.,]+)",
    "duration_ns":         r"Duration\s+ns\s+([\d.,]+)",
    "elapsed_cycles":      r"Elapsed Cycles\s+cycle\s+([\d.,]+)",
    "sm_active_cycles":    r"SM Active Cycles\s+cycle\s+([\d.,]+)",
    "sol_memory_pct":      r"Memory Throughput\s+%\s+([\d.,]+)",
    "sol_dram_pct":        r"DRAM Throughput\s+%\s+([\d.,]+)",
    "sol_compute_pct":     r"Compute \(SM\) Throughput\s+%\s+([\d.,]+)",
    "sol_l1_pct":          r"L1/TEX Cache Throughput\s+%\s+([\d.,]+)",
    "sol_l2_pct":          r"L2 Cache Throughput\s+%\s+([\d.,]+)",
    # Memory Workload Analysis block
    "l1_hit_pct":          r"L1/TEX Hit Rate\s+%\s+([\d.,]+)",
    "l2_hit_pct":          r"L2 Hit Rate\s+%\s+([\d.,]+)",
    "mem_throughput_gbs":  r"Memory Throughput\s+Gbyte/s\s+([\d.,]+)",
    "mem_max_bw_pct":      r"Max Bandwidth\s+%\s+([\d.,]+)",
    # Compute Workload Analysis block
    "ipc_active":          r"Executed Ipc Active\s+inst/cycle\s+([\d.,]+)",
    "issue_slots_busy":    r"Issue Slots Busy\s+%\s+([\d.,]+)",
    # Launch Statistics
    "registers":           r"Registers Per Thread\s+register/thread\s+([\d.,]+)",
    "block_size":          r"^\s+Block Size\s+([\d.,]+)",
    "grid_size":           r"^\s+Grid Size\s+([\d.,]+)",
    "static_smem_bytes":   r"Static Shared Memory Per Block\s+byte/block\s+([\d.,]+)",
    "dyn_smem_bytes":      r"Dynamic Shared Memory Per Block\s+byte/block\s+([\d.,]+)",
    "waves_per_sm":        r"Waves Per SM\s+([\d.,]+)",
    # Occupancy
    "occupancy_pct":       r"Achieved Occupancy\s+%\s+([\d.,]+)",
    "theo_occupancy_pct":  r"Theoretical Occupancy\s+%\s+([\d.,]+)",
    # Warp State
    "warp_stall_long":     r"Stall Long Scoreboard\s+([\d.,]+)",
    "warp_stall_barrier":  r"Stall Barrier\s+([\d.,]+)",
    "warp_stall_short":    r"Stall Short Scoreboard\s+([\d.,]+)",
    # Bank conflicts (Memory Workload Analysis / Shared Memory)
    "smem_bank_conflicts": r"Bank Conflicts\s+([\d.,]+)",
    # Source Counters (branch efficiency)
    "branch_efficiency":   r"Branch Efficiency\s+%\s+([\d.,]+)",
    "avg_pred_on":         r"Avg\. Not Predicated Off Threads Per Warp\s+([\d.,]+)",
}

# The kernel header line looks like:
#   <unnamed>::matmul_kernel(const float *, const float *, ...) (64, 64, 1)x(16, 16, 1), Context 1, ...
KERNEL_HEADER = re.compile(
    r"^\s+([\w:<>]+)::(\w+)\(.*?\)\s+\((.*?)\)x\((.*?)\),\s+Context",
    re.MULTILINE,
)


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


def parse_kernel_blocks(text: str) -> list[dict]:
    """Split the log into per-kernel-launch blocks and extract metrics."""
    # Find kernel launch headers.  Each block runs from one header to the next.
    starts = [(m.start(), m) for m in KERNEL_HEADER.finditer(text)]
    if not starts:
        return []

    ends = [s for (s, _) in starts[1:]] + [len(text)]
    kernels: list[dict] = []
    for (start, m), end in zip(starts, ends):
        block = text[start:end]
        kernel_name = m.group(2)
        grid = m.group(3).replace(" ", "")
        block_dim = m.group(4).replace(" ", "")
        record: dict = {
            "kernel": kernel_name,
            "grid": grid,
            "block": block_dim,
        }
        for name, pat in METRIC_PATTERNS.items():
            mm = re.search(pat, block, re.MULTILINE)
            if mm:
                try:
                    record[name] = _to_float(mm.group(1))
                except ValueError:
                    pass
        kernels.append(record)
    return kernels


def parse_log_meta(text: str) -> dict:
    meta: dict = {}
    for line in text.splitlines():
        if not line.startswith("# "):
            break
        # Format: "# key : value"
        m = re.match(r"^# ([\w ]+?)\s*:\s*(.*)$", line)
        if m:
            key = m.group(1).replace(" ", "_")
            meta[key] = m.group(2).strip()
    return meta


def process_log(path: Path) -> dict:
    text = path.read_text()
    return {
        "log": str(path.relative_to(REPO)),
        "meta": parse_log_meta(text),
        "kernels": parse_kernel_blocks(text),
    }


def normalize_duration_us(k: dict) -> float | None:
    """Return duration in µs regardless of what unit ncu picked."""
    if "duration_us" in k:
        return k["duration_us"]
    if "duration_ns" in k:
        return k["duration_ns"] / 1000.0
    if "duration_ms" in k:
        return k["duration_ms"] * 1000.0
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", default=[],
                        help="Only process this chapter (repeatable).")
    args = parser.parse_args()

    only = set(args.only)
    chapters = sorted((REPO / "src" / "cuda").glob("[0-9]*"))
    total_logs = 0
    for chapter in chapters:
        if not chapter.is_dir() or not (chapter / "ncu").exists():
            continue
        if only and chapter.name not in only:
            continue

        summary_entries: list[dict] = []
        for log_path in sorted((chapter / "ncu").glob("*.txt")):
            parsed = process_log(log_path)
            for k in parsed["kernels"]:
                dur = normalize_duration_us(k)
                if dur is not None:
                    k["duration_us_norm"] = dur

            # Write per-log JSON alongside the .txt.
            out = log_path.with_suffix(".json")
            out.write_text(json.dumps(parsed, indent=2))
            total_logs += 1

            # Also record a summary row per (log, kernel).
            for k in parsed["kernels"]:
                summary_entries.append({
                    "log": log_path.name,
                    "size_tag": parsed["meta"].get("size_tag", ""),
                    **k,
                })

        (chapter / "ncu" / "summary.json").write_text(
            json.dumps(summary_entries, indent=2))
        print(f"{chapter.name}: {len(list((chapter / 'ncu').glob('*.txt')))} logs, "
              f"{len(summary_entries)} kernel records")

    print(f"total logs processed: {total_logs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
