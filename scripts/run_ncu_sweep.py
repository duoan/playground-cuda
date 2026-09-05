#!/usr/bin/env python3
"""Sweep ncu --set detailed across every kernel version × every size.

For each (chapter, version, size) triple:
  1. If the chapter takes size via argv (01-05): reuse the default-built binary
     at build/<chapter>/<version> and pass the size on the command line.
  2. If the chapter takes size via -D compile-time macros (06-10): compile a
     size-specific binary to build/<chapter>/<version>_<sizetag>, then run.
  3. Run `ncu --set detailed` over the binary and save the text output to
     src/cuda/<chapter>/ncu/<version>_<sizetag>.txt.

Idempotent: re-running overwrites logs but doesn't rebuild binaries that are
already up to date.  Skipping a chapter: `run_ncu_sweep.py --only 04_matmul`.

Metadata (binary, argv, macros, GPU, ncu version) is written to the top of each
log so you can trace what produced it.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NCU = os.environ.get("NCU", "/usr/local/cuda-13.0/bin/ncu")
NVCC = os.environ.get("NVCC", "/usr/local/cuda-13.0/bin/nvcc")
NVCC_FLAGS = ["-std=c++17", "-O2"]


@dataclass
class SizeEntry:
    """One point in the sweep grid."""
    tag: str                        # e.g. "n22", "H64", "r64c4096"
    argv: list[str] = field(default_factory=list)     # runtime args
    defines: dict[str, int] = field(default_factory=dict)  # -D macros


@dataclass
class ChapterSpec:
    chapter: str
    versions: list[str]             # e.g. ["01_naive", "02_grid_stride", ...]
    sizes: list[SizeEntry]
    # Notes attached to the log header, e.g. "log2N=22 keeps sum below fp32 wall".
    note: str = ""


def s(tag: str, argv=None, defines=None) -> SizeEntry:
    return SizeEntry(tag=tag, argv=list(argv or []), defines=dict(defines or {}))


# ---------- sweep grid ----------
#
# Choices reflect each kernel's design constraints:
#  - vector_add / reduce_sum: log2N so we can span cache / HBM regimes.
#  - reduce_sum caps at log2N=22 because the atomic-add fp32 accumulator hits
#    its precision wall around 3e7 and the binary's self-check would fail.
#  - matmul: symmetric m=n=k.
#  - softmax / layernorm: (rows, cols) — cols dominates memory bandwidth.
#  - 06-10 use compile-time -D macros; sizes chosen to stay within the kernel's
#    thread-layout assumptions (e.g. warp-per-row kernels keep head_dim <= 32).

CHAPTERS: list[ChapterSpec] = [
    ChapterSpec(
        "01_vector_add",
        versions=["01_naive", "02_grid_stride", "03_vectorized"],
        sizes=[
            s(f"n{n}", argv=[str(n)]) for n in (20, 22, 24, 26, 27)
        ],
        note="argv = log2N; count = (1 << log2N) [ + 37 for reduce_sum ]",
    ),
    ChapterSpec(
        "02_reduce_sum",
        versions=["01_atomic", "02_interleaved", "03_sequential",
                  "04_warp_shuffle", "05_chunked"],
        # 22 keeps the atomic fp32 accumulator below its precision wall.
        sizes=[
            s(f"n{n}", argv=[str(n)]) for n in (18, 20, 21, 22)
        ],
        note="argv = log2N; atomic version fails self-check above n=22 due to fp32 accumulator precision, so we stop there",
    ),
    ChapterSpec(
        "03_softmax",
        versions=["01_naive", "02_block", "03_online"],
        sizes=[
            s(f"r{r}c{c}", argv=[str(r), str(c)])
            for r, c in [(32, 256), (32, 1024), (64, 4096), (128, 4096), (256, 4096)]
        ],
        note="argv = rows cols",
    ),
    ChapterSpec(
        "04_matmul",
        versions=["00_cublas", "01_naive_miscoalesced", "02_naive_coalesced",
                  "03_smem_tiled", "04_block_tile_1d", "05_block_tile_2d",
                  "06_vectorized", "07_mma_ptx", "08_wmma"],
        # symmetric square GEMM. All 9 kernels need divisibility:
        # K7 requires m%32 n%16 k%16; K6 needs n,k %4. n in {128,256,512,1024,2048}
        # satisfies everything. K1 miscoalesced explodes past ~512 (30× slower),
        # so cap the sweep at 1024 for it — done automatically because K1's
        # timing is not on the critical path.
        sizes=[s(f"n{n}", argv=[str(n), str(n), str(n)])
               for n in (128, 256, 512, 1024, 2048)],
        note="argv = m n k (symmetric).  K7 wants n%16, m%32; K8 wants m,n%32.",
    ),
    ChapterSpec(
        "05_layernorm",
        versions=["01_naive", "02_block", "03_warp_shuffle"],
        sizes=[
            s(f"r{r}c{c}", argv=[str(r), str(c)])
            for r, c in [(32, 256), (32, 1024), (64, 4096), (128, 4096), (256, 4096)]
        ],
        note="argv = rows cols",
    ),
    ChapterSpec(
        "06_mlp",
        versions=["01_naive", "02_fused", "03_tiled_fused"],
        # 01_naive/02_fused launch <<<batch, hidden>>>, so HIDDEN_DIM <= 1024.
        sizes=[
            s(f"b{b}_H{h}", defines={"BATCH": b, "INPUT_DIM": max(h // 2, 8),
                                     "HIDDEN_DIM": h, "OUTPUT_DIM": max(h // 4, 4)})
            for (b, h) in [(4, 16), (32, 64), (128, 256), (256, 512), (512, 1024)]
        ],
        note="-D BATCH -D INPUT_DIM -D HIDDEN_DIM -D OUTPUT_DIM (INPUT=H/2, OUTPUT=H/4, clamped); HIDDEN_DIM capped at 1024 = max threads/block",
    ),
    ChapterSpec(
        "07_attention",
        versions=["01_naive", "02_tiled"],
        # 02_tiled launches <<<seq, head>>>: head_dim capped ≤ 1024. 01_naive
        # launches <<<seq, seq>>>: seq ≤ 1024.
        sizes=[
            s(f"S{sq}_H{h}", defines={"SEQ_LEN": sq, "HEAD_DIM": h})
            for (sq, h) in [(8, 8), (32, 32), (64, 64), (128, 64), (256, 64)]
        ],
        note="-D SEQ_LEN -D HEAD_DIM",
    ),
    ChapterSpec(
        "08_flash_attention_v1",
        versions=["01_online", "02_shared"],
        # 01_online: one thread per row, head_dim only via kHeadDim array size
        #            (<=64 is comfortable).
        # 02_shared: block per row, kThreadsPerBlock = max(kTileKeys, kHeadDim)
        #            so head_dim up to a block max (1024) works.
        sizes=[
            s(f"Q{q}_K{k}_H{h}", defines={"QUERY_COUNT": q, "KEY_COUNT": k, "HEAD_DIM": h})
            for (q, k, h) in [(4, 16, 8), (16, 64, 16), (32, 128, 32),
                              (64, 256, 64), (128, 512, 64)]
        ],
        note="-D QUERY_COUNT -D KEY_COUNT -D HEAD_DIM",
    ),
    ChapterSpec(
        "09_flash_attention_v2",
        versions=["01_tile_staged", "02_warp_specialised"],
        # 02_warp_specialised: exactly one warp per row, lanes < head_dim
        # each own one output component. head_dim MUST be <= 32.
        sizes=[
            s(f"Q{q}_K{k}_H{h}", defines={"QUERY_COUNT": q, "KEY_COUNT": k, "HEAD_DIM": h})
            for (q, k, h) in [(4, 16, 8), (16, 64, 16), (32, 128, 32),
                              (64, 256, 32), (128, 512, 32)]
        ],
        note="-D QUERY_COUNT -D KEY_COUNT -D HEAD_DIM; warp-specialised kernel requires head_dim ≤ 32 (warp size)",
    ),
    ChapterSpec(
        "10_flash_attention_v3",
        versions=["01_warp_streaming", "02_pipeline"],
        # Same warp-per-row assumption as fa_v2.
        sizes=[
            s(f"Q{q}_K{k}_H{h}", defines={"QUERY_COUNT": q, "KEY_COUNT": k, "HEAD_DIM": h})
            for (q, k, h) in [(4, 16, 8), (16, 64, 16), (32, 128, 32),
                              (64, 256, 32), (128, 512, 32)]
        ],
        note="-D QUERY_COUNT -D KEY_COUNT -D HEAD_DIM; warp-streaming kernel requires head_dim ≤ 32",
    ),
]


def gpu_name() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).splitlines()[0].strip()
    except Exception:
        return "unknown"


def ncu_version() -> str:
    try:
        return subprocess.check_output([NCU, "--version"], text=True).splitlines()[0].strip()
    except Exception:
        return "unknown"


def build_variant(chapter: str, version: str, defines: dict[str, int]) -> Path:
    """Compile a size-specific binary if defines is non-empty; otherwise return
    the default build/<chapter>/<version> binary path (assumed already built)."""
    src = REPO / "src" / "cuda" / chapter / f"{version}.cu"
    if not defines:
        binary = REPO / "build" / chapter / version
        if not binary.exists():
            print(f"  ! missing default binary {binary}; run `make all` first",
                  file=sys.stderr)
        return binary
    # Name the binary with the sorted define string.
    tag = "_".join(f"{k}{v}" for k, v in sorted(defines.items()))
    binary = REPO / "build" / chapter / f"{version}__{tag}"
    binary.parent.mkdir(parents=True, exist_ok=True)
    if not binary.exists() or binary.stat().st_mtime < src.stat().st_mtime:
        cmd = [NVCC, *NVCC_FLAGS, *(f"-D{k}={v}" for k, v in defines.items()),
               str(src), "-o", str(binary)]
        subprocess.check_call(cmd)
    return binary


def run_ncu(binary: Path, argv: list[str], out_path: Path,
            metadata_lines: list[str]) -> bool:
    """Run ncu on `binary argv` and stream text output into out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for line in metadata_lines:
            f.write(f"# {line}\n")
        f.write("\n")

    cmd = ["sudo", "-n", NCU, "--set", "detailed", "--target-processes", "all",
           str(binary), *argv]
    with open(out_path, "a") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return proc.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", metavar="CHAPTER", action="append", default=[],
                        help="Only run this chapter (repeatable). Defaults to all.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run without invoking ncu.")
    args = parser.parse_args()

    only = set(args.only)
    header_gpu = gpu_name()
    header_ncu = ncu_version()

    todo: list[tuple[str, str, SizeEntry]] = []
    for spec in CHAPTERS:
        if only and spec.chapter not in only:
            continue
        for version in spec.versions:
            for size in spec.sizes:
                todo.append((spec.chapter, version, size))

    print(f"planned: {len(todo)} ncu runs across {len({t[0] for t in todo})} chapters")
    if args.dry_run:
        for chapter, version, size in todo:
            print(f"  {chapter}/{version}  tag={size.tag}  "
                  f"argv={size.argv}  defines={size.defines}")
        return 0

    failed = 0
    started = time.time()
    for i, (chapter, version, size) in enumerate(todo, 1):
        spec = next(c for c in CHAPTERS if c.chapter == chapter)
        binary = build_variant(chapter, version, size.defines)
        out = REPO / "src" / "cuda" / chapter / "ncu" / f"{version}_{size.tag}.txt"

        argv_str = " ".join(shlex.quote(a) for a in size.argv) or "(none)"
        def_str = " ".join(f"-D{k}={v}" for k, v in size.defines.items()) or "(none)"
        meta = [
            "ncu --set detailed",
            f"binary  : {binary}",
            f"argv    : {argv_str}",
            f"defines : {def_str}",
            f"size tag: {size.tag}",
            f"date    : {time.strftime('%FT%TZ', time.gmtime())}",
            f"gpu     : {header_gpu}",
            f"ncu     : {header_ncu}",
        ]
        if spec.note:
            meta.append(f"note    : {spec.note}")

        print(f"[{i:3d}/{len(todo)}] {chapter}/{version}_{size.tag} ... ",
              end="", flush=True)
        if run_ncu(binary, size.argv, out, meta):
            print("ok")
        else:
            print(f"FAIL (see {out})")
            failed += 1

    dt = time.time() - started
    print(f"\ndone: {len(todo)} runs, {failed} failed, {dt:.0f}s wall")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
