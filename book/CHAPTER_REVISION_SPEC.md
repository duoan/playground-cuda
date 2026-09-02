# Chapter revision spec (used by subagents)

Every subagent gets this as reference.  Purpose:
- Add / replace the "实测" (measurements) section in one chapter with a
  metric-driven analysis based on the freshly-generated `bench/<N>.md` file.
- Add 1–3 CetZ diagrams from `template.typ` where they help.
- Preserve everything else in the chapter.

## 0. Read these first

- `book/chapters/01_vector_add.typ` — the reference chapter (already revised).
  In particular, look at:
  - the `=== 实测` section (~line 100–170 after edit)
  - how it embeds `bench/01_vector_add.typ` via `#include`
  - how it uses `warp-lanes(...)` for coalescing diagrams
  - the two-table layout (perf + diag) and the way each column is *justified
    by a metric name*, not by hand-waving.

- `book/template.typ` — for the helpers you can use:
  - `warp-lanes(active: (indices), title: ..., note: ...)` — one row, 32 cells
  - `warp-grid(rows, cols, active: [(r,c)...], title: ...)` — general grid
  - `tree-reduction(mode: "interleaved"|"sequential", n: 8)` — reduction diagram
  - `hbar-chart(entries: (("label", value), ...), unit: "μs")` — bar chart
  - Callouts: `#note[...]`, `#insight[...]`, `#warn[...]`, `#interview[...]`

- `book/bench/<chapter>.md` — the fresh benchmark results.  Contains one row
  per kernel version with the columns:

  | column          | metric                                                                |
  |-----------------|-----------------------------------------------------------------------|
  | time (μs)       | `gpu__time_duration.sum`                                              |
  | HBM %           | `dram__bytes.sum.pct_of_peak_sustained_elapsed`                       |
  | SM %            | `sm__cycles_active.avg.pct_of_peak_sustained_elapsed` (mem chapters)  |
  | TC %            | `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` (compute chapters) |
  | warp %          | `sm__warps_active.avg.pct_of_peak_sustained_elapsed`                  |
  | GB/s            | `dram__bytes.sum / time`  (real HBM) `/`  `bytes_per_launch / time`   |
  |                 | (logical); the two should agree when the working set ≫ L2.            |
  | % peak          | GB/s ÷ 2039 (HBM peak, A100 80GB SXM4)                                |
  | issued/32       | `smsp__thread_inst_executed_per_inst_executed.ratio`                  |
  |                 | Lanes participating in each ISSUED instruction (32 = all).            |
  |                 | Lower than 32 means predication *or* branch divergence.               |
  | pred_on/32      | `smsp__average_thread_inst_executed_pred_on_per_inst_executed.ratio`  |
  |                 | Lanes that actually did work (predicated-on).                         |
  |                 | Big gap issued − pred_on ⇒ predication is masking work but the        |
  |                 | warp instruction still occupied an issue slot.                        |
  | smem conf.      | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld+st.sum`         |
  |                 | 0 means no bank conflicts (or no smem use).                           |
  | barrier stall   | `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio`    |
  |                 | Warps stalled at `__syncthreads` per issue-active cycle.              |
  | mem stall       | `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active`  |
  |                 | Warps stalled on global-memory latency.                               |

## 1. Terminology corrections (VERY IMPORTANT)

Two things that were wrong in the earlier draft:

1. *"Warp divergence"* — this term should only be used when different lanes
   in the same warp take **different basic blocks** (e.g. `if x else y`
   where both branches contain work).  A single-sided `if` with an empty
   `else` is compiled to a **predicated instruction**, which is NOT
   divergence.  The correct metric names:
   - `issued/32` (`smsp__thread_inst_executed_per_inst_executed.ratio`)
     = lanes participating in the issued instruction (32 = uniform).
   - `pred_on/32` (`smsp__average_thread_inst_executed_pred_on_per_inst_executed.ratio`)
     = lanes actually doing work.
   The **gap** between these two is where predicated-off lanes live.  Use
   language like "warp lane utilization" or "predicated-off lanes wasting
   issue slots" instead of loosely saying "warp divergence".

2. *"Bank conflict"* — the term is correct, but the metric to prove it is
   `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st}.sum`
   (already in `smem conf.`).  0 conflicts ⇒ no bank conflict — you cannot
   claim conflicts based on the source code alone.

## 2. Style

- Write in Chinese, follow ch1's tone: direct, no fluff.
- One sentence per idea when possible.
- When you cite a metric, name the ncu column and quote the number from
  `bench/<chapter>.md`.
- If a metric is small/uninformative (small-N chapters like softmax /
  attention run only 3-17 μs and hit L2 hard), *say so explicitly*.  Do not
  paper over it.  This is the honesty rule the user asked for.
- Use `#insight[...]` for the one key takeaway per section, `#warn[...]`
  for gotchas and caveats about scale.

## 3. Scale caveat block (paste when appropriate)

For chapters other than ch1/ch2 (which use log2N=27):

```typ
#warn[
  这一章的问题规模是教学 default（B×S×H ~ 数千个 float），kernel 单次运行只有 3–20 μs。ncu 的定性指标（`issued/32`、`bank conflicts`、`barrier stall`）仍能反映 kernel 结构，但*绝对数字对生产规模不完全可信*：
  - HBM % 会偏低（分母 elapsed time 含冷启动窗口）
  - dram_bytes 可能被 L2 消化，`GB/s (实测/逻辑)` 两列差距明显
  想拿到生产规模的数字，把主参数（rows/cols/hidden dim）加到让工作集远超 L2 (40 MB)。
]
```

## 4. Diagrams

Pick 1–3 that illuminate the *specific point* being made.  Do not add
diagrams for filler.  Typical useful ones:

- **Access pattern**: warp-lanes to show coalesced vs strided access
- **Ladder**: hbar-chart of time-per-kernel from the bench
- **Tree reduction**: `tree-reduction(mode: "interleaved"|"sequential")`
- **Tile / block layout**: warp-grid with cells colored by which warp owns
  the tile.

## 5. Do NOT

- Do not rewrite the whole chapter — only the measurement/analysis section.
- Do not remove existing content unless it directly contradicts a metric.
- Do not change the source code in `src/cuda/*.cu`.
- Do not touch `template.typ` or `book.typ`.
- Do not invent new metric names.  If you need one you don't see here,
  stop and ask.

## 6. Deliverable

After editing:
1. Run `cd book && ~/.local/bin/typst compile book.typ book.pdf` and
   ensure zero errors (warnings ok).
2. Report:
   - What section(s) you replaced
   - What diagrams you added
   - Any metric where the number surprised you or contradicted prior text
