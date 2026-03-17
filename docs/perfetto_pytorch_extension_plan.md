# Perfetto PyTorch Extension Plan

This doc turns the "can we make this a Perfetto extension?" idea into an implementation plan.

Short answer: yes, but there are two different targets.

## The Two Realistic Targets

## 1. SQL Pack in This Repo

This is the fastest path and the one we can use immediately.

What it gives us:

- versioned SQL files for common PyTorch profiler queries
- a repeatable analysis workflow across traces
- a foundation for later UI automation or a plugin

This is what lives in:

- [perfetto/sql](/home/duoan/playground-cuda/perfetto/sql)

## 2. Custom Perfetto UI Plugin

This is the more ambitious path.

What it would give us:

- a `PyTorch Summary` tab inside Perfetto UI
- buttons for common queries like `Top self CPU`, `Top CUDA`, `Launch overhead`
- auto-detection of PyTorch traces
- maybe a step picker and a custom notes panel

Important limitation:

- Perfetto's plugin model is real, but plugins are still in-tree today
- that means we cannot just side-load a private plugin into `ui.perfetto.dev`
- for a true custom plugin, we should plan on forking and self-hosting Perfetto UI

Official references:

- https://perfetto.dev/docs/contributing/ui-plugins
- https://perfetto.dev/docs/contributing/ui-getting-started
- https://perfetto.dev/docs/visualization/ui-automation
- https://perfetto.dev/docs/analysis/perfetto-sql-syntax

## Product Goal

Make Perfetto feel closer to `torch.profiler` summary output while keeping Perfetto's timeline and SQL strengths.

The user experience we want is:

1. open a PyTorch trace
2. click `PyTorch Summary`
3. instantly see:
   - top self CPU ops
   - top CUDA ops
   - most-called ops
   - launch overhead
   - top kernels
4. click one result row and jump back to timeline context

## Scope for an MVP

An MVP should avoid custom rendering complexity and stay close to Perfetto's built-in table/query flow.

MVP features:

- detect whether the trace looks like a PyTorch profiler trace
- expose a `PyTorch Summary` page or tab
- run a fixed set of SQL queries
- render the result tables
- optionally expose a quick filter by `ProfilerStep#`

Nice-to-have later:

- step comparison view
- auto-highlighting of `user_annotation` regions like `data_loader`, `h2d`, `forward`, `backward`
- saved presets for training, inference, DDP, and token-skew traces
- deep links from summary rows to relevant tracks or slices

## Proposed Architecture

## Layer 1: Query Assets

Keep the actual SQL in repo-owned files:

- [perfetto/sql/01_top_cpu_total.sql](/home/duoan/playground-cuda/perfetto/sql/01_top_cpu_total.sql)
- [perfetto/sql/02_top_self_cpu.sql](/home/duoan/playground-cuda/perfetto/sql/02_top_self_cpu.sql)
- [perfetto/sql/03_top_cuda_total.sql](/home/duoan/playground-cuda/perfetto/sql/03_top_cuda_total.sql)
- [perfetto/sql/04_top_cpu_calls.sql](/home/duoan/playground-cuda/perfetto/sql/04_top_cpu_calls.sql)
- [perfetto/sql/05_launch_overhead.sql](/home/duoan/playground-cuda/perfetto/sql/05_launch_overhead.sql)
- [perfetto/sql/06_top_gpu_kernels.sql](/home/duoan/playground-cuda/perfetto/sql/06_top_gpu_kernels.sql)
- [perfetto/sql/07_memcpy_summary.sql](/home/duoan/playground-cuda/perfetto/sql/07_memcpy_summary.sql)
- [perfetto/sql/08_busy_tracks.sql](/home/duoan/playground-cuda/perfetto/sql/08_busy_tracks.sql)
- [perfetto/sql/09_step_local_cpu_summary.sql](/home/duoan/playground-cuda/perfetto/sql/09_step_local_cpu_summary.sql)
- [perfetto/sql/10_user_annotations.sql](/home/duoan/playground-cuda/perfetto/sql/10_user_annotations.sql)

This keeps query logic independent from any eventual UI implementation.

## Layer 2: Trace Detection

The plugin should first ask whether the loaded trace is likely a PyTorch trace.

Heuristics:

- `slice.category` contains `cpu_op`
- `slice.category` contains `kernel`
- names like `aten::...`
- optional `ProfilerStep#...`
- optional `External id` on CPU and GPU slices

If the heuristics fail, the plugin can hide itself or show a helpful "not a PyTorch profiler trace" message.

## Layer 3: Summary Views

The initial UI only needs a handful of fixed tables:

- `Top Self CPU`
- `Top CUDA`
- `Top CPU Calls`
- `Launch Overhead`
- `Top Kernels`
- `Memcpy`
- `User Annotations`

This is enough to cover most first-pass analysis.

## Layer 4: Step-Aware Analysis

Once step markers are available, add:

- dropdown of `ProfilerStep#...`
- ability to rerun a step-local query
- comparison of one normal step vs one slow step

This would be especially useful for:

- data jitter
- warmup effects
- DDP skew
- variable-sequence-length traces

## Suggested Repository Layout

If we later split this into a dedicated Perfetto fork or side project, a clean starting layout would be:

```text
perfetto-pytorch-ui/
  ui/
    src/plugins/dev.duoan.pytorch_profiler/
      index.ts
      pytorch_trace_detection.ts
      pytorch_queries.ts
      pytorch_summary_page.ts
      pytorch_step_page.ts
  queries/
    01_top_cpu_total.sql
    02_top_self_cpu.sql
    03_top_cuda_total.sql
    ...
```

For now, this repo should only own:

- the SQL pack
- the analysis docs
- sample traces and experiments

## Implementation Phases

## Phase 1: Done in This Repo

- write the reusable SQL pack
- document the recommended query order
- document plugin design constraints

Deliverables:

- [perfetto_pytorch_profiler_sql.md](/home/duoan/playground-cuda/docs/perfetto_pytorch_profiler_sql.md)
- [perfetto/sql](/home/duoan/playground-cuda/perfetto/sql)
- this design doc

## Phase 2: Self-Hosted Query Dashboard

Before a full plugin, an intermediate step is useful:

- self-host Perfetto UI
- add saved query tabs or startup automation
- test the workflow on the traces in this repo

This gives most of the value with much less UI code.

## Phase 3: Minimal Plugin MVP

Build a small plugin that:

- registers a `PyTorch Summary` command
- opens a summary page
- runs the fixed SQL queries
- displays result tables

Success criteria:

- a new user can open a PyTorch trace and get a first-pass summary in under a minute

## Phase 4: Rich PyTorch Workflow

Only after the MVP works:

- add step picker
- add result-to-timeline navigation
- add presets for training, DDP, and token-skew cases
- maybe add a "what to look at next" panel

## Query Priorities for the MVP

If we keep scope tight, these are the only must-have queries:

1. top self CPU
2. top CUDA total
3. top CPU call count
4. launch overhead
5. top GPU kernels

Everything else can be a second pass.

## Risks and Caveats

- `self_cpu_time_total` is approximate in raw Perfetto SQL unless we reproduce PyTorch's exact nesting semantics.
- `cuda_time_total` depends on `External id`; some traces may not contain that field.
- large traces can make expensive SQL feel slow, so queries should stay simple at first.
- the best user experience likely requires a self-hosted Perfetto UI, not the public one.

## Recommended Next Step

The highest-leverage next step is not a full plugin yet.

It is:

1. keep refining the SQL pack on real traces
2. validate which tables are truly useful every day
3. only then freeze an MVP plugin surface

That keeps us from building UI around the wrong mental model.
