# Perfetto SQL for PyTorch Profiler Traces

Common SQL snippets for analyzing `torch.profiler` traces in Perfetto, especially when the trace was exported in Chrome/Perfetto JSON format.

## What This Is For

When you run PyTorch profiler like this:

```python
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
```

Perfetto does not show those exact summary tables by default.

Perfetto is a trace viewer, not a PyTorch-specific summary UI. The way to get similar information is:

- use the timeline for context
- use SQL for aggregation

This repo now keeps the queries in a reusable SQL pack:

- [perfetto/sql/01_top_cpu_total.sql](../perfetto/sql/01_top_cpu_total.sql)
- [perfetto/sql/02_top_self_cpu.sql](../perfetto/sql/02_top_self_cpu.sql)
- [perfetto/sql/03_top_cuda_total.sql](../perfetto/sql/03_top_cuda_total.sql)
- [perfetto/sql/04_top_cpu_calls.sql](../perfetto/sql/04_top_cpu_calls.sql)
- [perfetto/sql/05_launch_overhead.sql](../perfetto/sql/05_launch_overhead.sql)
- [perfetto/sql/06_top_gpu_kernels.sql](../perfetto/sql/06_top_gpu_kernels.sql)
- [perfetto/sql/07_memcpy_summary.sql](../perfetto/sql/07_memcpy_summary.sql)
- [perfetto/sql/08_busy_tracks.sql](../perfetto/sql/08_busy_tracks.sql)
- [perfetto/sql/09_step_local_cpu_summary.sql](../perfetto/sql/09_step_local_cpu_summary.sql)
- [perfetto/sql/10_user_annotations.sql](../perfetto/sql/10_user_annotations.sql)

There is also a follow-up implementation plan for turning this into a real Perfetto-side experience:

- [perfetto_pytorch_extension_plan.md](./perfetto_pytorch_extension_plan.md)

## Important Caveat

These queries are usually very close to PyTorch's console summary, but not always identical.

Reasons:

- `self_cpu_time_total` is a derived metric based on nested CPU ops
- `cuda_time_total` depends on how CPU ops are associated with GPU work
- different trace exports may contain slightly different fields

The queries below work well for traces that contain:

- `category = 'cpu_op'`
- `category = 'kernel'`
- `category = 'gpu_memcpy'`
- `category = 'gpu_memset'`
- `External id` in both CPU and GPU slices

That matches the PyTorch traces used in this repo.

## Recommended Query Order

When I am opening an unfamiliar PyTorch trace in Perfetto, I usually run queries in this order:

1. `02_top_self_cpu.sql`
2. `03_top_cuda_total.sql`
3. `04_top_cpu_calls.sql`
4. `05_launch_overhead.sql`
5. `06_top_gpu_kernels.sql`
6. `08_busy_tracks.sql`
7. `10_user_annotations.sql`

Then I zoom back into the timeline to answer:

- are we data-bound or compute-bound?
- do we have many tiny eager ops?
- is launch overhead visible?
- is the GPU hot path concentrated in a few kernels?
- are custom annotation regions lining up with what the summary said?

## Query Map

## 1. Top Ops by CPU Total Time

File:

- [01_top_cpu_total.sql](../perfetto/sql/01_top_cpu_total.sql)

Use this when you want something close to:

```python
prof.key_averages().table(sort_by="cpu_time_total")
```

## 2. Top Ops by Approximate Self CPU Time

File:

- [02_top_self_cpu.sql](../perfetto/sql/02_top_self_cpu.sql)

Use this when you want something close to:

```python
prof.key_averages().table(sort_by="self_cpu_time_total")
```

## 3. Top Ops by Approximate CUDA Total Time

File:

- [03_top_cuda_total.sql](../perfetto/sql/03_top_cuda_total.sql)

Use this when you want something close to:

```python
prof.key_averages().table(sort_by="cuda_time_total")
```

## 4. Most Frequently Called CPU Ops

File:

- [04_top_cpu_calls.sql](../perfetto/sql/04_top_cpu_calls.sql)

This is great for spotting eager and tiny-op problems.

## 5. CUDA Launch Overhead

File:

- [05_launch_overhead.sql](../perfetto/sql/05_launch_overhead.sql)

If this is large and your kernels are tiny, you are probably paying too much launch overhead.

## 6. Top GPU Kernels by Total Time

File:

- [06_top_gpu_kernels.sql](../perfetto/sql/06_top_gpu_kernels.sql)

Use this before going deeper with `ncu`.

## 7. H2D / Memcpy Summary

File:

- [07_memcpy_summary.sql](../perfetto/sql/07_memcpy_summary.sql)

Useful when you suspect copies are part of the problem.

## 8. Per-Track Busy Time

File:

- [08_busy_tracks.sql](../perfetto/sql/08_busy_tracks.sql)

Useful for:

- finding a hot CPU thread
- seeing if work is concentrated on one stream
- understanding where a trace is spending time structurally

## 9. Top Ops Inside a Single Profiler Step

File:

- [09_step_local_cpu_summary.sql](../perfetto/sql/09_step_local_cpu_summary.sql)

This is great for comparing:

- a normal step
- a slow step
- a warmup step

You need to edit the `step_window` CTE with the `ts` and `dur` for the step you care about.

## 10. Find DataLoader / Annotation Regions

File:

- [10_user_annotations.sql](../perfetto/sql/10_user_annotations.sql)

This is often the fastest top-down query in the whole trace if you used `record_function()`.

## Practical Interpretation Tips

- High `self_cpu_ms` for `DataLoader`-related ops usually means the input pipeline is the bottleneck.
- High `cuda_total_ms` concentrated in a small number of ops means you probably have a real GPU hot path worth deeper kernel analysis.
- Huge call count with tiny `avg_us` usually means eager overhead or too many tiny pointwise ops.
- High `cudaLaunchKernel` time usually means launch overhead is becoming visible.
- Large memcpy totals usually mean H2D or transfer behavior deserves attention.

## Suggested Follow-Up

If you use this a lot, the next useful addition is either:

- a saved set of query tabs in a self-hosted Perfetto UI
- a small PyTorch-specific Perfetto plugin that runs these queries for you

That makes Perfetto much closer to a PyTorch performance dashboard.
