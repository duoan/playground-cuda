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

This file collects the SQL I use most often.

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

## 1. Top Ops by CPU Total Time

This is the simplest useful table.

```sql
SELECT
  name,
  COUNT(*) AS calls,
  ROUND(SUM(dur) / 1e6, 3) AS cpu_total_ms,
  ROUND(AVG(dur) / 1e3, 3) AS cpu_avg_us
FROM slice
WHERE category = 'cpu_op' AND dur > 0
GROUP BY name
ORDER BY cpu_total_ms DESC
LIMIT 30;
```

Use this when you want something close to:

```python
prof.key_averages().table(sort_by="cpu_time_total")
```

## 2. Top Ops by Approximate Self CPU Time

This query approximates `self_cpu_time_total` by subtracting direct child CPU-op time from each CPU op.

```sql
WITH cpu_ops AS (
  SELECT
    id,
    name,
    track_id,
    ts,
    dur,
    depth
  FROM slice
  WHERE category = 'cpu_op' AND dur > 0
),
direct_child_time AS (
  SELECT
    p.id AS parent_id,
    COALESCE(SUM(c.dur), 0) AS child_dur
  FROM cpu_ops p
  LEFT JOIN cpu_ops c
    ON c.track_id = p.track_id
   AND c.depth = p.depth + 1
   AND c.ts >= p.ts
   AND c.ts + c.dur <= p.ts + p.dur
  GROUP BY p.id
),
per_call AS (
  SELECT
    p.id,
    p.name,
    p.dur,
    p.dur - d.child_dur AS self_dur
  FROM cpu_ops p
  JOIN direct_child_time d
    ON p.id = d.parent_id
)
SELECT
  name,
  COUNT(*) AS calls,
  ROUND(SUM(self_dur) / 1e6, 3) AS self_cpu_ms,
  ROUND(SUM(dur) / 1e6, 3) AS cpu_total_ms,
  ROUND(AVG(self_dur) / 1e3, 3) AS self_cpu_avg_us
FROM per_call
GROUP BY name
ORDER BY self_cpu_ms DESC
LIMIT 30;
```

Use this when you want something close to:

```python
prof.key_averages().table(sort_by="self_cpu_time_total")
```

## 3. Top Ops by Approximate CUDA Total Time

This query links CPU ops to GPU work using `External id`.

```sql
WITH cpu_ops AS (
  SELECT
    id,
    name,
    dur,
    CAST(EXTRACT_ARG(arg_set_id, 'External id') AS INT) AS ext_id
  FROM slice
  WHERE category = 'cpu_op' AND dur > 0
),
gpu_work AS (
  SELECT
    id,
    dur,
    CAST(EXTRACT_ARG(arg_set_id, 'External id') AS INT) AS ext_id
  FROM slice
  WHERE category IN ('kernel', 'gpu_memcpy', 'gpu_memset') AND dur > 0
),
per_cpu_call AS (
  SELECT
    c.id,
    c.name,
    c.dur,
    COALESCE(SUM(g.dur), 0) AS cuda_dur
  FROM cpu_ops c
  LEFT JOIN gpu_work g
    ON c.ext_id = g.ext_id
  GROUP BY c.id, c.name, c.dur
)
SELECT
  name,
  COUNT(*) AS calls,
  ROUND(SUM(cuda_dur) / 1e6, 3) AS cuda_total_ms,
  ROUND(AVG(cuda_dur) / 1e3, 3) AS cuda_avg_us,
  ROUND(SUM(dur) / 1e6, 3) AS cpu_total_ms
FROM per_cpu_call
GROUP BY name
ORDER BY cuda_total_ms DESC
LIMIT 30;
```

Use this when you want something close to:

```python
prof.key_averages().table(sort_by="cuda_time_total")
```

## 4. Most Frequently Called CPU Ops

This is great for spotting eager/tiny-op problems.

```sql
SELECT
  name,
  COUNT(*) AS calls,
  ROUND(SUM(dur) / 1e6, 3) AS total_ms,
  ROUND(AVG(dur) / 1e3, 3) AS avg_us
FROM slice
WHERE category = 'cpu_op' AND dur > 0
GROUP BY name
ORDER BY calls DESC
LIMIT 30;
```

Typical signals:

- many `aten::add`
- many `aten::mul`
- many `aten::relu`
- tiny average duration with huge call count

## 5. CUDA Launch Overhead

This is useful when the job looks launch-bound.

```sql
SELECT
  name,
  COUNT(*) AS calls,
  ROUND(SUM(dur) / 1e6, 3) AS total_ms,
  ROUND(AVG(dur) / 1e3, 3) AS avg_us
FROM slice
WHERE category = 'cuda_runtime'
  AND name LIKE '%cudaLaunchKernel%'
  AND dur > 0
GROUP BY name
ORDER BY total_ms DESC;
```

If this is large and your kernels are tiny, you are probably paying too much launch overhead.

## 6. Top GPU Kernels by Total Time

This is the direct GPU-side hot-kernel view.

```sql
SELECT
  name,
  COUNT(*) AS calls,
  ROUND(SUM(dur) / 1e6, 3) AS total_ms,
  ROUND(AVG(dur) / 1e3, 3) AS avg_us
FROM slice
WHERE category = 'kernel' AND dur > 0
GROUP BY name
ORDER BY total_ms DESC
LIMIT 30;
```

Use this before going deeper with `ncu`.

## 7. H2D / Memcpy Summary

Useful when you suspect copies are part of the problem.

```sql
SELECT
  category,
  name,
  COUNT(*) AS calls,
  ROUND(SUM(dur) / 1e6, 3) AS total_ms,
  ROUND(AVG(dur) / 1e3, 3) AS avg_us
FROM slice
WHERE category IN ('gpu_memcpy', 'cuda_runtime')
  AND (name LIKE '%Memcpy%' OR name LIKE '%copy%' OR name LIKE '%cudaMemcpyAsync%')
  AND dur > 0
GROUP BY category, name
ORDER BY total_ms DESC
LIMIT 30;
```

## 8. Per-Track Busy Time

This helps identify which CPU threads or GPU streams are busiest.

```sql
SELECT
  track.name AS track_name,
  COUNT(*) AS calls,
  ROUND(SUM(slice.dur) / 1e6, 3) AS total_ms
FROM slice
JOIN track ON slice.track_id = track.id
GROUP BY track_name
ORDER BY total_ms DESC
LIMIT 30;
```

Useful for:

- finding a hot CPU thread
- seeing if work is concentrated on one stream
- understanding where a trace is spending time structurally

## 9. Top Ops Inside a Single Profiler Step

If your trace includes `ProfilerStep#...`, you can isolate one step and summarize only that window.

First find the step:

```sql
SELECT
  id,
  name,
  ts,
  dur
FROM slice
WHERE name LIKE 'ProfilerStep#%'
ORDER BY ts;
```

Then plug the chosen `ts`/`dur` into:

```sql
WITH step_window AS (
  SELECT
    0 AS dummy,
    6542333657815.59 AS step_ts,
    9047.935 AS step_dur
),
cpu_ops AS (
  SELECT
    s.name,
    s.dur
  FROM slice s, step_window w
  WHERE s.category = 'cpu_op'
    AND s.ts >= w.step_ts
    AND s.ts + s.dur <= w.step_ts + w.step_dur
)
SELECT
  name,
  COUNT(*) AS calls,
  ROUND(SUM(dur) / 1e6, 3) AS cpu_total_ms
FROM cpu_ops
GROUP BY name
ORDER BY cpu_total_ms DESC
LIMIT 30;
```

This is great for comparing:

- a normal step
- a slow step
- a warmup step

## 10. Find DataLoader / Annotation Regions

If you added `record_function()` regions like `data_loader`, `h2d`, `forward`, `backward`, `optimizer`, this query summarizes them.

```sql
SELECT
  name,
  COUNT(*) AS calls,
  ROUND(SUM(dur) / 1e6, 3) AS total_ms,
  ROUND(AVG(dur) / 1e3, 3) AS avg_us
FROM slice
WHERE category = 'user_annotation' AND dur > 0
GROUP BY name
ORDER BY total_ms DESC;
```

This is often the fastest top-down query in the whole trace.

## How I Usually Use These

My usual order is:

1. Query top `self_cpu`
2. Query top `cuda_total`
3. Query call counts
4. Query `cudaLaunchKernel`
5. Query top kernels
6. Zoom back into timeline to see where the expensive things occur in the step

That gives a workflow very close to:

- console summary first
- Perfetto context second

## Practical Interpretation Tips

- High `self_cpu_ms` for `DataLoader`-related ops usually means the input pipeline is the bottleneck.
- High `cuda_total_ms` concentrated in a small number of ops means you probably have a real GPU hot path worth deeper kernel analysis.
- Huge call count with tiny `avg_us` usually means eager overhead or too many tiny pointwise ops.
- High `cudaLaunchKernel` time usually means launch overhead is becoming visible.
- Large memcpy totals usually mean H2D or transfer behavior deserves attention.

## Suggested Follow-Up

If you use this a lot, the next useful addition is a saved set of query tabs for:

- `self_cpu_time_total`
- `cuda_time_total`
- `calls`
- `launch overhead`
- `step-local summary`

That makes Perfetto much closer to a PyTorch performance dashboard.
