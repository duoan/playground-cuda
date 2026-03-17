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
