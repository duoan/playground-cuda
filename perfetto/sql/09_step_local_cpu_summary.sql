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
