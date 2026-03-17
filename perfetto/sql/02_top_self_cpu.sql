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
