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
