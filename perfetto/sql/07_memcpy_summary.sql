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
