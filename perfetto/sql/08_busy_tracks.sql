SELECT
  track.name AS track_name,
  COUNT(*) AS calls,
  ROUND(SUM(slice.dur) / 1e6, 3) AS total_ms
FROM slice
JOIN track ON slice.track_id = track.id
GROUP BY track_name
ORDER BY total_ms DESC
LIMIT 30;
