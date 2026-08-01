-- Passages the system saw and did not count. This is the evidence for a missing exit.
SELECT timestamp, camera, frames, slices, peak, lag, person, direction
FROM passages
WHERE counted = 0 AND timestamp BETWEEN ? AND ?
ORDER BY timestamp;
