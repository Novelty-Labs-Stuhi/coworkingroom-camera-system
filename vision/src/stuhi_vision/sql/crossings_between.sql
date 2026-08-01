-- Every crossing in a window, with how each exit was named. Ordered so the pairing can be
-- done in one pass.
SELECT timestamp, name, direction, camera, named_by, natural
FROM events
WHERE timestamp BETWEEN ? AND ?
ORDER BY timestamp;
