SELECT timestamp, name, direction, camera
FROM events
ORDER BY timestamp DESC
LIMIT ?;
