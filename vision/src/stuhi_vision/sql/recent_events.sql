SELECT timestamp, name, direction
FROM events
ORDER BY timestamp DESC
LIMIT ?;
