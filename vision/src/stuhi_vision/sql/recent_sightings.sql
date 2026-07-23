SELECT timestamp, name, clarity
FROM sightings
ORDER BY timestamp DESC
LIMIT ?;
