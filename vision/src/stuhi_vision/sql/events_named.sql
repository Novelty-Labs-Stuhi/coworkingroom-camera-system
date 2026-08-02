-- The rows carrying one name, so a merge can record exactly what it is about to change.
SELECT id FROM events WHERE name = ?;
