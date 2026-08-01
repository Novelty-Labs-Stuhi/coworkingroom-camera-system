-- Put a corrected name on the one crossing a sighting is about.
--
-- Matched on the moment and the direction rather than an id, because the two records were
-- written by different parts of the system and share only that. The window is a second: two
-- crossings the same way within a second are the same passage.
UPDATE events
   SET name = ?
 WHERE direction = ?
   AND timestamp BETWEEN ? AND ?;
