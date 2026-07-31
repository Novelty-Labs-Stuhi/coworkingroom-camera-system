-- Correct a name in the history. A misspelling was never a different person, so the events
-- should not read as though it was: two names in the record would look like two people who
-- each came and went half the time.
UPDATE events SET name = ? WHERE name = ?;
