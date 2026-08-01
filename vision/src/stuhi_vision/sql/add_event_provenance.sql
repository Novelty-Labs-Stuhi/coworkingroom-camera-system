-- How an exit was named, and what its own evidence said before the room was consulted.
-- Added in place: the history is the point of the table, so it is migrated rather than rebuilt.
ALTER TABLE events ADD COLUMN named_by TEXT NOT NULL DEFAULT '';
ALTER TABLE events ADD COLUMN natural TEXT NOT NULL DEFAULT '';
