CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL    NOT NULL,
    name      TEXT    NOT NULL,
    direction TEXT    NOT NULL CHECK (direction IN ('in', 'out')),
    camera    TEXT    NOT NULL DEFAULT '',
    -- How an exit got its name, and what its own evidence said before the room was consulted.
    named_by  TEXT    NOT NULL DEFAULT '',
    natural   TEXT    NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events (timestamp);
