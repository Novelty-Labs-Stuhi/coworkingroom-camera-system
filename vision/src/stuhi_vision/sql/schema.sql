CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL    NOT NULL,
    name      TEXT    NOT NULL,
    direction TEXT    NOT NULL CHECK (direction IN ('in', 'out'))
);

CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events (timestamp);
