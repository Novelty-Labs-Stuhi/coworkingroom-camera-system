CREATE TABLE IF NOT EXISTS sightings (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL    NOT NULL,
    name      TEXT    NOT NULL,
    clarity   REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sightings_timestamp ON sightings (timestamp);
