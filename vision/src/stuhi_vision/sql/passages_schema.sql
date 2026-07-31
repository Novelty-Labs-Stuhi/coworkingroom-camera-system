-- Every episode of the doorframe being covered, counted or refused.
--
-- The events table holds only what was committed, so a refusal leaves no trace and the
-- question "why are there fewer exits than entries" cannot be answered from it -- the log
-- that did hold the refusals is rotated within hours. This keeps the evidence: how long the
-- box was covered, how many slices lit, which way they lit, whether a person was on it, and
-- what was decided. Same shape for a passage and a refusal, so the two are comparable.
CREATE TABLE IF NOT EXISTS passages (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL    NOT NULL,
    camera    TEXT    NOT NULL,
    frames    INTEGER NOT NULL,   -- how long the box stayed covered
    slices    INTEGER NOT NULL,   -- how many vertical slices lit at some point
    peak      REAL    NOT NULL,   -- the most of any slice that changed
    lag       REAL    NOT NULL,   -- frames of delay per slice; 0 means no order at all
    person    INTEGER,            -- the track credited, NULL when nobody was detected
    direction TEXT,               -- NULL when it was not judged a passage
    counted   INTEGER NOT NULL    -- whether it reached the ledger
);

CREATE INDEX IF NOT EXISTS idx_passages_timestamp ON passages (timestamp);
