-- Adds the camera column to a database created before cameras were named. SQLite has no
-- "ADD COLUMN IF NOT EXISTS", so the caller checks first; running this twice is an error,
-- not a no-op.
ALTER TABLE events ADD COLUMN camera TEXT NOT NULL DEFAULT '';
