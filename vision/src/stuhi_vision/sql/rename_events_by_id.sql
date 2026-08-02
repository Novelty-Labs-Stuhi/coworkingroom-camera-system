-- Put a name on specific rows. Used by a merge, which has already written down which rows
-- it is changing, so undoing it later can find precisely those and no others.
UPDATE events SET name = ? WHERE id = ?;
