"""Durable event log, backed by SQLite from the standard library.

The ledger keeps live occupancy in memory; this records every entry/exit so the
history survives a restart and can be queried later. SQL lives in ``.sql`` files
(never glued into Python strings) and is always parameterised.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from importlib import resources
from pathlib import Path

from .accounting import Crossing
from .domain import Event


def _sql(name: str) -> str:
    return resources.files(f"{__package__}.sql").joinpath(name).read_text(encoding="utf-8")


def _connect(database: Path) -> sqlite3.Connection:
    # check_same_thread=False: the FastAPI server touches this connection from its
    # threadpool workers, not just the thread that opened it. A single Lock (below)
    # serialises access so the shared connection stays consistent.
    database.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(database), check_same_thread=False)


_log = logging.getLogger(__name__)


class EventStore:
    """Append-only store of occupancy :class:`Event` rows."""

    def __init__(self, database: Path) -> None:
        self._conn = _connect(database)
        self._lock = threading.Lock()
        self._conn.executescript(_sql("schema.sql"))
        self._migrate()

    def _migrate(self) -> None:
        """Bring an older database up to the current schema, in place.

        ``CREATE TABLE IF NOT EXISTS`` does nothing to a table that already exists, so a
        database written before cameras were named would silently lack the column and every
        insert would fail. Migrating beats recreating: the history is the point.
        """
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(events)")}
        if "camera" not in columns:
            self._conn.executescript(_sql("add_event_camera.sql"))
            self._conn.commit()
        if "named_by" not in columns:
            self._conn.executescript(_sql("add_event_provenance.sql"))
            self._conn.commit()

    def record(self, event: Event) -> None:
        with self._lock:
            self._conn.execute(
                _sql("insert_event.sql"),
                (
                    event.timestamp,
                    event.name,
                    event.direction.value,
                    event.camera,
                    event.named_by,
                    event.natural,
                ),
            )
            self._conn.commit()

    def rename(self, old: str, new: str) -> int:
        """Correct a name throughout the history. Returns how many rows changed.

        The store is otherwise append-only, and this is the one deliberate exception: a
        misspelling was never a second person, and leaving both spellings in the record makes
        one person look like two who each came and went half the time.
        """
        with self._lock:
            cursor = self._conn.execute(_sql("rename_event_name.sql"), (new, old))
            self._conn.commit()
            return cursor.rowcount

    def rename_crossing(self, at: float, direction: str, name: str, window: float = 1.0) -> int:
        """Put a corrected name on the one crossing a sighting is about.

        The figures on the leaderboard come from this log, so a label corrected by hand has to
        reach it -- otherwise the totals keep whatever the system guessed at the time, and no
        amount of careful labelling would ever change them.

        Matched on the moment and the direction, because the sighting and the event were
        written by different parts of the system and share only those. A second either side:
        two crossings the same way within a second are the same passage.
        """
        with self._lock:
            cursor = self._conn.execute(
                _sql("rename_one_event.sql"), (name, direction, at - window, at + window)
            )
            self._conn.commit()
            return cursor.rowcount

    def passages(self) -> list[tuple[str, float, str]]:
        """Every crossing: who, when, which way. What the presence figures are derived from."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT name, timestamp, direction FROM events ORDER BY timestamp"
            ).fetchall()
        return [(name, float(at), direction) for name, at, direction in rows]

    def crossings(self, since: float, until: float) -> list[Crossing]:
        """Every crossing in a window, as the accounting needs it."""
        with self._lock:
            rows = self._conn.execute(_sql("crossings_between.sql"), (since, until)).fetchall()
        return [
            Crossing(
                timestamp=float(timestamp),
                name=name,
                direction=direction,
                camera=camera or "",
                named_by=named_by or "",
                natural=natural or "",
            )
            for timestamp, name, direction, camera, named_by, natural in rows
        ]

    def recent(self, limit: int = 50) -> list[tuple[float, str, str, str]]:
        with self._lock:
            rows = self._conn.execute(_sql("recent_events.sql"), (limit,)).fetchall()
        return [(float(ts), name, direction, camera or "") for ts, name, direction, camera in rows]

    def close(self) -> None:
        self._conn.close()


class PassageStore:
    """Every episode of a doorframe being covered -- what it looked like and what was decided.

    Separate from :class:`EventStore` because it records the *refusals* too. The events table
    holds only committed crossings, so a passage that was seen and rejected leaves no trace,
    and questions like "why are there fewer exits than entries" have no evidence behind them.
    The log that did hold them is rotated within hours.
    """

    def __init__(self, database: Path) -> None:
        self._conn = _connect(database)
        self._lock = threading.Lock()
        self._conn.executescript(_sql("passages_schema.sql"))

    def record(self, timestamp: float, camera: str, passage) -> None:
        """File one episode. Never raises: diagnostics must not stop the pipeline."""
        coverage = passage.coverage
        try:
            with self._lock:
                self._conn.execute(
                    _sql("insert_passage.sql"),
                    (
                        timestamp,
                        camera,
                        coverage.frames,
                        coverage.slices,
                        coverage.peak,
                        coverage.lag,
                        passage.person,
                        passage.direction.value if passage.direction else None,
                        int(passage.direction is not None and passage.person is not None),
                    ),
                )
                self._conn.commit()
        except Exception as exc:
            _log.error("could not record the passage: %s", exc)

    def refused(self, since: float, until: float) -> list[dict]:
        """Passages seen and not counted, which is where a missing exit usually is."""
        with self._lock:
            rows = self._conn.execute(_sql("refused_between.sql"), (since, until)).fetchall()
        return [
            {
                "timestamp": float(timestamp),
                "camera": camera,
                "frames": frames,
                "slices": slices,
                "peak": peak,
                "lag": lag,
                "person": person,
                "direction": direction,
            }
            for timestamp, camera, frames, slices, peak, lag, person, direction in rows
        ]

    def close(self) -> None:
        self._conn.close()


class SightingStore:
    """Append-only log of doorway sightings (motion-photo path): who was seen, when."""

    def __init__(self, database: Path) -> None:
        self._conn = _connect(database)
        self._lock = threading.Lock()
        self._conn.executescript(_sql("sightings_schema.sql"))

    def record(self, timestamp: float, name: str, clarity: float) -> None:
        with self._lock:
            self._conn.execute(_sql("insert_sighting.sql"), (timestamp, name, clarity))
            self._conn.commit()

    def recent(self, limit: int = 50) -> list[tuple[float, str, float]]:
        with self._lock:
            rows = self._conn.execute(_sql("recent_sightings.sql"), (limit,)).fetchall()
        return [(float(ts), name, float(clarity)) for ts, name, clarity in rows]

    def close(self) -> None:
        self._conn.close()
