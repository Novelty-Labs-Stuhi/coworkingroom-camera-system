"""Durable event log, backed by SQLite from the standard library.

The ledger keeps live occupancy in memory; this records every entry/exit so the
history survives a restart and can be queried later. SQL lives in ``.sql`` files
(never glued into Python strings) and is always parameterised.
"""

from __future__ import annotations

import sqlite3
import threading
from importlib import resources
from pathlib import Path

from .domain import Event


def _sql(name: str) -> str:
    return resources.files(f"{__package__}.sql").joinpath(name).read_text(encoding="utf-8")


def _connect(database: Path) -> sqlite3.Connection:
    # check_same_thread=False: the FastAPI server touches this connection from its
    # threadpool workers, not just the thread that opened it. A single Lock (below)
    # serialises access so the shared connection stays consistent.
    database.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(database), check_same_thread=False)


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

    def record(self, event: Event) -> None:
        with self._lock:
            self._conn.execute(
                _sql("insert_event.sql"),
                (event.timestamp, event.name, event.direction.value, event.camera),
            )
            self._conn.commit()

    def recent(self, limit: int = 50) -> list[tuple[float, str, str, str]]:
        with self._lock:
            rows = self._conn.execute(_sql("recent_events.sql"), (limit,)).fetchall()
        return [(float(ts), name, direction, camera or "") for ts, name, direction, camera in rows]

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
