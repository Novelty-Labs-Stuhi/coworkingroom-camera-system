"""Durable event log, backed by SQLite from the standard library.

The ledger keeps live occupancy in memory; this records every entry/exit so the
history survives a restart and can be queried later. SQL lives in ``.sql`` files
(never glued into Python strings) and is always parameterised.
"""

from __future__ import annotations

import sqlite3
from importlib import resources
from pathlib import Path

from .domain import Event


def _sql(name: str) -> str:
    return resources.files(f"{__package__}.sql").joinpath(name).read_text(encoding="utf-8")


class EventStore:
    """Append-only store of occupancy :class:`Event` rows."""

    def __init__(self, database: Path) -> None:
        database.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(database))
        self._conn.executescript(_sql("schema.sql"))

    def record(self, event: Event) -> None:
        self._conn.execute(
            _sql("insert_event.sql"),
            (event.timestamp, event.name, event.direction.value),
        )
        self._conn.commit()

    def recent(self, limit: int = 50) -> list[tuple[float, str, str]]:
        rows = self._conn.execute(_sql("recent_events.sql"), (limit,)).fetchall()
        return [(float(ts), name, direction) for ts, name, direction in rows]

    def close(self) -> None:
        self._conn.close()


class SightingStore:
    """Append-only log of doorway sightings (motion-photo path): who was seen, when."""

    def __init__(self, database: Path) -> None:
        database.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(database))
        self._conn.executescript(_sql("sightings_schema.sql"))

    def record(self, timestamp: float, name: str, clarity: float) -> None:
        self._conn.execute(_sql("insert_sighting.sql"), (timestamp, name, clarity))
        self._conn.commit()

    def recent(self, limit: int = 50) -> list[tuple[float, str, float]]:
        rows = self._conn.execute(_sql("recent_sightings.sql"), (limit,)).fetchall()
        return [(float(ts), name, float(clarity)) for ts, name, clarity in rows]

    def close(self) -> None:
        self._conn.close()
