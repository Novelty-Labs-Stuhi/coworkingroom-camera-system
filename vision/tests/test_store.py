"""Tests for the SQLite event store."""

from __future__ import annotations

from pathlib import Path

from stuhi_vision.domain import Direction, Event
from stuhi_vision.store import EventStore


def test_record_and_read_recent(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "occ.db")
    try:
        store.record(Event(1.0, "alice", Direction.IN, "door-in"))
        store.record(Event(2.0, "bob", Direction.IN, "door-in"))
        store.record(Event(3.0, "alice", Direction.OUT, "door-out"))
        recent = store.recent(limit=10)
    finally:
        store.close()

    # newest first, and each event says which camera saw it
    assert [(name, direction, camera) for _ts, name, direction, camera in recent] == [
        ("alice", "out", "door-out"),
        ("bob", "in", "door-in"),
        ("alice", "in", "door-in"),
    ]


def test_an_older_database_gains_the_camera_column(tmp_path: Path) -> None:
    # CREATE TABLE IF NOT EXISTS does nothing to an existing table, so without a migration
    # a database written before cameras were named would fail every insert. The history is
    # the point, so it is migrated rather than recreated.
    import sqlite3

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(
        "CREATE TABLE events ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " timestamp REAL NOT NULL,"
        " name TEXT NOT NULL,"
        " direction TEXT NOT NULL CHECK (direction IN ('in', 'out')));"
    )
    old.execute("INSERT INTO events (timestamp, name, direction) VALUES (1.0, 'old', 'in')")
    old.commit()
    old.close()

    store = EventStore(path)
    try:
        store.record(Event(2.0, "new", Direction.IN, "door-in"))
        recent = store.recent(limit=10)
    finally:
        store.close()

    assert [(name, camera) for _ts, name, _direction, camera in recent] == [
        ("new", "door-in"),
        ("old", ""),  # the pre-existing row survives, with no camera
    ]


def test_creates_parent_directory(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "occ.db"
    store = EventStore(nested)
    store.close()
    assert nested.exists()
