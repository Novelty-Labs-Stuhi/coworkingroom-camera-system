"""Tests for the SQLite event store."""

from __future__ import annotations

from pathlib import Path

from stuhi_vision.domain import Direction, Event
from stuhi_vision.store import EventStore


def test_record_and_read_recent(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "occ.db")
    try:
        store.record(Event(1.0, "alice", Direction.IN))
        store.record(Event(2.0, "bob", Direction.IN))
        store.record(Event(3.0, "alice", Direction.OUT))
        recent = store.recent(limit=10)
    finally:
        store.close()

    # newest first
    assert [(name, direction) for _ts, name, direction in recent] == [
        ("alice", "out"),
        ("bob", "in"),
        ("alice", "in"),
    ]


def test_creates_parent_directory(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "occ.db"
    store = EventStore(nested)
    store.close()
    assert nested.exists()
