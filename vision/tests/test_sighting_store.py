"""Tests for the doorway sighting store."""

from __future__ import annotations

from pathlib import Path

from stuhi_vision.store import SightingStore


def test_record_and_read_recent(tmp_path: Path) -> None:
    store = SightingStore(tmp_path / "sightings.db")
    try:
        store.record(1.0, "ilari", 0.8)
        store.record(2.0, "unknown", 0.0)
        recent = store.recent(limit=10)
    finally:
        store.close()

    # newest first
    assert [(name, clarity) for _ts, name, clarity in recent] == [("unknown", 0.0), ("ilari", 0.8)]
