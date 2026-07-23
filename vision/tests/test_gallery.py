"""Tests for the disk-backed face gallery."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from stuhi_vision.recognition.gallery import FaceGallery


def test_add_save_load_roundtrip(tmp_path: Path) -> None:
    gallery = FaceGallery()
    gallery.add("alice", np.array([1.0, 0.0, 0.0], dtype=np.float32))
    gallery.add("alice", np.array([0.9, 0.1, 0.0], dtype=np.float32))
    gallery.add("bob", np.array([0.0, 1.0, 0.0], dtype=np.float32))
    gallery.save(tmp_path)

    loaded = FaceGallery.load(tmp_path)
    assert loaded.names == ["alice", "bob"]

    match = loaded.match(np.array([0.95, 0.05, 0.0], dtype=np.float32), threshold=0.5)
    assert match is not None and match.name == "alice"


def test_load_missing_directory_is_empty(tmp_path: Path) -> None:
    gallery = FaceGallery.load(tmp_path / "does-not-exist")
    assert gallery.names == []


def test_match_below_threshold_returns_none(tmp_path: Path) -> None:
    gallery = FaceGallery()
    gallery.add("alice", np.array([1.0, 0.0], dtype=np.float32))
    assert gallery.match(np.array([0.0, 1.0], dtype=np.float32), threshold=0.5) is None
