"""Tests for TOML config loading."""

from __future__ import annotations

from pathlib import Path

from stuhi_vision import config

_TOML = """
[source]
kind = "stream"
target = "http://cam/stream"

[doorway]
line_a = [10, 20]
line_b = [30, 40]
inside_side = "right"

[thresholds]
face_match = 0.4

[paths]
database = "data/x.db"
"""


def test_load_parses_all_sections(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(_TOML, encoding="utf-8")

    cfg = config.load(path)

    assert cfg.source.kind == "stream"
    assert cfg.source.target == "http://cam/stream"
    assert cfg.doorway.line_a == (10.0, 20.0)
    assert cfg.doorway.line_b == (30.0, 40.0)
    assert cfg.doorway.inside_side == "right"
    assert cfg.thresholds.face_match == 0.4
    assert cfg.thresholds.exit_similarity == 0.6  # default preserved
    assert cfg.thresholds.exit_margin == 0.05  # default preserved
    assert cfg.thresholds.min_track_age == 3  # default preserved
    assert cfg.thresholds.face_clarity_min == 0.5  # default preserved
    assert cfg.paths.database == Path("data/x.db")
    assert cfg.paths.gallery_dir == Path("gallery")  # default preserved
