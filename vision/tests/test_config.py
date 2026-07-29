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

    assert cfg.camera.source.kind == "stream"
    assert cfg.camera.source.target == "http://cam/stream"
    assert cfg.camera.doorway.line_a == (10.0, 20.0)
    assert cfg.camera.doorway.line_b == (30.0, 40.0)
    assert cfg.camera.doorway.inside_side == "right"
    assert cfg.thresholds.face_match == 0.4
    assert cfg.thresholds.exit_similarity == 0.6  # default preserved
    assert cfg.thresholds.exit_margin == 0.05  # default preserved
    assert cfg.thresholds.min_track_age == 2  # default preserved
    assert cfg.performance.detect_imgsz == 320  # default preserved
    assert cfg.performance.detect_model == "yolov8n.pt"  # default preserved
    assert cfg.thresholds.face_margin == 0.05  # default preserved
    assert cfg.paths.database == Path("data/x.db")
    assert cfg.paths.gallery_dir == Path("gallery")  # default preserved
    assert cfg.paths.review_dir == Path("data/review")  # default preserved


_TWO_CAMERAS = """
[[camera]]
name = "door-in"
target = "http://in/stream"
rotate = 180
line_a = [300, 0]
line_b = [300, 480]
inside_side = "left"
announce = "in"

[[camera]]
name = "door-out"
target = "http://out/stream"
line_a = [160, 0]
line_b = [160, 240]
inside_side = "right"
announce = "out"
"""


def test_two_cameras_each_keep_their_own_view(tmp_path: Path) -> None:
    # The doorway line belongs to the camera, not the room: each has its own pixels and its
    # own idea of which side is inside, and inside_side flips between opposed cameras.
    path = tmp_path / "two.toml"
    path.write_text(_TWO_CAMERAS, encoding="utf-8")

    cfg = config.load(path)

    assert [camera.name for camera in cfg.cameras] == ["door-in", "door-out"]
    first, second = cfg.cameras
    assert first.source.target == "http://in/stream"
    assert first.source.rotate == 180
    assert first.doorway.announce == "in"
    assert second.source.rotate == 0  # defaulted
    assert second.doorway.inside_side == "right"
    assert second.doorway.announce == "out"


def test_the_single_camera_form_still_loads(tmp_path: Path) -> None:
    # An existing deployment must keep working unchanged: one [source] plus one [doorway]
    # is exactly one camera, which is what it always was.
    path = tmp_path / "one.toml"
    path.write_text(_TOML, encoding="utf-8")

    cfg = config.load(path)

    assert len(cfg.cameras) == 1
    assert cfg.cameras[0].name == "camera"
    assert cfg.camera.doorway.announce == "both"


def test_duplicate_camera_names_are_refused(tmp_path: Path) -> None:
    # Names identify a camera in events and in the UI, so duplicates would make the record
    # of which camera saw what meaningless.
    path = tmp_path / "dupes.toml"
    path.write_text(
        _TWO_CAMERAS.replace('name = "door-out"', 'name = "door-in"'), encoding="utf-8"
    )

    try:
        config.load(path)
    except ValueError as error:
        assert "unique" in str(error)
        return
    raise AssertionError("duplicate camera names should be refused")


def test_a_config_with_no_camera_at_all_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "empty.toml"
    path.write_text("[thresholds]\nface_match = 0.4\n", encoding="utf-8")

    try:
        config.load(path)
    except ValueError as error:
        assert "camera" in str(error)
        return
    raise AssertionError("a config with no camera should be refused")
