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
    assert first.announce == "in"
    assert second.source.rotate == 0  # defaulted
    assert second.detector.inside_side == "right"
    assert second.announce == "out"


def test_the_single_camera_form_still_loads(tmp_path: Path) -> None:
    # An existing deployment must keep working unchanged: one [source] plus one [doorway]
    # is exactly one camera, which is what it always was.
    path = tmp_path / "one.toml"
    path.write_text(_TOML, encoding="utf-8")

    cfg = config.load(path)

    assert len(cfg.cameras) == 1
    assert cfg.cameras[0].name == "camera"
    assert cfg.camera.announce == "both"


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


_DOORFRAME_CAMERA = """
[[camera]]
name = "door-in"
target = "http://in/stream"
rotate = 180
zone = [0.0, 0.0, 0.30, 1.0]
edge = "left"
passing_means = "in"
min_height = 0.4
announce = "in"
"""


def test_a_camera_can_use_the_doorframe_rule_instead_of_a_line(tmp_path: Path) -> None:
    from stuhi_vision.threshold import ThresholdConfig

    path = tmp_path / "doorframe.toml"
    path.write_text(_DOORFRAME_CAMERA, encoding="utf-8")

    cfg = config.load(path)
    detector = cfg.camera.detector

    assert isinstance(detector, ThresholdConfig)
    assert detector.zone == (0.0, 0.0, 0.30, 1.0)
    assert detector.edge == "left"
    assert detector.min_height == 0.4
    # No line exists for this camera, so anything that draws one must be able to tell.
    assert cfg.camera.doorway is None


def _written(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_roles_decide_which_camera_commits(tmp_path: Path) -> None:
    """The live arrangement: the doorway camera counts, the room camera only names."""
    from types import SimpleNamespace

    from stuhi_vision.assembly import _committer
    from stuhi_vision.handlers import Doorkeeper, Identifier
    from stuhi_vision.witness import LeavingWitness

    cfg = config.load(
        _written(
            tmp_path,
            """
[[camera]]
name = "door-in"
target = "http://one/stream"
zone = [0.0, 0.0, 0.2, 1.0]
edge = "left"
passing_means = "in"
role = "count"

[[camera]]
name = "door-out"
target = "http://two/stream"
zone = [0.0, 0.0, 1.0, 1.0]
discriminator = "approach"
passing_means = "out"
role = "identify"
""",
        )
    )
    counter, namer = cfg.cameras
    assert (counter.role, namer.role) == ("count", "identify")

    shared = SimpleNamespace(ledger=object(), witness=LeavingWitness())
    assert isinstance(_committer(counter, object(), shared, 2), Doorkeeper)
    assert isinstance(_committer(namer, object(), shared, 2), Identifier)


def test_a_camera_counts_unless_told_otherwise(tmp_path: Path) -> None:
    cfg = config.load(
        _written(
            tmp_path,
            """
[[camera]]
name = "only"
target = "http://one/stream"
zone = [0.0, 0.0, 0.2, 1.0]
""",
        )
    )
    # A single-camera deployment must keep working: nobody would be counting otherwise.
    assert cfg.cameras[0].role == "count"


def test_a_camera_can_be_told_to_read_the_box_by_pixel_change(tmp_path: Path) -> None:
    """The coverage rule, and its own thresholds, come from the camera's own block."""
    from stuhi_vision.assembly import _monitor
    from stuhi_vision.passage import PassageMonitor
    from stuhi_vision.threshold import ThresholdMonitor
    from stuhi_vision.zones import ZoneStore

    cfg = config.load(
        _written(
            tmp_path,
            """
[[camera]]
name = "door-in"
target = "http://one/stream"
zone = [0.0, 0.0, 0.12, 1.0]
edge = "left"
passing_means = "in"
rule = "coverage"

[camera.coverage]
slices = 6
covered = 0.3

[[camera]]
name = "door-out"
target = "http://two/stream"
zone = [0.0, 0.0, 1.0, 1.0]
discriminator = "approach"
""",
        )
    )
    pixels, tracks = cfg.cameras
    assert (pixels.rule, tracks.rule) == ("coverage", "tracks")
    assert (pixels.coverage.slices, pixels.coverage.covered) == (6, 0.3)
    # Defaults elsewhere, so a camera that says nothing about pixels keeps the old rule.
    assert tracks.coverage.slices == 5

    zones = ZoneStore(tmp_path / "zones")
    watching, attention = _monitor(pixels, zones)
    plain, none = _monitor(tracks, zones)

    assert isinstance(watching, PassageMonitor)
    assert attention is not None   # the pixels also decide when the detector wakes
    assert isinstance(plain, ThresholdMonitor)
    assert none is None


def test_a_camera_can_set_its_own_motion_gate(tmp_path: Path) -> None:
    """The right gate belongs to the view: the dark corridor needs none, the lit room can."""
    from stuhi_vision.assembly import _motion

    cfg = config.load(
        _written(
            tmp_path,
            """
[[camera]]
name = "corridor"
target = "http://one/stream"
zone = [0.0, 0.0, 0.12, 1.0]
motion_min_fraction = 0.0

[[camera]]
name = "room"
target = "http://two/stream"
zone = [0.0, 0.0, 1.0, 1.0]

[performance]
motion_min_fraction = 0.004
""",
        )
    )
    corridor, room = cfg.cameras
    # Stated zero must survive: falling back on it would re-enable the gate that suppressed
    # the frames holding a person in the dark, which is a silent loss of every passage.
    assert _motion(corridor, cfg.performance) == 0.0
    assert _motion(room, cfg.performance) == 0.004


def test_a_camera_can_set_its_own_persistence_gate(tmp_path: Path) -> None:
    """With the coverage rule the pixels already prove the passage; the gate is redundant.

    Measured live, it was worse than redundant: two real walks out were read correctly and
    thrown away as "seen 1 frames", because a dark corridor fragments track ids.
    """
    from stuhi_vision.assembly import _persistence

    cfg = config.load(
        _written(
            tmp_path,
            """
[[camera]]
name = "door-in"
target = "http://one/stream"
zone = [0.0, 0.0, 0.12, 1.0]
rule = "coverage"
min_track_age = 1

[[camera]]
name = "door-out"
target = "http://two/stream"
zone = [0.0, 0.0, 1.0, 1.0]

[thresholds]
min_track_age = 2
""",
        )
    )
    pixels, tracks = cfg.cameras
    assert _persistence(pixels, cfg.thresholds) == 1
    assert _persistence(tracks, cfg.thresholds) == 2
