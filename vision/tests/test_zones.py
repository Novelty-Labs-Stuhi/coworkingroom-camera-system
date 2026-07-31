"""Hand-drawn zones, stored beside the config rather than inside it."""

from __future__ import annotations

import pytest

from stuhi_vision.zones import DrawnZone, ZoneStore


def test_corners_dragged_backwards_are_normalised() -> None:
    # A rectangle dragged up and to the left arrives reversed; stored as-is, every later
    # overlap test would be quietly false.
    zone = DrawnZone.from_corners(0.8, 0.9, 0.2, 0.1)

    assert zone.as_tuple() == (0.2, 0.1, 0.8, 0.9)


def test_a_zone_is_clamped_to_the_frame() -> None:
    zone = DrawnZone.from_corners(-0.5, -0.2, 1.5, 2.0)

    assert zone.as_tuple() == (0.0, 0.0, 1.0, 1.0)


def test_a_stray_click_is_refused() -> None:
    # Otherwise a mis-click would silently produce a zone nobody can overlap.
    with pytest.raises(ValueError, match="too small"):
        DrawnZone.from_corners(0.5, 0.5, 0.503, 0.9)


def test_saving_and_reloading_keeps_the_zone(tmp_path) -> None:
    store = ZoneStore(tmp_path)
    store.save("door-in", DrawnZone.from_corners(0.1, 0.0, 0.42, 1.0))

    reopened = ZoneStore(tmp_path)

    assert reopened.get("door-in").as_tuple() == (0.1, 0.0, 0.42, 1.0)


def test_a_camera_with_no_drawn_zone_falls_back_to_the_config(tmp_path) -> None:
    store = ZoneStore(tmp_path)
    store.save("door-in", DrawnZone.from_corners(0.1, 0.0, 0.4, 1.0))

    assert store.get("door-out") is None  # meaning: use whatever the config says


def test_each_camera_keeps_its_own_reference_frame(tmp_path) -> None:
    store = ZoneStore(tmp_path)

    first = store.reference_path("door-in")
    second = store.reference_path("door-out")

    assert first != second
    # Names come from the config, so they must not be trusted as filenames.
    assert "/" not in store.reference_path("odd/name..").name


def test_redrawing_replaces_rather_than_accumulates(tmp_path) -> None:
    store = ZoneStore(tmp_path)
    store.save("door-in", DrawnZone.from_corners(0.0, 0.0, 0.2, 1.0))
    store.save("door-in", DrawnZone.from_corners(0.1, 0.0, 0.5, 1.0))

    assert ZoneStore(tmp_path).get("door-in").as_tuple() == (0.1, 0.0, 0.5, 1.0)
    assert list(ZoneStore(tmp_path).all()) == ["door-in"]


def test_removing_a_zone_survives_a_reload(tmp_path) -> None:
    store = ZoneStore(tmp_path)
    store.save("door-in", DrawnZone.from_corners(0.4, 0.0, 0.6, 1.0))
    store.save("door-out", DrawnZone.from_corners(0.0, 0.0, 1.0, 1.0))

    assert store.remove("door-in") is True
    # Reloaded, because a removal that only happened in memory would come back on restart --
    # the zone would be gone from the page and still in use by the pipeline.
    reloaded = ZoneStore(tmp_path)
    assert reloaded.get("door-in") is None
    assert reloaded.get("door-out") is not None


def test_removing_a_zone_that_was_never_drawn_says_so(tmp_path) -> None:
    assert ZoneStore(tmp_path).remove("door-in") is False


def test_a_saved_zone_is_announced_so_a_running_camera_can_follow_it(tmp_path) -> None:
    """A zone is redrawn because the camera moved, so the count is wrong now, not at restart."""
    store = ZoneStore(tmp_path)
    heard: list = []
    store.watch(lambda camera, zone: heard.append((camera, zone)))

    zone = DrawnZone.from_corners(0.0, 0.0, 0.2, 1.0)
    store.save("door-in", zone)
    assert heard == [("door-in", zone)]

    store.remove("door-in")
    assert heard[-1] == ("door-in", None)   # None: fall back to the config's zone


def test_removing_a_zone_that_was_never_there_announces_nothing(tmp_path) -> None:
    store = ZoneStore(tmp_path)
    heard: list = []
    store.watch(lambda camera, zone: heard.append((camera, zone)))

    assert store.remove("door-in") is False
    assert heard == []
