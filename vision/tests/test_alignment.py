"""Noticing a moved camera, so a hand-drawn zone can be redrawn before it misleads."""

from __future__ import annotations

import numpy as np

from stuhi_vision.alignment import DriftWatch


def _scene(shift_x: int = 0, shift_y: int = 0, width: int = 640, height: int = 480):
    """A synthetic room with enough structure for phase correlation to lock onto."""
    rng = np.random.default_rng(7)
    image = rng.integers(20, 60, (height, width, 3), dtype=np.uint8)
    # A few strong features, as a real doorway has: edges, a bright opening, a dark frame.
    for x, w, value in [(60, 40, 230), (300, 120, 200), (500, 30, 240)]:
        image[:, x : x + w] = value
    for y, h, value in [(100, 30, 90), (350, 40, 210)]:
        image[y : y + h, :] = value
    return np.roll(np.roll(image, shift_x, axis=1), shift_y, axis=0)


def _watch(tmp_path, **overrides) -> DriftWatch:
    return DriftWatch(tmp_path / "reference.jpg", **overrides)


def test_no_reference_means_no_opinion(tmp_path) -> None:
    watch = _watch(tmp_path)
    assert watch.has_reference is False
    assert watch.check(_scene()) is None
    assert watch.has_moved is False


def test_an_unmoved_camera_reads_as_no_drift(tmp_path) -> None:
    watch = _watch(tmp_path)
    watch.remember(_scene())

    drift = watch.check(_scene())

    assert drift is not None
    assert drift.magnitude < 2.0
    assert watch.has_moved is False


def test_a_shifted_view_is_measured_in_pixels(tmp_path) -> None:
    watch = _watch(tmp_path, tolerance_px=10.0, confirmations=2)
    watch.remember(_scene())

    drift = watch.check(_scene(shift_x=40))

    # The magnitude is what makes a warning actionable: "shifted 40 px", not a score.
    assert drift.magnitude > 20
    assert abs(drift.shift_x) > 20


def test_one_bad_reading_is_not_enough(tmp_path) -> None:
    # A single frame can disagree from noise or auto-exposure; that is not a moved camera.
    watch = _watch(tmp_path, tolerance_px=10.0, confirmations=3)
    watch.remember(_scene())

    watch.check(_scene(shift_x=40))

    assert watch.has_moved is False


def test_sustained_movement_is_reported(tmp_path) -> None:
    watch = _watch(tmp_path, tolerance_px=10.0, confirmations=3)
    watch.remember(_scene())

    for _ in range(3):
        watch.check(_scene(shift_x=40))

    assert watch.has_moved is True


def test_going_back_to_normal_clears_the_count(tmp_path) -> None:
    watch = _watch(tmp_path, tolerance_px=10.0, confirmations=3)
    watch.remember(_scene())
    watch.check(_scene(shift_x=40))
    watch.check(_scene(shift_x=40))

    watch.check(_scene())  # steady again: whatever it was, it was not a moved camera

    assert watch.has_moved is False


def test_acknowledging_stops_it_repeating(tmp_path) -> None:
    # The camera is still moved and the zone still wrong, but saying so every frame is noise.
    watch = _watch(tmp_path, tolerance_px=10.0, confirmations=2)
    watch.remember(_scene())
    watch.check(_scene(shift_x=40))
    watch.check(_scene(shift_x=40))
    assert watch.has_moved is True

    watch.acknowledge()

    assert watch.has_moved is False


def test_the_reference_survives_a_restart(tmp_path) -> None:
    _watch(tmp_path).remember(_scene())

    reopened = _watch(tmp_path, tolerance_px=10.0, confirmations=1)

    assert reopened.has_reference is True
    assert reopened.check(_scene()).magnitude < 2.0


def test_a_different_resolution_is_not_treated_as_drift(tmp_path) -> None:
    # Swapping a camera for one with another sensor is not a knock; it needs recalibrating
    # outright, and reporting a pixel shift would be meaningless.
    watch = _watch(tmp_path)
    watch.remember(_scene())

    assert watch.check(_scene(width=320, height=240)) is None


def test_redrawing_adopts_the_new_view(tmp_path) -> None:
    watch = _watch(tmp_path, tolerance_px=10.0, confirmations=1)
    watch.remember(_scene())
    watch.check(_scene(shift_x=40))
    assert watch.has_moved is True

    watch.remember(_scene(shift_x=40))  # zone redrawn on the new view

    assert watch.has_moved is False
    assert watch.check(_scene(shift_x=40)).magnitude < 2.0
