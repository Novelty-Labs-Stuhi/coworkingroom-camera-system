"""The motion gate: cheap frames in, expensive work only when something changes."""

from __future__ import annotations

import numpy as np

from stuhi_vision.gating import MotionGate


def _blank(value: int = 0) -> np.ndarray:
    return np.full((480, 640, 3), value, dtype=np.uint8)


def _with_blob(size: int = 200, value: int = 255) -> np.ndarray:
    image = _blank()
    image[100 : 100 + size, 100 : 100 + size] = value
    return image


def test_first_frame_is_always_active() -> None:
    # No reference to compare against yet, so the only safe answer is "look at it".
    assert MotionGate().is_active(_blank()) is True


def test_a_still_scene_goes_quiet_once_the_linger_expires() -> None:
    gate = MotionGate(linger_frames=2)
    gate.is_active(_blank())
    for _ in range(3):  # burn the first frame's linger
        gate.is_active(_blank())

    assert gate.is_active(_blank()) is False


def test_a_large_change_reactivates_the_gate() -> None:
    gate = MotionGate(linger_frames=0)
    gate.is_active(_blank())
    gate.is_active(_blank())

    assert gate.is_active(_with_blob()) is True
    assert gate.last_fraction > 0


def test_linger_keeps_the_gate_open_after_movement_stops() -> None:
    gate = MotionGate(linger_frames=3)
    gate.is_active(_blank())
    gate.is_active(_with_blob())  # movement

    # A person pausing mid-stride must not flicker the gate shut and break the track.
    still = _with_blob()
    assert [gate.is_active(still) for _ in range(3)] == [True, True, True]
    assert gate.is_active(still) is False


def test_sensor_noise_does_not_trip_the_gate() -> None:
    rng = np.random.default_rng(1234)
    gate = MotionGate(min_fraction=0.02, linger_frames=0)
    base = _blank(120)
    gate.is_active(base)
    gate.is_active(base)

    noisy = np.clip(base.astype(np.int16) + rng.integers(-6, 7, base.shape), 0, 255)
    assert gate.is_active(noisy.astype(np.uint8)) is False
