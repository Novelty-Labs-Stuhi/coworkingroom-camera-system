"""Tests for doorway line-crossing and direction (pure geometry, no models)."""

from __future__ import annotations

import numpy as np

from stuhi_vision.config import DoorwayConfig
from stuhi_vision.domain import Box, Direction, Frame, TrackedPerson
from stuhi_vision.doorway import DoorwayMonitor


def _frame() -> Frame:
    return Frame(timestamp=0.0, image=np.zeros((200, 200, 3), dtype=np.uint8))


def _person(track_id: int, foot_y: float) -> TrackedPerson:
    # A 20px-wide box centred at x=100 whose bottom edge (foot) sits at foot_y.
    return TrackedPerson(track_id, Box(90, foot_y - 10, 110, foot_y))


def _monitor() -> DoorwayMonitor:
    # Horizontal threshold at y=100; "inside" is the y>100 (lower) half.
    return DoorwayMonitor(DoorwayConfig(line_a=(0, 100), line_b=(200, 100), inside_side="left"))


def test_first_frame_never_crosses() -> None:
    monitor = _monitor()
    assert monitor.update([_person(1, foot_y=90)], _frame()) == []


def test_moving_inward_is_an_entry() -> None:
    monitor = _monitor()
    monitor.update([_person(1, foot_y=90)], _frame())  # above the line (outside)
    crossings = monitor.update([_person(1, foot_y=110)], _frame())  # below (inside)
    assert len(crossings) == 1
    assert crossings[0].direction is Direction.IN


def test_moving_outward_is_an_exit() -> None:
    monitor = _monitor()
    monitor.update([_person(1, foot_y=110)], _frame())  # inside
    crossings = monitor.update([_person(1, foot_y=90)], _frame())  # outside
    assert len(crossings) == 1
    assert crossings[0].direction is Direction.OUT


def test_no_crossing_when_staying_on_one_side() -> None:
    monitor = _monitor()
    monitor.update([_person(1, foot_y=90)], _frame())
    assert monitor.update([_person(1, foot_y=80)], _frame()) == []
