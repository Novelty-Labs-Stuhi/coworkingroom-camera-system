"""The gate and the crop must be invisible: same coordinates, fewer model calls."""

from __future__ import annotations

import numpy as np

from stuhi_vision.domain import Box, Frame, TrackedPerson
from stuhi_vision.gating import MotionGate
from stuhi_vision.region import Region
from stuhi_vision.tracking import GatedTracker


class FakeTracker:
    """Records what it was asked to look at, and reports a fixed box."""

    def __init__(self, box: Box | None = None) -> None:
        self.calls: list[tuple[int, int]] = []
        self._box = box or Box(10, 20, 30, 40)

    def update(self, frame: Frame) -> list[TrackedPerson]:
        height, width = frame.image.shape[:2]
        self.calls.append((width, height))
        return [TrackedPerson(track_id=1, box=self._box)]


def _frame(value: int = 0) -> Frame:
    return Frame(timestamp=0.0, image=np.full((480, 640, 3), value, dtype=np.uint8))


def test_passes_through_with_no_gate_or_region() -> None:
    tracker = FakeTracker()
    people = GatedTracker(tracker).update(_frame())

    assert tracker.calls == [(640, 480)]
    assert people[0].box == Box(10, 20, 30, 40)


def test_a_still_frame_never_reaches_the_model() -> None:
    tracker = FakeTracker()
    gated = GatedTracker(tracker, gate=MotionGate(linger_frames=0))

    gated.update(_frame())  # first frame is always active
    calls_after_first = len(tracker.calls)
    for _ in range(3):
        # None, not [] -- an empty list would tell the doorway monitor that everyone
        # left, making it forget which side each track was on.
        assert gated.update(_frame()) is None

    assert len(tracker.calls) == calls_after_first  # the model was not called again


def test_movement_reaches_the_model_again() -> None:
    tracker = FakeTracker()
    gated = GatedTracker(tracker, gate=MotionGate(linger_frames=0))
    gated.update(_frame())
    gated.update(_frame())

    moved = _frame()
    moved.image[100:300, 100:300] = 255
    assert gated.update(moved) != []


def test_boxes_from_a_crop_come_back_in_full_frame_coordinates() -> None:
    tracker = FakeTracker(box=Box(0, 0, 10, 10))
    gated = GatedTracker(tracker, region_builder=lambda w, h: Region(100, 50, 500, 400))

    people = gated.update(_frame())

    assert tracker.calls == [(400, 350)]  # the model saw only the crop
    assert people[0].box == Box(100, 50, 110, 60)  # translated back for everyone else


def test_the_region_is_built_once_and_reused() -> None:
    built: list[tuple[int, int]] = []

    def builder(width: int, height: int) -> Region:
        built.append((width, height))
        return Region(0, 0, 320, 240)

    gated = GatedTracker(FakeTracker(), region_builder=builder)
    gated.update(_frame())
    gated.update(_frame(value=200))

    assert built == [(640, 480)]
