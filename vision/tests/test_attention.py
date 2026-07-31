"""The detector sleeps until the doorframe is busy -- and then sees the approach too."""

from __future__ import annotations

import numpy as np

from stuhi_vision.attention import Attention
from stuhi_vision.domain import Frame

WIDTH, HEIGHT = 320, 240


class Doorframe:
    """Coverage on demand, standing in for the pixel work."""

    def __init__(self) -> None:
        self.busy = False
        self.episodes: list = []

    def update(self, image):
        return self.episodes.pop(0) if self.episodes else None


# A tenth of a second apart, so a pre-roll in seconds maps to a countable number of frames.
INTERVAL = 0.1


def _frame(index: int) -> Frame:
    return Frame(
        timestamp=index * INTERVAL, image=np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    )


def test_nothing_is_examined_while_the_doorframe_is_clear() -> None:
    door = Doorframe()
    attention = Attention(door, pre_roll_seconds=0.4, linger_seconds=0.2)

    assert [attention.examine(_frame(i)) for i in range(6)] == [[], [], [], [], [], []]


def test_waking_up_replays_the_approach_first() -> None:
    """The recognisable face is before the box is touched, so those frames must be examined.

    Without this the models would only ever see somebody already at the doorframe -- often
    side-on or past it -- and every passage would be counted and nobody named.
    """
    door = Doorframe()
    # Not a whole number of frames: a real pre-roll never lands exactly on a frame boundary,
    # and a test that does is testing floating point rather than behaviour.
    attention = Attention(door, pre_roll_seconds=0.45, linger_seconds=0.0)

    for index in range(6):        # an empty corridor: nothing examined, frames kept
        assert attention.examine(_frame(index)) == []

    door.busy = True
    examined = attention.examine(_frame(6))

    # The last 0.45 s of approach, oldest first, then the frame that woke it.
    assert [round(frame.timestamp, 2) for frame in examined] == [0.2, 0.3, 0.4, 0.5, 0.6]


def test_the_approach_is_replayed_once_not_every_frame() -> None:
    door = Doorframe()
    attention = Attention(door, pre_roll_seconds=0.4, linger_seconds=0.0)
    for index in range(5):
        attention.examine(_frame(index))

    door.busy = True
    attention.examine(_frame(5))
    again = attention.examine(_frame(6))

    assert [round(frame.timestamp, 2) for frame in again] == [0.6]


def test_it_keeps_looking_briefly_after_the_doorframe_clears() -> None:
    """A track has to end on its own; cutting it off mid-stride loses the departure."""
    door = Doorframe()
    attention = Attention(door, pre_roll_seconds=0.2, linger_seconds=0.2)
    door.busy = True
    attention.examine(_frame(0))

    door.busy = False
    assert [round(f.timestamp, 2) for f in attention.examine(_frame(1))] == [0.1]
    assert [round(f.timestamp, 2) for f in attention.examine(_frame(2))] == [0.2]
    assert attention.examine(_frame(4)) == []


def test_only_the_recent_approach_is_replayed_however_long_it_was_quiet() -> None:
    door = Doorframe()
    attention = Attention(door, pre_roll_seconds=0.35, linger_seconds=0.0)
    for index in range(50):
        attention.examine(_frame(index))

    door.busy = True
    examined = attention.examine(_frame(50))

    assert [round(frame.timestamp, 2) for frame in examined] == [4.7, 4.8, 4.9, 5.0]


def test_the_kept_approach_is_bounded_by_frames_as_well_as_seconds() -> None:
    """Waking costs a detection per replayed frame, so a fast camera must not run away."""
    door = Doorframe()
    attention = Attention(door, pre_roll_seconds=60.0, linger_seconds=0.0, most_frames=5)
    for index in range(50):
        attention.examine(_frame(index))

    door.busy = True
    assert len(attention.examine(_frame(50))) == 6   # five kept, plus the waking frame
