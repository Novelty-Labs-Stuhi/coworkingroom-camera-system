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


def _frame(index: int) -> Frame:
    return Frame(timestamp=float(index), image=np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))


def test_nothing_is_examined_while_the_doorframe_is_clear() -> None:
    door = Doorframe()
    attention = Attention(door, pre_roll=4, linger=2)

    assert [attention.examine(_frame(i)) for i in range(6)] == [[], [], [], [], [], []]


def test_waking_up_replays_the_approach_first() -> None:
    """The recognisable face is before the box is touched, so those frames must be examined.

    Without this the models would only ever see somebody already at the doorframe -- often
    side-on or past it -- and every passage would be counted and nobody named.
    """
    door = Doorframe()
    attention = Attention(door, pre_roll=4, linger=0)

    for index in range(6):        # an empty corridor: nothing examined, frames kept
        assert attention.examine(_frame(index)) == []

    door.busy = True
    examined = attention.examine(_frame(6))

    # The last four approach frames, oldest first, then the frame that woke it.
    assert [frame.timestamp for frame in examined] == [2.0, 3.0, 4.0, 5.0, 6.0]


def test_the_approach_is_replayed_once_not_every_frame() -> None:
    door = Doorframe()
    attention = Attention(door, pre_roll=4, linger=0)
    for index in range(5):
        attention.examine(_frame(index))

    door.busy = True
    attention.examine(_frame(5))
    again = attention.examine(_frame(6))

    assert [frame.timestamp for frame in again] == [6.0]


def test_it_keeps_looking_briefly_after_the_doorframe_clears() -> None:
    """A track has to end on its own; cutting it off mid-stride loses the departure."""
    door = Doorframe()
    attention = Attention(door, pre_roll=2, linger=2)
    door.busy = True
    attention.examine(_frame(0))

    door.busy = False
    assert [f.timestamp for f in attention.examine(_frame(1))] == [1.0]
    assert [f.timestamp for f in attention.examine(_frame(2))] == [2.0]
    assert attention.examine(_frame(3)) == []


def test_the_buffer_holds_only_the_recent_approach() -> None:
    door = Doorframe()
    attention = Attention(door, pre_roll=3, linger=0)
    for index in range(50):
        attention.examine(_frame(index))

    door.busy = True
    examined = attention.examine(_frame(50))

    assert [frame.timestamp for frame in examined] == [47.0, 48.0, 49.0, 50.0]
