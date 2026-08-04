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


def _stamps(examined) -> list[float]:
    """The moments handed to the models. `examine` returns frames paired with the doorframe's
    state on each, so the timestamp is one level in."""
    return [round(seen.frame.timestamp, 2) for seen in examined]


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
    assert _stamps(examined) == [0.2, 0.3, 0.4, 0.5, 0.6]


def test_the_approach_is_replayed_once_not_every_frame() -> None:
    door = Doorframe()
    attention = Attention(door, pre_roll_seconds=0.4, linger_seconds=0.0)
    for index in range(5):
        attention.examine(_frame(index))

    door.busy = True
    attention.examine(_frame(5))
    again = attention.examine(_frame(6))

    assert _stamps(again) == [0.6]


def test_it_keeps_looking_briefly_after_the_doorframe_clears() -> None:
    """A track has to end on its own; cutting it off mid-stride loses the departure."""
    door = Doorframe()
    attention = Attention(door, pre_roll_seconds=0.2, linger_seconds=0.2)
    door.busy = True
    attention.examine(_frame(0))

    door.busy = False
    assert _stamps(attention.examine(_frame(1))) == [0.1]
    assert _stamps(attention.examine(_frame(2))) == [0.2]
    assert attention.examine(_frame(4)) == []


def test_only_the_recent_approach_is_replayed_however_long_it_was_quiet() -> None:
    door = Doorframe()
    attention = Attention(door, pre_roll_seconds=0.35, linger_seconds=0.0)
    for index in range(50):
        attention.examine(_frame(index))

    door.busy = True
    examined = attention.examine(_frame(50))

    assert _stamps(examined) == [4.7, 4.8, 4.9, 5.0]


def test_the_kept_approach_is_bounded_by_frames_as_well_as_seconds() -> None:
    """Waking costs a detection per replayed frame, so a fast camera must not run away."""
    door = Doorframe()
    attention = Attention(door, pre_roll_seconds=60.0, linger_seconds=0.0, most_frames=5)
    for index in range(50):
        attention.examine(_frame(index))

    door.busy = True
    # Five frames in total, the newest of which is the one that woke it: the cap is on what
    # gets replayed, which is what costs a detection each.
    assert len(attention.examine(_frame(50))) == 5


def test_somebody_arriving_right_behind_still_gets_an_approach() -> None:
    """The buffer must keep filling while the first person is being watched.

    It used to be cleared on waking and not refilled until the box was quiet again, so a
    second person a couple of seconds behind the first was examined only once they were
    already on the doorframe -- which is exactly where their face cannot be seen.
    """
    door = Doorframe()
    attention = Attention(door, pre_roll_seconds=0.45, linger_seconds=0.0)

    for index in range(5):
        attention.examine(_frame(index))

    door.busy = True                       # the first person arrives
    attention.examine(_frame(5))
    for index in range(6, 10):             # ...and is watched across the doorframe
        attention.examine(_frame(index))
    door.busy = False
    attention.examine(_frame(10))

    door.busy = True                       # the second, moments later
    examined = attention.examine(_frame(11))

    # Only what the models have not already seen, and nothing replayed twice.
    assert _stamps(examined) == [1.1]
    assert attention.examine(_frame(12)) != []


def test_the_approach_of_a_second_person_is_replayed_when_it_was_missed() -> None:
    door = Doorframe()
    attention = Attention(door, pre_roll_seconds=0.45, linger_seconds=0.0)

    door.busy = True                       # first person
    attention.examine(_frame(0))
    door.busy = False
    for index in range(1, 8):              # a quiet gap: frames kept, models idle
        attention.examine(_frame(index))

    door.busy = True                       # second person, with an approach of their own
    examined = attention.examine(_frame(8))

    assert _stamps(examined) == [0.4, 0.5, 0.6, 0.7, 0.8]


# --- the backwards, chunked replay -----------------------------------------------------------


def test_each_frame_carries_the_coverage_it_had_not_the_coverage_now() -> None:
    """The distinction the whole direction rests on.

    The replay happens once the box is covered, so asking the doorway now would say "covered"
    for every frame of the approach -- collapsing "seen before the covering" and "seen after it"
    into one answer. Each frame therefore carries its own state.
    """
    door = Doorframe()
    attention = Attention(door, pre_roll_seconds=0.45, linger_seconds=0.0)
    for index in range(5):
        attention.examine(_frame(index))

    door.busy = True
    examined = attention.examine(_frame(5))

    assert [seen.covered for seen in examined] == [False, False, False, False, True]
    assert examined[-1].frame.timestamp == 0.5   # the frame that woke it, and the only covered one


def test_waking_hands_over_one_chunk_not_the_whole_buffer() -> None:
    """A close passage should cost one chunk of detections, not the worst case every time."""
    door = Doorframe()
    attention = Attention(door, pre_roll_seconds=60.0, linger_seconds=0.0, chunk_frames=4)
    for index in range(40):
        attention.examine(_frame(index))

    door.busy = True
    examined = attention.examine(_frame(40))

    # Four frames of approach, then the one that woke it. Not forty.
    assert _stamps(examined) == [3.6, 3.7, 3.8, 3.9, 4.0]


def test_asking_again_reaches_one_chunk_further_back() -> None:
    """For the passage the newest chunk could not settle: pay for more only then."""
    door = Doorframe()
    attention = Attention(door, pre_roll_seconds=60.0, linger_seconds=0.0, chunk_frames=3)
    for index in range(20):
        attention.examine(_frame(index))

    door.busy = True
    attention.examine(_frame(20))          # 1.7, 1.8, 1.9 + the waking frame 2.0

    assert _stamps(attention.earlier()) == [1.4, 1.5, 1.6]
    assert _stamps(attention.earlier()) == [1.1, 1.2, 1.3]
    # ...and each chunk is still oldest-first, so tracking sees time moving forwards.


def test_reaching_back_stops_at_the_end_of_what_was_kept() -> None:
    door = Doorframe()
    attention = Attention(
        door, pre_roll_seconds=60.0, linger_seconds=0.0, most_frames=6, chunk_frames=3
    )
    for index in range(20):
        attention.examine(_frame(index))

    door.busy = True
    attention.examine(_frame(20))
    attention.earlier()

    # Six frames were kept; the wake and one chunk have used them. There is nothing before.
    assert attention.earlier() == []


def test_reaching_back_before_anything_woke_gives_nothing() -> None:
    door = Doorframe()
    attention = Attention(door, pre_roll_seconds=0.5, linger_seconds=0.0)
    for index in range(5):
        attention.examine(_frame(index))

    assert attention.earlier() == []


def test_reaching_back_does_not_go_past_the_pre_roll() -> None:
    """The pre-roll bounds how far a question may reach, however many frames are kept."""
    door = Doorframe()
    attention = Attention(
        door, pre_roll_seconds=0.35, linger_seconds=0.0, most_frames=200, chunk_frames=2
    )
    for index in range(30):
        attention.examine(_frame(index))

    door.busy = True
    attention.examine(_frame(30))          # 2.8, 2.9 + the waking frame 3.0

    # A 0.35 s reach from a 3.0 s wake stops at 2.65, so only 2.7 is left to give.
    assert _stamps(attention.earlier()) == [2.7]
    assert attention.earlier() == []


def test_one_waking_cannot_spend_more_than_its_budget() -> None:
    """The fault that took the pipeline down twice, now a rule rather than a hope.

    Unbounded, an awkward passage replayed the whole buffer -- over a minute of detection in
    one call. The frame reader is on that thread, so it read nothing for that minute, the
    heartbeat stopped, and the watchdog killed the pipeline as stuck.
    """
    door = Doorframe()
    attention = Attention(
        door, pre_roll_seconds=60.0, linger_seconds=0.0, most_frames=200,
        chunk_frames=4, most_replayed=10,
    )
    for index in range(100):
        attention.examine(_frame(index))

    door.busy = True
    spent = len(attention.examine(_frame(100))) - 1     # less the waking frame itself
    while True:
        more = attention.earlier()
        if not more:
            break
        spent += len(more)

    assert spent == 10, "the ceiling is what one waking may spend, however many chunks it asks"
