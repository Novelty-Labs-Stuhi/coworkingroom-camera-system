"""Pixels say a passage happened and which way; the tracker says it was a person."""

from __future__ import annotations

import numpy as np

from stuhi_vision.attention import Attention
from stuhi_vision.domain import Box, Direction, Frame, TrackedPerson
from stuhi_vision.occlusion import Coverage, Occlusion
from stuhi_vision.passage import PassageMonitor
from stuhi_vision.threshold import ThresholdConfig

WIDTH, HEIGHT = 640, 480


class ScriptedEpisodes:
    """Coverage on demand, so the composition can be tested without synthesising pixels."""

    def __init__(self, episodes: dict[int, Coverage]) -> None:
        self._episodes = episodes
        self._frame = -1

    def __call__(self) -> Coverage | None:
        self._frame += 1
        return self._episodes.get(self._frame)


def _coverage(lag: float, frames: int = 4, slices: int = 4) -> Coverage:
    return Coverage(
        frames=frames, peak=0.6, slices=slices, lag=lag, order=(0.0, 1.0, 2.0, 3.0)
    )


def _frame(timestamp: float = 1.0) -> Frame:
    return Frame(timestamp=timestamp, image=np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))


def _person(track_id: int = 1, left: float = 0.0, height: float = 0.8) -> TrackedPerson:
    return TrackedPerson(
        track_id=track_id,
        box=Box(left * WIDTH, (1 - height) * HEIGHT, (left + 0.2) * WIDTH, HEIGHT),
    )


def _monitor(episodes: dict[int, Coverage], watched: list | None = None) -> PassageMonitor:
    return PassageMonitor(
        ThresholdConfig(
            zone=(0.0, 0.0, 0.12, 1.0),
            edge="left",
            passing_means=Direction.IN,
            min_height=0.35,
        ),
        ScriptedEpisodes(episodes),
        watcher=None if watched is None else watched.append,
    )


def test_an_order_towards_the_doorway_edge_with_a_person_is_counted() -> None:
    # lag < 0: the slices nearest the edge lit first, so the covering went that way.
    monitor = _monitor({1: _coverage(lag=-0.8)})

    monitor.update([_person()], _frame())
    crossings = monitor.update([_person()], _frame(timestamp=2.0))

    assert [crossing.direction for crossing in crossings] == [Direction.IN]
    assert crossings[0].track_id == 1


def test_the_other_order_is_the_other_direction() -> None:
    monitor = _monitor({1: _coverage(lag=+0.8)})

    monitor.update([_person()], _frame())
    crossings = monitor.update([_person()], _frame(timestamp=2.0))

    assert [crossing.direction for crossing in crossings] == [Direction.OUT]


def test_covered_with_no_order_is_somebody_on_the_doorframe() -> None:
    watched: list = []
    monitor = _monitor({1: _coverage(lag=0.0)}, watched)

    monitor.update([_person()], _frame())
    crossings = monitor.update([_person()], _frame(timestamp=2.0))

    assert crossings == []
    # Reported rather than dropped: standing at the door is a real thing that happened.
    assert len(watched) == 1
    assert watched[0].direction is None
    assert watched[0].person == 1


def test_covered_with_nobody_detected_is_never_counted() -> None:
    watched: list = []
    monitor = _monitor({1: _coverage(lag=-0.8)}, watched)

    monitor.update([], _frame())
    crossings = monitor.update([], _frame(timestamp=2.0))

    # The door swinging looks exactly like this, so it cannot be counted -- but it is said out
    # loud, because repeated it means the detector is failing rather than the door moving.
    assert crossings == []
    assert watched[0].person is None
    assert "nobody detected" in watched[0].readable


def test_somebody_across_the_room_does_not_claim_the_passage() -> None:
    """A person nowhere near the box, and too small, is not who covered it."""
    watched: list = []
    monitor = _monitor({1: _coverage(lag=-0.8)}, watched)

    far = _person(track_id=7, left=0.7, height=0.2)
    monitor.update([far], _frame())
    crossings = monitor.update([far], _frame(timestamp=2.0))

    assert crossings == []
    assert watched[0].person is None


def test_the_person_longest_on_the_box_is_the_one_credited() -> None:
    monitor = _monitor({3: _coverage(lag=-0.8)})

    monitor.update([_person(track_id=1), _person(track_id=2)], _frame())
    monitor.update([_person(track_id=2)], _frame(timestamp=2.0))
    monitor.update([_person(track_id=2)], _frame(timestamp=3.0))
    crossings = monitor.update([_person(track_id=2)], _frame(timestamp=4.0))

    assert crossings[0].track_id == 2


def test_the_pixels_and_the_tracker_together_over_synthetic_frames() -> None:
    """End to end with the real Occlusion: a dark shape sweeping across a lit doorframe."""
    # The real thing: Attention runs the pixels once a frame and hands the episode over.
    doorframe = Occlusion(zone=(0.0, 0.0, 0.5, 1.0), config=_settings())
    attention = Attention(doorframe, pre_roll_seconds=2.0)
    monitor = PassageMonitor(
        ThresholdConfig(
            zone=(0.0, 0.0, 0.5, 1.0), edge="left", passing_means=Direction.IN, min_height=0.35
        ),
        attention.episode,
    )

    rng = np.random.default_rng(11)
    empty = rng.integers(90, 110, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    empty[:, 100:140] = 210          # the doorframe

    crossings: list = []
    for _ in range(8):               # learn the empty doorway
        quiet = Frame(timestamp=0.0, image=empty)
        attention.examine(quiet)
        crossings += monitor.update([], quiet)

    # A body crossing right to left, tracked at the same time.
    for step, left in enumerate([0.42, 0.30, 0.18, 0.06]):
        covered = empty.copy()
        covered[:, int(left * WIDTH) : int((left + 0.14) * WIDTH)] = 15
        moment = Frame(timestamp=step + 1.0, image=covered)
        attention.examine(moment)
        crossings += monitor.update([_person(left=left)], moment)

    clear = Frame(timestamp=9.0, image=empty)
    attention.examine(clear)
    crossings += monitor.update([], clear)

    assert [crossing.direction for crossing in crossings] == [Direction.IN]


def _settings():
    from stuhi_vision.occlusion import CoverageConfig

    return CoverageConfig(slices=4, covered=0.2, min_frames=2)


def test_a_person_at_the_door_counts_even_if_the_overlap_frame_was_missed() -> None:
    """A twelfth-of-a-frame box is overlapped for a frame or two; detection may miss those.

    From the live log: the doorframe was plainly covered -- five slices, peak 0.78, a clear
    order -- and the passage was thrown away as "nobody detected", because no detected box
    happened to overlap the strip on the frames the detector managed.
    """
    watched: list = []
    monitor = _monitor({2: _coverage(lag=-0.8)}, watched)

    # Tall enough to be at this door, but never over the narrow box itself.
    beside = _person(track_id=9, left=0.30, height=0.8)
    monitor.update([beside], _frame())
    monitor.update([beside], _frame(timestamp=2.0))
    crossings = monitor.update([beside], _frame(timestamp=3.0))

    assert [crossing.direction for crossing in crossings] == [Direction.IN]
    assert crossings[0].track_id == 9


def test_over_the_box_still_wins_over_merely_being_near_it() -> None:
    monitor = _monitor({2: _coverage(lag=-0.8)})

    over = _person(track_id=1, left=0.0, height=0.8)
    beside = _person(track_id=2, left=0.30, height=0.8)
    monitor.update([over, beside], _frame())
    monitor.update([beside], _frame(timestamp=2.0))
    crossings = monitor.update([beside], _frame(timestamp=3.0))

    assert crossings[0].track_id == 1


def test_a_redrawn_box_is_judged_against_immediately() -> None:
    """Redrawing must not wait for a restart: it is done because the count is wrong now."""
    monitor = _monitor({1: _coverage(lag=-0.8)})
    monitor.use_zone((0.40, 0.0, 0.62, 1.0))

    # Over the new box, nowhere near the old one.
    over_new = _person(track_id=5, left=0.45, height=0.8)
    monitor.update([over_new], _frame())
    crossings = monitor.update([over_new], _frame(timestamp=2.0))

    assert crossings[0].track_id == 5


def test_the_coverage_monitor_accepts_the_covered_argument_and_ignores_it() -> None:
    """Both monitors must answer the same call, or the pipeline has to know which it holds.

    It got that wrong once: the replay passes each frame's own coverage, and the first replayed
    frame with somebody on it crashed the pipeline with an unexpected keyword argument. The
    supervisor then restarted it into the same crash.
    """
    monitor = _monitor({})

    # The keyword is accepted; this rule reads the doorframe itself and ignores the opinion.
    assert monitor.update([], _frame(0.0), covered=True) == []
    assert monitor.update([_person()], _frame(1.0), covered=False) == []
