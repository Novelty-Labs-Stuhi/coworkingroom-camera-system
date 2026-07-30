"""Passages recognised from where a track begins and ends, not from a line crossing."""

from __future__ import annotations

import numpy as np

from stuhi_vision.domain import Box, Direction, Frame, TrackedPerson
from stuhi_vision.threshold import ThresholdConfig, ThresholdMonitor

WIDTH, HEIGHT = 640, 480


def _frame(timestamp: float = 0.0) -> Frame:
    return Frame(timestamp=timestamp, image=np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))


def _person(track_id: int, left: float, width_fraction: float = 0.25, height: float = 0.8):
    """A person as fractions of the frame, so the tests read like the rule does."""
    return TrackedPerson(
        track_id=track_id,
        box=Box(
            left * WIDTH,
            (1 - height) * HEIGHT,
            (left + width_fraction) * WIDTH,
            HEIGHT,
        ),
    )


def _monitor(**overrides) -> ThresholdMonitor:
    return ThresholdMonitor(ThresholdConfig(**overrides))


def _run(monitor: ThresholdMonitor, positions, track_id: int = 1, height: float = 0.8):
    """Walk a track across the given left-edge positions, then let it be lost."""
    crossings = []
    for index, left in enumerate(positions):
        crossings += monitor.update([_person(track_id, left, height=height)], _frame(index))
    for index in range(20):  # nobody in frame: the track finishes
        crossings += monitor.update([], _frame(100 + index))
    return crossings


def test_walking_through_and_out_of_the_left_edge_is_a_passage() -> None:
    # Enters mid-frame over the doorframe zone, moves left, disappears past the edge.
    crossings = _run(_monitor(), [0.55, 0.40, 0.25, 0.10, 0.02])

    assert [c.direction for c in crossings] == [Direction.IN]


def test_arriving_from_the_edge_is_the_opposite_direction() -> None:
    crossings = _run(_monitor(), [0.02, 0.12, 0.30, 0.50, 0.60])

    assert [c.direction for c in crossings] == [Direction.OUT]


def test_background_traffic_never_touching_the_doorframe_is_ignored() -> None:
    # Beyond the opening on the right, crossing laterally: never overlaps the zone.
    crossings = _run(_monitor(zone=(0.0, 0.0, 0.25, 1.0)), [0.60, 0.65, 0.70, 0.75, 0.80])

    assert crossings == []


def test_someone_far_away_is_ignored_even_over_the_zone() -> None:
    # Lines up with the doorframe in x, but is too small to be at the door.
    crossings = _run(_monitor(min_height=0.35), [0.20, 0.15, 0.05], height=0.15)

    assert crossings == []


def test_loitering_at_the_edge_without_passing_is_not_counted() -> None:
    # Starts and ends at the edge: stepped in and back out. Guessing would add noise.
    crossings = _run(_monitor(), [0.02, 0.10, 0.03])

    assert crossings == []


def test_a_track_still_present_produces_nothing_yet() -> None:
    # The decision needs the track to be *finished*; whether someone passed through is only
    # knowable once they stop being visible.
    monitor = _monitor()
    crossings = []
    for index, left in enumerate([0.55, 0.40, 0.20, 0.05]):
        crossings += monitor.update([_person(1, left)], _frame(index))

    assert crossings == []


def test_a_brief_gap_does_not_end_a_track_early() -> None:
    # At a few frames per second a person is easily missed for a frame mid-stride.
    monitor = _monitor(lost_after=6)
    crossings = []
    for index, left in enumerate([0.55, 0.40]):
        crossings += monitor.update([_person(1, left)], _frame(index))
    for index in range(3):  # missed frames
        crossings += monitor.update([], _frame(10 + index))
    assert crossings == []

    crossings += monitor.update([_person(1, 0.05)], _frame(20))
    for index in range(20):
        crossings += monitor.update([], _frame(30 + index))

    assert [c.direction for c in crossings] == [Direction.IN]


def test_two_people_are_judged_independently() -> None:
    # Track ids keep them apart: one walks in, the other is background traffic.
    monitor = _monitor()
    crossings = []
    walker = [0.55, 0.40, 0.20, 0.04]
    loiterer = [0.70, 0.72, 0.74, 0.76]
    for index in range(4):
        crossings += monitor.update(
            [_person(1, walker[index]), _person(2, loiterer[index])], _frame(index)
        )
    for index in range(20):
        crossings += monitor.update([], _frame(50 + index))

    assert [(c.track_id, c.direction) for c in crossings] == [(1, Direction.IN)]


def test_the_bottom_edge_works_for_a_room_facing_camera() -> None:
    # People approach the lens and leave downwards; the rule is the same shape.
    monitor = ThresholdMonitor(
        ThresholdConfig(
            zone=(0.0, 0.0, 1.0, 1.0),
            edge="bottom",
            passing_means=Direction.OUT,
            min_height=0.3,
        )
    )
    crossings = []
    for index, bottom in enumerate([0.5, 0.7, 0.9, 1.0]):
        person = TrackedPerson(
            track_id=1, box=Box(200, (bottom - 0.4) * HEIGHT, 400, bottom * HEIGHT)
        )
        crossings += monitor.update([person], _frame(index))
    for index in range(20):
        crossings += monitor.update([], _frame(50 + index))

    assert [c.direction for c in crossings] == [Direction.OUT]


def _approaching(track_id: int, heights, width_fraction: float = 0.3):
    """A person whose apparent size changes while staying against the bottom edge.

    Which is what the room-facing camera sees: measured tracks began *and* ended touching
    the bottom (0.97-1.00 of the frame), so only the size change distinguishes them.
    """
    for height in heights:
        yield TrackedPerson(
            track_id=track_id,
            box=Box(
                0.4 * WIDTH,
                (1 - height) * HEIGHT,
                (0.4 + width_fraction) * WIDTH,
                HEIGHT,  # always flush against the bottom edge
            ),
        )


def _approach_monitor(**overrides) -> ThresholdMonitor:
    return ThresholdMonitor(
        ThresholdConfig(
            zone=(0.0, 0.0, 1.0, 1.0),
            discriminator="approach",
            passing_means=Direction.OUT,
            min_height=0.45,
            growth_margin=0.12,
            **overrides,
        )
    )


def _finish(monitor: ThresholdMonitor, people_frames) -> list:
    crossings = []
    for index, person in enumerate(people_frames):
        crossings += monitor.update([person], _frame(index))
    for index in range(20):
        crossings += monitor.update([], _frame(200 + index))
    return crossings


def test_growing_towards_the_lens_is_a_passage_out() -> None:
    # Measured: heights 0.64 -> 0.82 as someone walked at the camera on their way out.
    crossings = _finish(_approach_monitor(), _approaching(1, [0.64, 0.70, 0.76, 0.82]))

    assert [c.direction for c in crossings] == [Direction.OUT]


def test_shrinking_away_from_the_lens_is_the_opposite() -> None:
    # Measured: 0.69 -> 0.41 as someone who had come in walked away into the room.
    crossings = _finish(_approach_monitor(), _approaching(1, [0.69, 0.60, 0.50, 0.41]))

    assert [c.direction for c in crossings] == [Direction.IN]


def test_barely_changing_size_is_not_a_passage() -> None:
    # Measured at ~0.09: somebody shifting about near the door rather than going through.
    crossings = _finish(_approach_monitor(), _approaching(1, [0.70, 0.66, 0.63, 0.61]))

    assert crossings == []


def test_a_seated_person_across_the_room_is_ignored() -> None:
    # That view has somebody visible in every frame; only size keeps them out of the count.
    crossings = _finish(_approach_monitor(), _approaching(1, [0.29, 0.28, 0.29, 0.28]))

    assert crossings == []


def test_the_edge_test_would_have_discarded_these_passages() -> None:
    # Why "approach" exists: flush against the bottom throughout, so the edge test sees
    # "arrived at the edge AND left at the edge" and calls it no passage at all.
    edge_based = ThresholdMonitor(
        ThresholdConfig(
            zone=(0.0, 0.0, 1.0, 1.0),
            edge="bottom",
            passing_means=Direction.OUT,
            min_height=0.45,
        )
    )
    assert _finish(edge_based, _approaching(1, [0.64, 0.70, 0.76, 0.82])) == []


# --- the travel rule: crossing the box is the passage, its direction is the direction -------

def _travelling(**overrides) -> ThresholdMonitor:
    """The live doorway camera: a narrow drawn box at the left, judged by direction."""
    settings = {
        "zone": (0.0, 0.0, 0.12, 1.0),
        "edge": "left",
        "discriminator": "travel",
        "passing_means": Direction.IN,
        "min_height": 0.35,
    }
    settings.update(overrides)
    return ThresholdMonitor(ThresholdConfig(**settings))


def test_moving_across_the_box_towards_the_edge_counts() -> None:
    crossings = _run(_travelling(), [0.40, 0.28, 0.16, 0.06, 0.00])

    assert [crossing.direction for crossing in crossings] == [Direction.IN]


def test_moving_across_the_box_the_other_way_counts_the_other_way() -> None:
    crossings = _run(_travelling(), [0.00, 0.06, 0.16, 0.28, 0.40])

    assert [crossing.direction for crossing in crossings] == [Direction.OUT]


def test_a_doorframe_away_from_the_picture_edge_still_works() -> None:
    """The edge rule needed the track to end at the edge of the frame; this one does not.

    A camera can be mounted so the doorframe sits in the middle of the view, and a track is
    often lost before reaching any edge anyway -- the light is poor and the frame rate low.
    Under the old rule both were passages silently dropped: neither the first nor the last
    sighting is at an edge, so it refused to judge.
    """
    walk = [0.75, 0.62, 0.50, 0.40]   # crosses a mid-frame doorframe, stops well short of it
    middle = {"zone": (0.40, 0.0, 0.62, 1.0)}

    assert _run(_travelling(**middle), walk)[0].direction is Direction.IN
    assert _run(_travelling(discriminator="edge", **middle), walk) == []


def test_standing_in_the_doorway_is_not_a_passage() -> None:
    crossings = _run(_travelling(), [0.10, 0.12, 0.09, 0.11, 0.10])

    assert crossings == []


def test_crossing_the_room_behind_the_door_is_still_ignored() -> None:
    # Never overlaps the box, however far it travels.
    crossings = _run(_travelling(), [0.80, 0.65, 0.50, 0.35, 0.20])

    assert crossings == []


def test_someone_too_small_to_be_at_the_door_is_ignored() -> None:
    crossings = _run(_travelling(), [0.40, 0.28, 0.16, 0.04], height=0.20)

    assert crossings == []


def test_the_direction_of_travel_can_be_read_the_other_way_round() -> None:
    """If the camera is mounted facing the other way, one word in the config flips it."""
    crossings = _run(_travelling(passing_means=Direction.OUT), [0.40, 0.28, 0.16, 0.06])

    assert [crossing.direction for crossing in crossings] == [Direction.OUT]


def test_a_person_passing_close_to_the_lens_is_not_read_as_standing_still() -> None:
    """From replayed footage: a real passage whose centre barely moves.

    Walking close past the camera swells the box in both directions at once. This track went
    from (0.30..0.42) to (0.01..0.68) -- a centre shift of 0.015, which reads as standing
    still, while the near side swept 0.29 across the picture.
    """
    monitor = _travelling()
    boxes = [(0.30, 0.42), (0.22, 0.50), (0.12, 0.60), (0.01, 0.68)]
    crossings = []
    for index, (left, right) in enumerate(boxes):
        person = TrackedPerson(
            track_id=1,
            box=Box(left * WIDTH, 0.10 * HEIGHT, right * WIDTH, HEIGHT),
        )
        crossings += monitor.update([person], _frame(index))
    for index in range(20):
        crossings += monitor.update([], _frame(100 + index))

    assert [crossing.direction for crossing in crossings] == [Direction.IN]
