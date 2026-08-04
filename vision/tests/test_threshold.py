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


def test_a_track_that_reaches_the_box_is_reported_either_way() -> None:
    """A refused passage and one the tracker never saw look identical without this.

    They need opposite fixes -- a threshold adjusted versus a camera or a light -- so the
    difference has to be visible rather than inferred.
    """
    touches = []
    monitor = ThresholdMonitor(
        ThresholdConfig(
            zone=(0.0, 0.0, 0.12, 1.0), edge="left", discriminator="travel", min_height=0.35
        ),
        report=touches.append,
    )

    _run(monitor, [0.40, 0.28, 0.16, 0.06])       # a passage
    _run(monitor, [0.10, 0.12, 0.09, 0.11], track_id=2)   # stood in the doorway
    _run(monitor, [0.80, 0.65, 0.50], track_id=3)         # never reached the box

    assert [touch.direction for touch in touches] == [Direction.IN, None]
    assert "travelled -0.34" in touches[0].readable


# --- the "covering" discriminator: direction from the doorframe's pixels ----------------
#
# The doorframe is covered only when a body is in front of it, so the coverage state at the
# two moments a track is bounded by says which way the person went. Nothing here measures a
# distance, so these tests hold a track still and vary only the coverage.


def _covering_monitor(coverage: list[bool], **overrides) -> ThresholdMonitor:
    """A monitor reading coverage from a list, one entry consumed per frame."""
    frames = iter(coverage)
    return ThresholdMonitor(
        ThresholdConfig(discriminator="covering", **overrides),
        covered=lambda: next(frames, False),
    )


def _run_covering(monitor: ThresholdMonitor, positions, track_id: int = 1):
    crossings = []
    for index, left in enumerate(positions):
        crossings += monitor.update([_person(track_id, left)], _frame(index))
    for index in range(20):
        crossings += monitor.update([], _frame(100 + index))
    return crossings


def test_vanishing_while_the_doorframe_is_covered_is_a_passage_through_it() -> None:
    # Clear when they appear, covered on the frame they were last seen: they left through
    # the doorway. The box barely moves -- only the coverage decides.
    crossings = _run_covering(
        _covering_monitor([False, False, True, True]), [0.20, 0.20, 0.20, 0.20]
    )

    assert [c.direction for c in crossings] == [Direction.IN]


def test_appearing_while_covered_then_clearing_is_the_opposite_direction() -> None:
    crossings = _run_covering(
        _covering_monitor([True, True, False, False]), [0.20, 0.20, 0.20, 0.20]
    )

    assert [c.direction for c in crossings] == [Direction.OUT]


def test_covered_at_both_ends_is_refused_rather_than_guessed() -> None:
    # To the doorway and back again: a passage in neither direction.
    crossings = _run_covering(
        _covering_monitor([True, False, False, True]), [0.20, 0.20, 0.20, 0.20]
    )

    assert crossings == []


def test_never_covering_the_doorframe_is_not_a_passage() -> None:
    crossings = _run_covering(
        _covering_monitor([False, False, False, False]), [0.20, 0.20, 0.20, 0.20]
    )

    assert crossings == []


def test_covering_needs_no_travel_at_all() -> None:
    """A stationary track still resolves -- which is the whole point of the rule.

    ``travel`` refuses this: the leading edge moves nothing, so it reads as standing in the
    doorway. The pixels say otherwise, and here they are what is asked.
    """
    still = [0.20, 0.20, 0.20, 0.20]
    covering = _run_covering(_covering_monitor([False, False, True, True]), still)
    travelling = _run(_monitor(discriminator="travel"), still)

    assert [c.direction for c in covering] == [Direction.IN]
    assert travelling == []


def test_coverage_is_credited_only_to_a_track_that_reached_the_doorframe() -> None:
    """Coverage is one fact about the doorway, so something must tie it to a person.

    Without the zone gate, somebody crossing the background while a door swings would be
    handed the doorway's coverage and counted.
    """
    crossings = _run_covering(
        _covering_monitor([False, False, True, True], zone=(0.0, 0.0, 0.25, 1.0)),
        [0.60, 0.65, 0.70, 0.75],
    )

    assert crossings == []


def test_without_a_coverage_source_the_rule_judges_nothing() -> None:
    """An unwired monitor must refuse, not invent a direction from a default."""
    monitor = ThresholdMonitor(ThresholdConfig(discriminator="covering"))

    assert _run_covering(monitor, [0.55, 0.40, 0.25, 0.10]) == []


# --- the "preceded" rule ---------------------------------------------------------------------
#
# Direction from one question: when the doorframe activated, had this person already been seen
# off it? Seen off it first means they walked in; the frame covering first means they came out.
# Nothing reads the order the frame's slices lit, which is what the coverage rule needs and
# cannot get when somebody close to the lens covers every slice within one frame.


class _Doorframe:
    """The coverage signal, driven by the test the way the pipeline drives it: a callable."""

    def __init__(self) -> None:
        self.covered = False

    def __call__(self) -> bool:
        return self.covered


def _preceded(passing_means: Direction = Direction.IN, report=None):
    door = _Doorframe()
    monitor = ThresholdMonitor(
        ThresholdConfig(
            discriminator="preceded",
            zone=(0.0, 0.0, 0.30, 1.0),
            passing_means=passing_means,
        ),
        report=report,
        covered=door,
    )
    return monitor, door


def _walk(monitor, door, steps, track_id: int = 1, height: float = 0.8):
    """Walk a track through (left, covered) steps, then let it be lost so it is judged."""
    crossings = []
    for index, (left, covered) in enumerate(steps):
        door.covered = covered
        crossings += monitor.update([_person(track_id, left, height=height)], _frame(index))
    door.covered = False
    for index in range(20):
        crossings += monitor.update([], _frame(100 + index))
    return crossings


def test_seen_off_the_doorframe_before_it_covered_is_walking_in() -> None:
    """Approaching is visible first, and the frame is covered on the way through."""
    monitor, door = _preceded()

    crossings = _walk(monitor, door, [(0.70, False), (0.50, False), (0.20, True), (0.05, True)])

    assert [c.direction for c in crossings] == [Direction.IN]


def test_the_doorframe_covering_first_is_coming_out() -> None:
    """Somebody leaving covers the frame on their way to being visible at all."""
    monitor, door = _preceded()

    crossings = _walk(monitor, door, [(0.05, True), (0.20, True), (0.50, False), (0.70, False)])

    assert [c.direction for c in crossings] == [Direction.OUT]


def test_a_doorframe_that_never_covered_is_not_a_passage() -> None:
    """Somebody crossing the room behind the door is not somebody going through it."""
    monitor, door = _preceded()

    assert _walk(monitor, door, [(0.70, False), (0.60, False), (0.50, False)]) == []


def test_never_being_seen_off_the_doorframe_is_refused() -> None:
    """Standing on the frame the whole time has no before and no after to read."""
    monitor, door = _preceded()

    assert _walk(monitor, door, [(0.05, True), (0.10, True), (0.05, True)]) == []


def test_covered_in_the_same_frame_the_person_appeared_is_coming_out() -> None:
    """"Covered at the moment they appeared" and "covered before" mean the same thing.

    At this frame rate somebody stepping out is detected and covers the frame within a single
    frame, so refusing the same-frame case would throw away the commonest exit there is. The
    frame was already going when they showed up, and that is what coming out looks like.
    """
    monitor, door = _preceded()

    crossings = _walk(monitor, door, [(0.70, True), (0.60, True)])

    assert [c.direction for c in crossings] == [Direction.OUT]


def test_passing_means_inverts_the_pair_for_the_other_camera() -> None:
    """The sign belongs to the camera, not to the rule."""
    monitor, door = _preceded(passing_means=Direction.OUT)

    crossings = _walk(monitor, door, [(0.70, False), (0.50, False), (0.20, True)])

    assert [c.direction for c in crossings] == [Direction.OUT]


def test_the_ordering_is_reported_so_a_wrong_verdict_can_be_read_back() -> None:
    seen = []
    monitor, door = _preceded(report=seen.append)

    _walk(monitor, door, [(0.70, False), (0.50, False), (0.20, True)])

    assert seen, "a track that reached the box must be reported whether or not it counted"
    touch = seen[-1]
    # Frames are numbered from 1: the index is advanced before the frame is read.
    assert touch.outside_at == 1
    assert touch.covered_at == 3
    assert "off the box at 1, covered at 3" in touch.order
    assert "seen off the box first" in touch.order


def test_the_slice_order_being_unreadable_does_not_stop_this_rule() -> None:
    """The failure this rule exists for: coverage arriving all at once, with no order in it.

    The coverage rule needs the slices to light in sequence and gets nothing when somebody close
    to the lens covers them within a single frame. This rule only needs the person to have been
    seen somewhere off the box first, which is true of the same walk.
    """
    monitor, door = _preceded()

    # One frame of approach, then fully covered instantly -- no gradual sweep at all.
    crossings = _walk(monitor, door, [(0.80, False), (0.02, True)])

    assert [c.direction for c in crossings] == [Direction.IN]


def test_the_pixels_decide_the_passage_not_a_box_overlapping_the_zone() -> None:
    """The case the doorframe exists to catch, and the one a box-overlap test throws away.

    Somebody going through is often half behind the door, or beside the frame, while the frame's
    pixels plainly change. Their detected box never overlaps the zone. Under a rule that asks
    "did the person's box reach the rectangle" that crossing is refused; under this one the
    covering is the event and the box is only asked when it happened.
    """
    monitor, door = _preceded()

    # The person stays well clear of the zone (which ends at 0.30) for their whole track, and
    # the doorframe covers anyway -- the door itself, or a shoulder outside the box.
    crossings = _walk(monitor, door, [(0.80, False), (0.70, False), (0.60, True), (0.55, True)])

    assert [c.direction for c in crossings] == [Direction.IN]


def test_pixels_that_never_change_are_not_a_passage_however_close_somebody_walks() -> None:
    """Background traffic: the room behind the door is not the door."""
    monitor, door = _preceded()

    # Walks right across the zone, but the doorframe's pixels never register a covering.
    assert _walk(monitor, door, [(0.80, False), (0.20, False), (0.02, False)]) == []
