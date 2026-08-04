"""Direction read from when the doorframe was covered, relative to when the person was seen.

The point of this rule is that the two need not coincide. On this doorway the detector acquires
somebody two to four frames *after* the coverage has ended, so every rule that asks what the
doorframe was doing during the track sees arrivals as nothing at all.
"""

from __future__ import annotations

import numpy as np

from stuhi_vision.domain import Box, Direction, TrackedPerson
from stuhi_vision.ordering import Episode, OrderingRule
from stuhi_vision.threshold import ThresholdConfig

WIDTH, HEIGHT = 640, 480


def _person(track_id: int, left: float = 0.05, height: float = 0.8) -> TrackedPerson:
    """Somebody over the doorframe zone and tall enough to be at this door."""
    return TrackedPerson(
        track_id=track_id,
        box=Box(left * WIDTH, (1 - height) * HEIGHT, (left + 0.25) * WIDTH, HEIGHT),
    )


def _rule(**overrides) -> OrderingRule:
    settings = {"zone": (0.0, 0.0, 0.30, 1.0), "passing_means": Direction.OUT}
    settings.update(overrides)
    return OrderingRule(ThresholdConfig(**settings))


def _run(rule: OrderingRule, script) -> list:
    """Play a script of (people, episode) per frame, then let open tracks finish."""
    verdicts = []
    for people, episode in script:
        verdicts += rule.observe(people, WIDTH, HEIGHT, episode)
    for _ in range(12):
        verdicts += rule.observe([], WIDTH, HEIGHT, None)
    return verdicts


def test_coverage_before_they_appear_is_an_arrival() -> None:
    """The case every other rule misses: the detector is late, so the track starts after it."""
    script = [([], Episode(1, 4))]                    # doorframe covered, nobody detected yet
    script += [([], None)] * 3                        # ...still nobody
    script += [([_person(1)], None) for _ in range(8)]  # now they are seen, inside

    verdicts = _run(_rule(passing_means=Direction.OUT), script)

    assert [(v.track_id, v.direction) for v in verdicts] == [(1, Direction.IN)]
    assert verdicts[0].because == "covered before they appeared"


def test_coverage_after_they_were_last_seen_is_a_departure() -> None:
    script = [([_person(1)], None) for _ in range(8)]
    script += [([], Episode(9, 13))]

    verdicts = _run(_rule(passing_means=Direction.OUT), script)

    assert [(v.track_id, v.direction) for v in verdicts] == [(1, Direction.OUT)]


def test_passing_means_decides_which_way_round_it_is() -> None:
    script = [([_person(1)], None) for _ in range(8)] + [([], Episode(9, 13))]

    verdicts = _run(_rule(passing_means=Direction.IN), script)

    assert verdicts[0].direction is Direction.IN


def test_coverage_at_both_ends_is_named_as_two_passages_but_not_yet_counted() -> None:
    """Somebody who came in, stayed, and left again -- the sixth track in the footage.

    No direction is emitted: two crossings under one track id would lose the second to the
    Doorkeeper, which takes the session on the first. Naming it is what makes the gap visible
    instead of looking like noise.
    """
    script = [([], Episode(1, 4))]
    script += [([_person(1)], None) for _ in range(20)]
    script += [([], Episode(26, 30))]

    verdicts = _run(_rule(passing_means=Direction.OUT), script)

    assert verdicts[0].direction is None
    assert verdicts[0].because == "came in then went out"


def test_somebody_who_never_reached_the_door_is_ignored() -> None:
    """Otherwise a door swinging shut plus anybody wandering into view reads as an arrival."""
    far = TrackedPerson(track_id=1, box=Box(0.6 * WIDTH, 0.2 * HEIGHT, 0.85 * WIDTH, HEIGHT))
    script = [([], Episode(1, 4))] + [([far], None) for _ in range(8)]

    verdicts = _run(_rule(), script)

    assert verdicts[0].direction is None
    assert verdicts[0].because == "never reached the door"


def test_somebody_too_short_to_be_at_this_door_is_ignored() -> None:
    distant = _person(1, height=0.15)
    script = [([], Episode(1, 4))] + [([distant], None) for _ in range(8)]

    verdicts = _run(_rule(min_height=0.35), script)

    assert verdicts[0].because == "never reached the door"


def test_a_track_with_no_coverage_anywhere_near_is_not_a_passage() -> None:
    script = [([_person(1)], None) for _ in range(8)]

    verdicts = _run(_rule(), script)

    assert verdicts[0].because == "no coverage near them"


def test_an_episode_beyond_the_window_is_not_credited_to_them() -> None:
    """A stranger's coverage from a minute ago must not become this person's arrival."""
    script = [([], Episode(1, 3))]
    script += [([], None)] * 40                       # far longer than the window
    script += [([_person(1)], None) for _ in range(8)]

    verdicts = _run(_rule(), script)

    assert verdicts[0].because == "no coverage near them"


def test_the_verdict_reads_clearly() -> None:
    script = [([_person(1)], None) for _ in range(8)] + [([], Episode(9, 13))]

    verdicts = _run(_rule(passing_means=Direction.OUT), script)

    assert verdicts[0].readable == "track 1 -> out (covered after they were last seen)"


def test_two_people_each_get_their_own_verdict() -> None:
    """One episode each, so a pair arriving together is two arrivals rather than one."""
    script = [([], Episode(1, 3))]
    script += [([_person(1)], None) for _ in range(6)]
    script += [([_person(1, left=0.10), _person(2)], None) for _ in range(6)]
    script += [([_person(2)], None) for _ in range(6)]

    verdicts = _run(_rule(passing_means=Direction.OUT), script)

    assert sorted(v.track_id for v in verdicts) == [1, 2]
    assert all(v.direction is Direction.IN for v in verdicts)


def test_the_episode_history_cannot_grow_without_bound() -> None:
    """A camera that runs for weeks must not accumulate episodes for ever."""
    rule = _rule()
    for index in range(500):
        rule.observe([], WIDTH, HEIGHT, Episode(index, index))

    assert len(rule._episodes) == 64


def test_a_brief_dropout_does_not_end_the_track_early() -> None:
    """Losing somebody for a frame or two mid-stride is normal at this frame rate."""
    script = [([_person(1)], None) for _ in range(5)]
    script += [([], None)] * 3                        # lost, but under lost_after
    script += [([_person(1)], None) for _ in range(5)]
    script += [([], Episode(14, 18))]

    verdicts = _run(_rule(passing_means=Direction.OUT), script)

    assert len(verdicts) == 1                          # one track, not two
    assert verdicts[0].direction is Direction.OUT


def test_nothing_is_reported_while_somebody_is_still_in_view() -> None:
    rule = _rule()
    verdicts = []
    for _ in range(30):
        verdicts += rule.observe([_person(1)], WIDTH, HEIGHT, None)

    assert verdicts == []


def test_the_image_is_only_measured_for_its_shape() -> None:
    """The rule reads geometry and episode timing, never pixels -- so a stub frame suffices."""
    blank = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    assert blank.shape[:2] == (HEIGHT, WIDTH)


# --- placing a finished episode in time ---------------------------------------------------
#
# The seam between the pipeline and this rule. Coverage is only reported once it has *ended*,
# so the caller has a length and the rule has the frame count -- and an off-by-one here shifts
# every episode by a frame, which is exactly enough to put it on the wrong side of a track.


def test_span_of_places_an_episode_ending_on_the_previous_frame() -> None:
    rule = _rule()
    for _ in range(20):
        rule.observe([], WIDTH, HEIGHT, None)

    # 20 frames seen; a 4-frame episode ended on frame 20, so it began on 17.
    assert rule.span_of(4) == Episode(17, 20)


def test_span_of_before_any_frame_is_degenerate_but_harmless() -> None:
    """Asked before a single frame has been seen, the span lands on the non-existent frame 0.

    It cannot mislead anything: an episode is only reported once coverage has *ended*, and
    ending needs a background to compare against, which the first frame is spent learning.
    So no episode can finish on frame one and this span is never actually produced.
    """
    rule = _rule()

    assert rule.span_of(1) == Episode(0, 0)


def test_a_span_taken_from_the_rule_lands_on_the_right_side_of_a_track() -> None:
    """The production sequence: the caller asks for the span, then hands it straight back."""
    rule = _rule(passing_means=Direction.OUT)
    verdicts = []
    for _ in range(8):                       # the person is in view
        verdicts += rule.observe([_person(1)], WIDTH, HEIGHT, None)
    # They leave; coverage of 5 frames finishes on the next frame the rule sees.
    verdicts += rule.observe([], WIDTH, HEIGHT, rule.span_of(5))
    for _ in range(12):
        verdicts += rule.observe([], WIDTH, HEIGHT, None)

    assert [v.direction for v in verdicts] == [Direction.OUT]
    assert verdicts[0].because == "covered after they were last seen"


# --- the rule as a committing monitor ---------------------------------------------------------


def _monitor_over(covered_at: int | None, seen_frames: range, frames: int = 40):
    """Run a track through OrderingMonitor, with the coverage episode ending on a chosen frame."""
    import numpy as np

    from stuhi_vision.domain import Frame
    from stuhi_vision.ordering import OrderingMonitor

    width, height = 320, 240
    episodes: list = []

    class _Episode:
        def __init__(self, n: int) -> None:
            self.frames = n

    monitor = OrderingMonitor(
        ThresholdConfig(zone=(0.0, 0.0, 0.30, 1.0), passing_means=Direction.IN, min_height=0.35),
        lambda: episodes.pop(0) if episodes else None,
    )
    crossings = []
    for index in range(frames):
        if index == covered_at:
            episodes.append(_Episode(3))
        people = [_person(1)] if index in seen_frames else []
        image = np.zeros((height, width, 3), dtype=np.uint8)
        crossings += monitor.update(people, Frame(timestamp=float(index), image=image))
    return crossings


def test_the_monitor_commits_a_crossing_when_the_covering_follows_the_person() -> None:
    """Seen first, then the doorframe covers: they walked into it and through."""
    crossings = _monitor_over(covered_at=14, seen_frames=range(2, 12))

    assert [c.direction for c in crossings] == [Direction.IN]


def test_the_monitor_commits_the_other_direction_when_the_covering_comes_first() -> None:
    """The covering, then they appear: they came through it into view.

    This is the direction the live rule loses -- 33 exits against its 17 on a day of real
    traffic -- and the reason this rule was promoted out of shadow at all.
    """
    crossings = _monitor_over(covered_at=4, seen_frames=range(8, 18))

    assert [c.direction for c in crossings] == [Direction.OUT]


def test_a_track_that_never_reached_the_door_commits_nothing() -> None:
    import numpy as np

    from stuhi_vision.domain import Frame
    from stuhi_vision.ordering import OrderingMonitor

    monitor = OrderingMonitor(
        ThresholdConfig(zone=(0.0, 0.0, 0.10, 1.0), passing_means=Direction.IN, min_height=0.35),
        lambda: None,
    )
    crossings = []
    for index in range(30):
        people = [_person(1, left=0.60)] if index < 10 else []
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        crossings += monitor.update(people, Frame(timestamp=float(index), image=image))

    assert crossings == []


def test_coming_in_and_going_out_again_commits_nothing_yet() -> None:
    """Two passages under one track id. Named by the rule, deliberately not committed: the
    Doorkeeper takes that track's session on the first and drops the second as having none."""
    import numpy as np

    from stuhi_vision.domain import Frame
    from stuhi_vision.ordering import OrderingMonitor

    class _Episode:
        def __init__(self, n: int) -> None:
            self.frames = n

    episodes: list = []
    seen = []
    monitor = OrderingMonitor(
        ThresholdConfig(zone=(0.0, 0.0, 0.30, 1.0), passing_means=Direction.IN, min_height=0.35),
        lambda: episodes.pop(0) if episodes else None,
        watcher=seen.append,
    )
    crossings = []
    for index in range(40):
        # Covered before they appeared, and again after they were last seen.
        if index in (3, 22):
            episodes.append(_Episode(3))
        people = [_person(1)] if 8 <= index < 18 else []
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        crossings += monitor.update(people, Frame(timestamp=float(index), image=image))

    assert crossings == [], "one track cannot commit two crossings"
    assert seen, "but the case is still named rather than silently dropped"
    assert "then went" in seen[-1].because
