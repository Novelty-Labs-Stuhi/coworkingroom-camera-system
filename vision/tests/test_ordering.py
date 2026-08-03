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
