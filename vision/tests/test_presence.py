"""Time in the room, derived from the entries and exits already recorded."""

from __future__ import annotations

from datetime import datetime, timedelta

from stuhi_vision.presence import (
    Passage,
    day_starting,
    leaderboard,
    office_day,
    readable,
    streaks,
    visits_from,
    window_bounds,
)


def _at(day: int, hour: int, minute: int = 0) -> float:
    return datetime(2026, 7, day, hour, minute).timestamp()


def _in(name: str, day: int, hour: int, minute: int = 0) -> Passage:
    return Passage(name=name, at=_at(day, hour, minute), direction="in")


def _out(name: str, day: int, hour: int, minute: int = 0) -> Passage:
    return Passage(name=name, at=_at(day, hour, minute), direction="out")


def _each_day(name: str, days) -> list[Passage]:
    """One short visit on each of those days, for streak tests."""
    return [
        passage
        for day in days
        for passage in (_in(name, day, 9), _out(name, day, 10))
    ]


def test_an_entry_and_the_following_exit_are_one_visit() -> None:
    visits = visits_from([_in("ilari", 20, 9), _out("ilari", 20, 17)])

    assert len(visits) == 1
    assert visits[0].seconds(until=_at(20, 23)) == 8 * 3600


def test_somebody_still_inside_counts_up_to_now() -> None:
    visits = visits_from([_in("art", 20, 9)])

    assert visits[0].left is None
    assert visits[0].seconds(until=_at(20, 12)) == 3 * 3600


def test_an_exit_with_no_entry_contributes_nothing() -> None:
    """There is no moment to measure from, and choosing one invents hours."""
    assert visits_from([_out("nobody", 20, 17)]) == []


def test_a_missed_exit_does_not_let_one_visit_swallow_the_next() -> None:
    """Two entries in a row: they were plainly not in the room twice at once."""
    visits = visits_from(
        [_in("ilari", 20, 9), _in("ilari", 20, 14), _out("ilari", 20, 15)]
    )

    assert [(v.entered, v.left) for v in visits] == [
        (_at(20, 9), _at(20, 14)),
        (_at(20, 14), _at(20, 15)),
    ]


def test_the_office_day_starts_at_half_past_four() -> None:
    """Somebody in the room at one in the morning is finishing the previous day."""
    assert office_day(_at(21, 1)) == datetime(2026, 7, 20).date()
    assert office_day(_at(21, 5)) == datetime(2026, 7, 21).date()


def test_a_visit_past_midnight_is_split_between_the_days_it_covers() -> None:
    """Giving it to one day would make the board depend on when people arrive."""
    visits = visits_from([_in("ilari", 20, 22), _out("ilari", 21, 2)])
    monday = day_starting(datetime(2026, 7, 20).date())

    held = visits[0].overlap(monday, monday + 24 * 3600, until=_at(21, 12))
    assert held == 4 * 3600   # 22:00 to 02:00, all within the office day that began at 04:30


def test_the_leaderboard_lists_only_people_who_appeared() -> None:
    visits = visits_from(
        [
            _in("ilari", 20, 9), _out("ilari", 20, 17),
            _in("art", 20, 13), _out("art", 20, 14),
            _in("away", 10, 9), _out("away", 10, 10),
        ]
    )
    start, end = day_starting(datetime(2026, 7, 20).date()), _at(21, 4)

    board = leaderboard(visits, start, end, now=_at(21, 4))

    assert [(s.name, s.seconds / 3600) for s in board] == [("ilari", 8.0), ("art", 1.0)]


def test_somebody_still_inside_is_marked_as_such() -> None:
    visits = visits_from([_in("art", 20, 9)])
    start, end = day_starting(datetime(2026, 7, 20).date()), _at(20, 12)

    board = leaderboard(visits, start, end, now=_at(20, 12))

    assert board[0].still_inside is True


def test_a_window_covers_whole_office_days() -> None:
    """A rolling window would let the same visit drift in and out by the hour."""
    now = _at(21, 15)
    start, end = window_bounds("week", offset=0, now=now)

    assert start == day_starting(datetime(2026, 7, 15).date())
    assert end == now


def test_going_back_a_window_lands_on_the_previous_one() -> None:
    now = _at(21, 15)
    this_week, _ = window_bounds("week", offset=0, now=now)
    last_week_start, last_week_end = window_bounds("week", offset=1, now=now)

    assert last_week_end <= this_week + 24 * 3600
    assert last_week_start == this_week - 7 * 24 * 3600


def test_a_streak_is_consecutive_office_days_ending_today() -> None:
    visits = visits_from(_each_day("ilari", (18, 19, 20)))

    board = streaks(visits, now=_at(20, 12))

    assert [(s.name, s.seconds) for s in board] == [("ilari", 3.0)]


def test_a_gap_ends_the_streak() -> None:
    visits = visits_from(_each_day("art", (16, 19, 20)))

    assert streaks(visits, now=_at(20, 12))[0].seconds == 2.0


def test_a_streak_is_not_reported_broken_before_anybody_arrives() -> None:
    """At nine in the morning nobody is in yet; yesterday's streak still stands."""
    visits = visits_from(_each_day("art", (19, 20)))

    assert streaks(visits, now=_at(21, 6))[0].seconds == 2.0


def test_durations_read_the_way_somebody_would_say_them() -> None:
    assert readable(3 * 3600 + 12 * 60) == "3 h 12 m"
    assert readable(48 * 60) == "48 m"
    assert readable(40) == "40 s"


def test_a_year_window_is_a_year_of_office_days() -> None:
    now = _at(21, 15)
    start, _ = window_bounds("year", offset=0, now=now)

    assert start == day_starting((datetime(2026, 7, 21) - timedelta(days=364)).date())


def test_unclosed_entries_are_who_the_record_thinks_is_inside() -> None:
    visits = visits_from(
        [_in("ilari", 20, 9), _out("ilari", 20, 12), _in("art", 20, 10), _in("miko", 20, 11)]
    )

    from stuhi_vision.presence import could_have_left, unclosed

    assert sorted(v.name for v in unclosed(visits)) == ["art", "miko"]
    assert could_have_left(visits, "art") is True
    # An exit cannot belong to somebody the record does not have inside: naming one that way
    # means an earlier entry carries the wrong name, and that mistake is in the unclosed list.
    assert could_have_left(visits, "ilari") is False
