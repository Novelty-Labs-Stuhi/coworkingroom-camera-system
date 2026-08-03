"""The activity chart's data: a box per day, and one day opened up."""

from __future__ import annotations

from datetime import date, timedelta

from stuhi_vision.activity import SHADES, calendar, held_on, on_day, spanning
from stuhi_vision.presence import Passage, Visit, day_starting, office_day


def _visit(day: date, from_hour: float, hours: float) -> Visit:
    entered = day_starting(day) + from_hour * 3600
    return Visit(name="ilari", entered=entered, left=entered + hours * 3600)


def _now(day: date) -> float:
    """Late enough in ``day`` that every visit in these tests has finished."""
    return day_starting(day) + 20 * 3600


def test_every_day_in_the_span_gets_a_box() -> None:
    """A missing box would shift every box after it into the wrong column."""
    first, last = date(2026, 6, 1), date(2026, 6, 30)

    days = calendar([], first, last, epoch=0.0, now=_now(last))

    assert len(days) == 30
    assert days[0].day == first and days[-1].day == last
    assert all(day.level == 0 for day in days)


def test_a_longer_day_is_darker_than_a_shorter_one() -> None:
    first, last = date(2026, 6, 1), date(2026, 6, 3)
    visits = [_visit(first, 9, 1), _visit(date(2026, 6, 2), 9, 8)]

    boxes = {day.day: day for day in calendar(visits, first, last, 0.0, _now(last))}

    assert boxes[date(2026, 6, 2)].level == SHADES     # the busiest day is the darkest
    assert 0 < boxes[first].level < SHADES
    assert boxes[date(2026, 6, 3)].level == 0          # and a day off is empty


def test_a_short_visit_still_gets_a_visible_box() -> None:
    """Being here briefly is the thing the chart is for, so it must not round away to nothing."""
    first, last = date(2026, 6, 1), date(2026, 6, 2)
    visits = [_visit(first, 9, 10 / 60), _visit(date(2026, 6, 2), 9, 10)]

    boxes = {day.day: day for day in calendar(visits, first, last, 0.0, _now(last))}

    assert boxes[first].level == 1
    assert boxes[first].seconds > 0


def test_the_shades_are_relative_to_that_person_alone() -> None:
    """Somebody who comes in for an hour a day has a readable chart, not a blank one."""
    first, last = date(2026, 6, 1), date(2026, 6, 2)
    brief = [_visit(first, 9, 0.25), _visit(date(2026, 6, 2), 9, 1)]

    boxes = {day.day: day for day in calendar(brief, first, last, 0.0, _now(last))}

    assert boxes[date(2026, 6, 2)].level == SHADES


def test_a_day_before_counting_began_is_marked_as_uncounted() -> None:
    """An empty box then means "nobody was looking", which is not "they were not here"."""
    first, last = date(2026, 6, 1), date(2026, 6, 4)
    epoch = day_starting(date(2026, 6, 3))

    days = {day.day: day for day in calendar([], first, last, epoch, _now(last))}

    assert days[date(2026, 6, 1)].counted is False
    assert days[date(2026, 6, 2)].counted is False
    assert days[date(2026, 6, 3)].counted is True
    assert days[date(2026, 6, 4)].counted is True


def test_a_visit_over_the_boundary_is_split_across_both_days() -> None:
    """A long evening reads as the two days it covered, not as one enormous one."""
    first = date(2026, 6, 1)
    last = first + timedelta(days=1)
    # In at 22:00, out at 06:00 -- which crosses the 04:30 boundary.
    over = _visit(first, 17.5, 8)

    days = {day.day: day for day in calendar([over], first, last, 0.0, _now(last + timedelta(1)))}

    assert days[first].seconds > 0
    assert days[last].seconds > 0
    assert round(days[first].seconds + days[last].seconds) == 8 * 3600


def test_a_day_counts_the_visits_that_touch_it() -> None:
    first = date(2026, 6, 1)
    visits = [_visit(first, 9, 1), _visit(first, 13, 2)]

    days = calendar(visits, first, first, 0.0, _now(first))

    assert days[0].visits == 2


def test_the_chart_starts_on_a_monday_so_the_rows_line_up() -> None:
    """Otherwise each row would hold a different weekday depending on when it was opened."""
    for day_offset in range(7):
        moment = day_starting(date(2026, 6, 1) + timedelta(days=day_offset)) + 12 * 3600
        first, last = spanning(moment, weeks=53)
        assert first.weekday() == 0, day_offset
        assert last == office_day(moment)
        assert (last - first).days + 1 >= 53 * 7 - 6


def test_one_day_holds_only_its_own_crossings() -> None:
    wanted = date(2026, 6, 2)
    passages = [
        Passage("ilari", day_starting(date(2026, 6, 1)) + 10 * 3600, "in"),
        Passage("ilari", day_starting(wanted) + 9 * 3600, "in"),
        Passage("ilari", day_starting(wanted) + 17 * 3600, "out"),
        Passage("ilari", day_starting(date(2026, 6, 3)) + 9 * 3600, "in"),
    ]

    found = on_day(passages, wanted)

    assert [passage.direction for passage in found] == ["in", "out"]
    assert found[0].at < found[1].at


def test_a_crossing_after_midnight_belongs_to_the_day_that_is_ending() -> None:
    """The day runs 04:30 to 04:30, so leaving at one in the morning closes the previous day."""
    working = date(2026, 6, 2)
    late = Passage("ilari", day_starting(working) + 20.5 * 3600, "out")   # 01:00 the next date

    assert on_day([late], working) == [late]
    assert on_day([late], working + timedelta(days=1)) == []


def test_how_long_somebody_was_here_on_one_day() -> None:
    day = date(2026, 6, 1)

    held = held_on([_visit(day, 9, 3)], day, _now(day))

    assert round(held) == 3 * 3600
