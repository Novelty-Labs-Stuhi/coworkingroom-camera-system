"""How much somebody was here on each day, and what happened on one of them.

Two views of the crossings already recorded, for the two halves of the activity chart: a year
of days to see a habit in, and one day opened up to see the entries and exits it is made of.

Derived rather than stored, for the same reason the leaderboard is (see :mod:`.presence`): a
name corrected today changes what last Tuesday's box should be, and a stored count would keep
the old answer until somebody remembered to rebuild it.

Two things about the boxes are worth stating, because both are choices rather than arithmetic:

* **A day runs 04:30 to 04:30**, as everywhere else here. Somebody who left at one in the
  morning gets one dark box for a long day, not two pale ones for two half days.
* **The shades are relative to that person's own busiest day** in the chart, which is what
  makes the pattern readable for somebody who comes in for an hour and for somebody who lives
  here. It also means a shade cannot be compared between two people's charts, only within one.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

from .presence import Passage, Visit, day_starting, office_day

# How many shades sit above "nothing at all". Four, as on the chart this copies -- enough to
# read a pattern from, few enough that each step is visibly different from the last.
SHADES = 4


@dataclass(frozen=True, slots=True)
class Day:
    """One box on the chart."""

    day: date
    seconds: float
    visits: int
    # Whether the record covers this day at all. An empty box before counting began means
    # "nobody was looking", which is not the same as "they were not here" -- and a chart that
    # renders the two identically is quietly lying about the emptiest part of itself.
    counted: bool
    level: int  # 0 for nothing, then 1..SHADES


def spanning(now: float, weeks: int) -> tuple[date, date]:
    """The first and last day of a chart of the last ``weeks`` weeks, ending today.

    The start is pulled back to a Monday so every column of the chart is a whole week and the
    weekday rows line up -- otherwise each row would hold a different weekday depending on
    which day of the week the chart happened to be opened on.
    """
    last = office_day(now)
    start = last - timedelta(days=7 * max(1, weeks) - 1)
    return start - timedelta(days=start.weekday()), last


def calendar(
    visits: Sequence[Visit], first: date, last: date, epoch: float, now: float
) -> list[Day]:
    """One :class:`Day` per day from ``first`` to ``last`` inclusive, none skipped.

    Every day is present even when nothing happened on it: the chart's shape is a grid of days,
    and a missing box would silently shift every box after it into the wrong column.

    A visit spanning a boundary is split across the days it touches rather than credited to the
    one it began on, so a long evening reads as the two days it actually covered.
    """
    measured: list[tuple[date, float, int, bool]] = []
    day = first
    while day <= last:
        start, end = day_starting(day), day_starting(day + timedelta(days=1))
        held = 0.0
        counted = 0
        for visit in visits:
            overlap = visit.overlap(start, end, now)
            if overlap > 0:
                held += overlap
                counted += 1
        measured.append((day, held, counted, end > epoch))
        day += timedelta(days=1)
    return _shaded(measured)


def _shaded(measured: Sequence[tuple[date, float, int, bool]]) -> list[Day]:
    """Put each day on a scale of 0..SHADES against the busiest day in the same chart."""
    busiest = max((seconds for _, seconds, _, _ in measured), default=0.0)
    return [
        Day(
            day=day,
            seconds=seconds,
            visits=visits,
            counted=counted,
            level=_level(seconds, busiest),
        )
        for day, seconds, visits, counted in measured
    ]


def _level(seconds: float, busiest: float) -> int:
    """Which shade a day gets: 0 for nothing, then evenly up to the busiest day.

    Any time at all reaches level 1, so a ten-minute visit is a visible box rather than
    rounding away to look like a day off -- being here briefly is the thing the chart is for.
    """
    if seconds <= 0 or busiest <= 0:
        return 0
    return min(SHADES, 1 + int(SHADES * seconds / busiest))


def on_day(passages: Iterable[Passage], day: date) -> list[Passage]:
    """Every crossing that falls within one office day, in the order it happened."""
    start, end = day_starting(day), day_starting(day + timedelta(days=1))
    within = [passage for passage in passages if start <= passage.at < end]
    return sorted(within, key=lambda passage: passage.at)


def held_on(visits: Sequence[Visit], day: date, now: float) -> float:
    """How long somebody was in the room on one day, counting only that day's share of it."""
    start, end = day_starting(day), day_starting(day + timedelta(days=1))
    return sum(visit.overlap(start, end, now) for visit in visits)
