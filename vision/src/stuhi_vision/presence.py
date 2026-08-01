"""Time spent in the room, derived from the entries and exits already recorded.

Nothing new is stored. The event log holds every entry and exit with the moment it happened,
which is enough to answer every question here, and deriving rather than storing means a
correction to a name -- or a merge of two spellings -- shows up in the figures immediately.
A second table of totals would have to be rebuilt after every correction, and would be wrong
in the meantime.

A **visit** is an entry paired with the next exit by the same person. Unpaired halves are the
normal case rather than an error, and each is treated as what it is:

* an entry with no exit yet is somebody *still inside*, so it counts up to now;
* an exit with no entry before it is a passage the system attributed wrongly, or one whose
  entry it missed. It contributes no time, because there is no moment to measure from -- and
  inventing one would put invented hours on somebody's total.

The office day begins at **04:30**, not midnight: somebody in the room at one in the morning
is finishing the previous day rather than starting a new one, and a boundary at midnight would
split that visit across two days and break a streak that should stand.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

# The office day starts here. Anybody still in the room at 04:30 is counted against the day
# that is ending, which is why the streak of somebody who works past midnight is not broken.
DAY_STARTS_AT = time(4, 30)

Window = str  # "day" | "week" | "month" | "year" | "all"


@dataclass(frozen=True, slots=True)
class Passage:
    """One recorded crossing: who, when, and which way."""

    name: str
    at: float
    direction: str  # "in" or "out"


@dataclass(frozen=True, slots=True)
class Visit:
    """One stay in the room. ``left`` is None while somebody is still there."""

    name: str
    entered: float
    left: float | None

    def seconds(self, until: float) -> float:
        """How long this visit lasted, treating an unfinished one as running until ``until``."""
        end = self.left if self.left is not None else until
        return max(0.0, end - self.entered)

    def overlap(self, start: float, end: float, until: float) -> float:
        """Seconds of this visit that fall inside a window.

        Clipped rather than counted whole: a visit spanning midnight belongs partly to each
        day, and giving it to one of them would make a leaderboard depend on when people
        happen to arrive rather than how long they stay.
        """
        finish = min(self.left if self.left is not None else until, end)
        return max(0.0, finish - max(self.entered, start))


@dataclass(frozen=True, slots=True)
class Standing:
    """One person's place on a leaderboard."""

    name: str
    seconds: float
    visits: int
    days: int          # distinct office days they appeared on, within the window
    still_inside: bool


def visits_from(passages: Iterable[Passage]) -> list[Visit]:
    """Pair each person's entries with their following exits, in time order.

    Two entries in a row means the first visit's exit was missed. It is closed at the second
    entry rather than dropped or left open: they were plainly not in the room twice at once,
    and leaving it open would run one visit into the next and inflate a total without limit.
    """
    open_entry: dict[str, float] = {}
    visits: list[Visit] = []
    for passage in sorted(passages, key=lambda p: p.at):
        if passage.direction == "in":
            if passage.name in open_entry:
                # Their previous exit was missed: close it here rather than let one visit
                # swallow the next.
                visits.append(
                    Visit(passage.name, entered=open_entry[passage.name], left=passage.at)
                )
            open_entry[passage.name] = passage.at
        elif passage.name in open_entry:
            entered = open_entry.pop(passage.name)
            visits.append(Visit(passage.name, entered=entered, left=passage.at))
        # An exit with nothing open contributes nothing: there is no moment to measure from,
        # and choosing one would put invented hours on somebody's total.
    visits.extend(
        Visit(name, entered=entered, left=None) for name, entered in open_entry.items()
    )
    return sorted(visits, key=lambda visit: visit.entered)


def epoch_for(path: Path) -> float:
    """When counting begins, fixed the first time it is asked for and never moved after.

    The figures start at 04:30 on the day this was switched on, not at the start of whatever
    the log happens to contain. Two reasons. The log holds months of crossings recorded while
    the rules were being changed hourly -- counting them would present four days of that as
    somebody's record. And a boundary that moved with the current date would silently rewrite
    every past total each morning.

    Written down rather than computed, because "today" stops being today tomorrow.
    """
    if path.exists():
        return float(json.loads(path.read_text(encoding="utf-8"))["from"])
    starts = datetime.combine(datetime.now().date(), DAY_STARTS_AT).timestamp()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"from": starts}, indent=2), encoding="utf-8")
    return starts


def available_from(window: Window, epoch: float) -> float:
    """When a window first has anything to say.

    A week's leaderboard on its second day is not a week's leaderboard -- it looks like one
    while being a day's, which is worse than showing nothing. So each window waits until one
    of it has actually passed since counting began. The running total and the streak start
    with the epoch itself: both are honest from the first minute, being explicitly "so far".
    """
    if window in ("all", "streak"):
        return epoch
    return epoch + {"day": 1, "week": 7, "month": 30, "year": 365}.get(window, 1) * 24 * 3600


def office_day(moment: float) -> date:
    """Which office day a moment belongs to, with the day starting at 04:30."""
    when = datetime.fromtimestamp(moment)
    return (when - timedelta(hours=DAY_STARTS_AT.hour, minutes=DAY_STARTS_AT.minute)).date()


def day_starting(day: date) -> float:
    """The moment an office day begins, as a timestamp."""
    return datetime.combine(day, DAY_STARTS_AT).timestamp()


def window_bounds(window: Window, offset: int, now: float) -> tuple[float, float]:
    """The start and end of a window, ``offset`` periods back from the one running now.

    Every window is aligned to an office day boundary, so "last week" is seven whole office
    days rather than a rolling 168 hours -- otherwise the same visit would drift in and out of
    the total depending on the hour the page was opened.
    """
    today = office_day(now)
    if window == "all":
        return 0.0, now

    length = {"day": 1, "week": 7, "month": 30, "year": 365}.get(window, 1)
    start_day = today - timedelta(days=length * offset + (length - 1))
    start = day_starting(start_day)
    end = day_starting(today - timedelta(days=length * offset)) + 24 * 3600
    return start, min(end, now)


def leaderboard(
    visits: Sequence[Visit], start: float, end: float, now: float
) -> list[Standing]:
    """Who spent the most time in the room within a window, longest first.

    Only people who appeared in the window are listed: a leaderboard for last week that
    includes somebody who was away all week, on nought hours, is a longer list saying less.
    """
    seconds: dict[str, float] = {}
    counted: dict[str, int] = {}
    days: dict[str, set[date]] = {}
    inside: set[str] = set()
    for visit in visits:
        held = visit.overlap(start, end, now)
        if held <= 0:
            continue
        seconds[visit.name] = seconds.get(visit.name, 0.0) + held
        counted[visit.name] = counted.get(visit.name, 0) + 1
        days.setdefault(visit.name, set()).add(office_day(max(visit.entered, start)))
        if visit.left is None:
            inside.add(visit.name)
    return sorted(
        (
            Standing(
                name=name,
                seconds=held,
                visits=counted[name],
                days=len(days[name]),
                still_inside=name in inside,
            )
            for name, held in seconds.items()
        ),
        key=lambda standing: -standing.seconds,
    )


def streaks(visits: Sequence[Visit], now: float) -> list[Standing]:
    """The run of consecutive office days each person has appeared on, longest first.

    Counted backwards from today, and from yesterday if today has not happened yet -- so a
    streak is not reported broken first thing in the morning before anybody has arrived.
    """
    appeared: dict[str, set[date]] = {}
    for visit in visits:
        appeared.setdefault(visit.name, set()).add(office_day(visit.entered))

    today = office_day(now)
    standings = []
    for name, days in appeared.items():
        last = today if today in days else today - timedelta(days=1)
        run = 0
        while last - timedelta(days=run) in days:
            run += 1
        if run:
            standings.append(
                Standing(name=name, seconds=float(run), visits=len(days), days=run,
                         still_inside=False)
            )
    return sorted(standings, key=lambda standing: -standing.seconds)


def readable(seconds: float) -> str:
    """A duration as somebody would say it: "3 h 12 m", "48 m", "40 s"."""
    if seconds >= 3600:
        hours, rest = divmod(int(seconds), 3600)
        return f"{hours} h {rest // 60} m"
    if seconds >= 60:
        return f"{int(seconds) // 60} m"
    return f"{int(seconds)} s"


def unclosed(visits: Sequence[Visit]) -> list[Visit]:
    """Entries with no exit after them: everybody the record thinks is still in the room.

    This is the list to reach for when a correction is impossible. An exit can only belong to
    somebody who came in and has not left, so naming an exit as somebody the record does not
    have inside means an *earlier* entry carries the wrong name -- either theirs was missed, or
    somebody else's entry was recorded as them. The mistake is in this list, and it is the only
    place it can be.
    """
    return [visit for visit in visits if visit.left is None]


def could_have_left(visits: Sequence[Visit], name: str) -> bool:
    """Whether the record has this person inside, so an exit could be theirs."""
    return any(visit.name == name for visit in unclosed(visits))
