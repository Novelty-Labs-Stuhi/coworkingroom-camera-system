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

Two different "start" moments live here, and they are easy to confuse:

* **04:30** is where each *day* begins, above. It is about which day a visit belongs to.
* the **epoch** is when *counting* began for this deployment (:func:`epoch_for`) -- fixed the
  first time it is asked for, so past totals do not silently rewrite themselves each morning.
  Nothing before it is counted at all.

A window shorter than the record is shown anyway, labelled with how much of itself it covers
(:func:`covered`). It is only the first five minutes after the epoch that show nothing.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# The office day starts here. Anybody still in the room at 04:30 is counted against the day
# that is ending, which is why the streak of somebody who works past midnight is not broken.
DAY_STARTS_AT = time(4, 30)

# ...at half past four *in the office*, which is not where the server is. The machine runs on
# UTC, so taking the day boundary from its clock put it at 07:30 Helsinki -- three hours late,
# which moves a morning's visits into the previous day and would break a streak on the strength
# of where a computer happens to live. Stated explicitly rather than left to the environment.
OFFICE = ZoneInfo("Europe/Helsinki")

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


def recorded_visits(passages: Iterable[tuple[str, float, str]], since: float) -> list[Visit]:
    """Visits from the rows the event store keeps, ignoring everything before ``since``.

    The one place the store's tuples become visits. Every page asking about time in the room
    asks the same question of the same log, and two spellings of this step would eventually
    disagree about something -- which window the epoch cuts, most likely.
    """
    return visits_from(
        Passage(name=name, at=at, direction=direction)
        for name, at, direction in passages
        if at >= since
    )


def totals(visits: Sequence[Visit], epoch: float, now: float) -> dict[Window, float]:
    """One person's seconds in the room over each window, as their own figures.

    Clipped to the epoch as well as to the window: "this year" cannot honestly include time
    from before anybody was counting.
    """
    figures: dict[Window, float] = {}
    for window in ("day", "week", "month", "year", "all"):
        start, end = window_bounds(window, 0, now)
        figures[window] = sum(visit.overlap(max(start, epoch), end, now) for visit in visits)
    return figures


def counting_from(directory: Path) -> float:
    """When counting began for this deployment, from the file beside its review data."""
    return epoch_for(directory / "stats-epoch.json")


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
    starts = day_starting(datetime.now(OFFICE).date())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"from": starts}, indent=2), encoding="utf-8")
    return starts


def window_days(window: Window) -> int:
    """How many office days one of these windows is, for the windows that are a fixed length."""
    return {"day": 1, "week": 7, "month": 30, "year": 365}.get(window, 1)


# How long after counting begins before any board says anything: long enough that the very
# first moments of a deployment read as "just started" rather than "nobody was here", short
# enough that nobody waits for it.
GRACE_SECONDS = 5 * 60.0


def available_from(window: Window, epoch: float) -> float:
    """When a window first has anything to say: five minutes after counting began.

    It used to be one whole window -- a week's board waited seven days, a year's waited a year.
    The reasoning was that a week's board on its second day looks like a week's while being a
    day's. The reasoning was right and the remedy was wrong: it left three of the six tabs
    blank on a working system with hundreds of recorded crossings behind them, and the year tab
    blank until 2027. Hiding real figures to avoid mislabelling them is the worse trade.

    So every window shows what it has, and says how much of itself it actually covers -- see
    :func:`covered`. A partial week is labelled a partial week instead of being withheld.
    """
    return epoch + GRACE_SECONDS


def covered(window: Window, offset: int, epoch: float, now: float) -> tuple[int, int]:
    """How many office days of this window the record covers, and how many it has.

    ``(3, 7)`` means "three days of this week are behind these figures". Equal numbers mean the
    window is whole. Zero days in the window means the question does not apply -- "all time" and
    a streak are not a fixed length, so there is no fraction of them to be short of.
    """
    if window in ("all", "streak"):
        return 0, 0
    whole = window_days(window)
    start, end = window_bounds(window, offset, now)
    first = office_day(max(start, epoch))
    last = office_day(min(end, now))
    return max(0, (last - first).days + 1), whole


def office_day(moment: float) -> date:
    """Which office day a moment belongs to, with the day starting at 04:30 in the office."""
    when = datetime.fromtimestamp(moment, OFFICE)
    return (when - timedelta(hours=DAY_STARTS_AT.hour, minutes=DAY_STARTS_AT.minute)).date()


def day_starting(day: date) -> float:
    """The moment an office day begins, as a timestamp."""
    return datetime.combine(day, DAY_STARTS_AT, tzinfo=OFFICE).timestamp()


def window_bounds(window: Window, offset: int, now: float) -> tuple[float, float]:
    """The start and end of a window, ``offset`` periods back from the one running now.

    Every window is aligned to an office day boundary, so "last week" is seven whole office
    days rather than a rolling 168 hours -- otherwise the same visit would drift in and out of
    the total depending on the hour the page was opened.
    """
    today = office_day(now)
    if window == "all":
        return 0.0, now

    length = window_days(window)
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
