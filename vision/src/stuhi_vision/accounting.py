"""Does the count add up? Two questions asked of the recorded crossings.

The count of who is in the room is the point of the whole system, and it is only ever as good
as the crossings behind it. Two things can quietly ruin it, and neither shows up as an error:

**An entry with no exit.** Somebody walked in, walked out, and the walking out was never
registered -- so they stay "inside" for ever and the count climbs. The rule is simple and worth
stating plainly: *every entry must have a matching exit*. When one does not, the interesting
thing is not the entry, it is what the system saw and threw away afterwards. That is recorded
now, in the passages table, so the frames it refused can be handed over instead of a shrug.

**An exit the room renamed.** An exit is named from its own face where it can be, and otherwise
by matching against the people known to be inside. That second rule is what makes a poor camera
usable, and it is also the one that can put the wrong name on a departure -- if it is wrong, two
people are wrong at once, since somebody else is now left inside who has gone. So those exits
are paired with the entry they were matched to and offered for checking together: the pair is
the unit, because confirming one without the other proves nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

# An entry is expected to be followed by its exit within the working day. Beyond that the two
# are not a pair in any useful sense -- somebody who entered yesterday and left today is a
# different fact from a lost exit, and pairing them would hide the lost one.
_SAME_DAY = 24 * 60 * 60


@dataclass(frozen=True, slots=True)
class Crossing:
    """One recorded entry or exit, as the accounting needs it."""

    timestamp: float
    name: str
    direction: str
    camera: str = ""
    named_by: str = ""
    natural: str = ""


@dataclass(frozen=True, slots=True)
class Unmatched:
    """Somebody the record says came in and never left."""

    name: str
    entered_at: float
    # What the system saw at that doorway afterwards and did not count. This is the evidence
    # for the missing exit: usually it is here, refused for a reason now visible.
    missed: tuple[dict, ...]

    @property
    def readable(self) -> str:
        return (
            f"{self.name} entered and never left"
            f"; {len(self.missed)} passage(s) were seen afterwards and not counted"
        )


@dataclass(frozen=True, slots=True)
class Renamed:
    """An exit the room renamed: what it looked like, and who it was matched to."""

    exit_at: float
    given: str        # the name recorded
    natural: str      # what the exit's own evidence said, if anything
    entry_at: float | None   # the entry it was matched to, if that person has one

    @property
    def readable(self) -> str:
        was = self.natural or "nobody"
        return f"exit named {self.given} by the room; its own evidence said {was}"


def unmatched_entries(crossings: list[Crossing], since: float, until: float) -> list[Unmatched]:
    """Entries in the window with no exit after them.

    An exit closes that person's *most recent* open entry, not their oldest. Consider what a
    lost exit actually looks like: in, [exit missed], in, out. The exit that was lost is the
    one belonging to the first entry, and closing the oldest first would pair the recorded
    exit with it and flag the second -- pointing at the wrong moment, and at footage where
    nothing went wrong.
    """
    open_entries: dict[str, list[float]] = {}
    for crossing in sorted(crossings, key=lambda c: c.timestamp):
        if crossing.timestamp < since or crossing.timestamp > until:
            continue
        if crossing.direction == "in":
            open_entries.setdefault(crossing.name, []).append(crossing.timestamp)
        elif open_entries.get(crossing.name):
            open_entries[crossing.name].pop()

    return [
        Unmatched(name=name, entered_at=entered, missed=())
        for name, times in open_entries.items()
        for entered in times
    ]


def with_missed(unmatched: list[Unmatched], refused: list[dict], until: float) -> list[Unmatched]:
    """Attach to each unmatched entry the passages seen afterwards and not counted.

    Only up to the next thing that *was* counted for that person, and only within the day: a
    refusal an hour later is not evidence about this entry, and offering it as such would send
    somebody to look at the wrong footage.
    """
    attached = []
    for entry in unmatched:
        window_end = min(entry.entered_at + _SAME_DAY, until)
        near = tuple(
            passage
            for passage in refused
            if entry.entered_at <= passage["timestamp"] <= window_end
        )
        attached.append(
            Unmatched(name=entry.name, entered_at=entry.entered_at, missed=near)
        )
    return attached


def renamed_exits(crossings: list[Crossing], since: float, until: float) -> list[Renamed]:
    """Exits the room named, paired with the entry of whoever they were matched to.

    An exit named "pool" is one the occupancy rule decided: its own face either said nothing or
    said somebody else. Both halves of that pairing have to be right, so both are offered.
    """
    entries: dict[str, list[float]] = {}
    for crossing in sorted(crossings, key=lambda c: c.timestamp):
        if crossing.direction == "in":
            entries.setdefault(crossing.name, []).append(crossing.timestamp)

    found = []
    for crossing in sorted(crossings, key=lambda c: c.timestamp):
        if crossing.direction != "out" or crossing.named_by != "pool":
            continue
        if crossing.timestamp < since or crossing.timestamp > until:
            continue
        theirs = [when for when in entries.get(crossing.name, []) if when <= crossing.timestamp]
        found.append(
            Renamed(
                exit_at=crossing.timestamp,
                given=crossing.name,
                natural=crossing.natural,
                entry_at=theirs[-1] if theirs else None,
            )
        )
    return found
