"""What the room holds across a restart, and what it must not hold across a night.

Two faults produced the same symptom -- people recorded as being in the room for days --
and both are about state outliving the thing it describes:

* **The ledger lived only in memory.** A restart forgot everybody inside, so their exits
  matched nobody and were filed as ``unknown``; the entries they should have closed stayed
  open for ever. Every restart therefore *added* permanent occupants. Restoring the ledger
  from the record at startup is what stops that.
* **Nothing ever expired.** With no boundary, one missed exit was wrong indefinitely rather
  than until the end of the day.

The second is what makes the first safe. Restoring occupants means a bad state now survives
a restart too, so it may only be done alongside a boundary that clears it -- which is why
both live in this one module rather than being separately reachable.

**The office day begins at 04:30**, the same boundary the presence figures already use:
somebody in the room at one in the morning is finishing yesterday, not starting today. At
that moment anybody still recorded inside is closed out. It is a guess, but a bounded one,
and it is the difference between a number that is wrong until morning and a number that is
wrong for ever.

Closures are marked ``named_by = "reset"``, so a visit the boundary ended can always be told
from one the doorway actually saw end.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable

from .presence import day_starting, office_day

# How the boundary marks the exits it writes, so they are never mistaken for observed ones.
RESET = "reset"

# A crossing as this module needs it: when, who, which way.
Crossing = tuple[float, str, str]


def open_visits(crossings: Iterable[Crossing]) -> dict[str, float]:
    """Who the record says is inside, and since when, by replaying every crossing in order.

    An entry opens a visit only if that name has none open -- ``setdefault`` rather than
    assignment -- so two entries in a row keep the *earlier* start. That is the reading that
    makes an unclosed visit look as old as it really is, which is what the boundary needs to
    decide it is stale.
    """
    inside: dict[str, float] = {}
    for timestamp, name, direction in crossings:
        if direction == "in":
            inside.setdefault(name, timestamp)
        else:
            inside.pop(name, None)
    return inside


def partition(inside: dict[str, float], day_began: float) -> tuple[dict, dict]:
    """Split who is recorded inside into (restore, close) against the current day's start.

    Somebody whose visit began today may genuinely still be here, so the ledger takes them
    back and their exit can still be attributed. Somebody whose visit began before today
    cannot still be here -- nobody sleeps in the office -- so their visit is closed instead.

    Doing this at startup rather than only on the boundary matters: a restart is the moment
    the record is re-read, and without it a night of downtime would carry yesterday's
    occupants into today untouched.
    """
    restore = {name: since for name, since in inside.items() if since >= day_began}
    close = {name: since for name, since in inside.items() if since < day_began}
    return restore, close


class DayBoundary:
    """Fires once per office day, closing whatever the room is still holding.

    Deliberately idempotent and driven by *being asked*, not by sleeping until a deadline.
    Several callers ask -- startup, every crossing, and a slow tick -- and only the first one
    after the boundary does anything. A timer thread that dies takes its schedule with it;
    three cheap callers that all no-op cannot.
    """

    def __init__(self, ledger, day: int | None = None) -> None:
        self._ledger = ledger
        # The office day already accounted for. Set on the first ``check`` when unknown, so
        # construction does not need a clock passed to it.
        self._day = day
        # One camera per thread, both asking. Without this both can pass the day check
        # before either records it, and the boundary fires twice. The second firing closes
        # an already-empty room, so the harm is one confusing log line rather than a wrong
        # count -- but a "fires once" guarantee that only usually holds is not one.
        self._lock = threading.Lock()

    def check(self, now: float) -> list[str]:
        """Close everyone out if a new office day has started. Returns who was closed."""
        day = office_day(now).toordinal()
        with self._lock:
            if self._day is None:
                self._day = day
                return []
            if day == self._day:
                return []
            self._day = day
        # Outside the lock: the ledger has its own, and holding two at once in one order
        # here and the other order elsewhere is how a deadlock is built.
        return self._ledger.close_all(now, RESET)


def start_of_day(now: float) -> float:
    """The moment the current office day began."""
    return day_starting(office_day(now))
