"""What the room-facing camera can testify about a passage the doorway camera counted.

The doorway camera can tell that *something* covered the doorframe, and from the order of
things which way it was going. What it cannot see is where the person ended up: its view stops
at the frame. The room camera can, and the two halves are different in kind:

* somebody who came **in** *appears* in the room's view and stays there;
* somebody who went **out** walks straight at that lens -- growing, often clipped by the frame
  edge -- and then goes out of frame entirely.

So each counted passage can be asked for a second opinion. This is deliberately corroboration
and not a vote: a passage the room camera cannot confirm is recorded as **unconfirmed**, not
refused. Refusing it would trade a miss for a false count, and a miss of an exit leaves somebody
in the room for ever. Unconfirmed ones are what a human looks at when the count reads wrong.

A confirmation is consumed when it is claimed, so two people arriving one behind the other need
two appearances between them. Without that, one person walking in would confirm every entry the
doorway camera imagined for the next quarter minute.

Nothing here decides who somebody is -- :mod:`.witness` already carries names between the
cameras, and a second answer to that question is worse than none.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

# How far from a crossing the room camera's observation may be and still be about it. The
# person passes one camera and then the other, and the doorway camera only commits once the
# track has been gone a moment -- so the two are always seconds apart, never simultaneous.
# The same fifteen seconds the name handover uses, and for the same reason.
_WINDOW_SECONDS = 15.0
# ...and a little slack the other way, because which camera sees them first depends on which
# way they were walking, not on which camera commits.
_SLACK_SECONDS = 3.0


@dataclass(frozen=True, slots=True)
class Appearing:
    """Somebody the room camera saw arrive in its view."""

    at: float


@dataclass(frozen=True, slots=True)
class Leaving:
    """Somebody the room camera watched walk at it and go out of frame."""

    at: float
    grew: float


class Corroboration:
    """The room camera's observations, waiting for the doorway camera to ask about them.

    Both cameras run their own pipeline thread: one writes here, the other claims.
    """

    def __init__(
        self, window_seconds: float = _WINDOW_SECONDS, slack_seconds: float = _SLACK_SECONDS
    ) -> None:
        self._window = window_seconds
        self._slack = slack_seconds
        self._appeared: list[Appearing] = []
        self._left: list[Leaving] = []
        self._lock = threading.RLock()

    # --- what the room camera saw -------------------------------------------------------
    def appeared(self, at: float) -> None:
        """Somebody came into the room camera's view."""
        with self._lock:
            self._appeared.append(Appearing(at=at))

    def left_frame(self, at: float, grew: float) -> None:
        """Somebody walked at the room camera and went out of its frame."""
        with self._lock:
            self._left.append(Leaving(at=at, grew=grew))

    # --- what the doorway camera asks ---------------------------------------------------
    def confirms_entry(self, at: float) -> Appearing | None:
        """Did anybody appear in the room around the moment this entry was counted?

        Looked for *after* the crossing as well as before: they cross the doorframe and then
        walk into the room, so the appearance usually follows. The slack the other way covers
        a person already inside the room camera's view as they came through.
        """
        with self._lock:
            return self._claim(self._appeared, at, before=self._slack, after=self._window)

    def confirms_exit(self, at: float) -> Leaving | None:
        """Did anybody walk at the room camera and out of frame around this exit?

        Looked for mostly *before* the crossing: they cross the room first, then reach the
        door, and the doorway camera commits last of all.
        """
        with self._lock:
            return self._claim(self._left, at, before=self._window, after=self._slack)

    def _claim(self, seen: list, at: float, before: float, after: float):
        """The nearest observation inside the window, removed so it cannot serve twice."""
        self._forget(seen, at)
        within = [s for s in seen if -before <= s.at - at <= after]
        if not within:
            return None
        nearest = min(within, key=lambda s: abs(s.at - at))
        seen.remove(nearest)
        return nearest

    def _forget(self, seen: list, now: float) -> None:
        """Drop anything too old to be about anything, so the lists cannot grow all day."""
        cutoff = now - (self._window + self._slack)
        seen[:] = [s for s in seen if s.at >= cutoff]

    @property
    def waiting(self) -> tuple[int, int]:
        """Unclaimed appearances and departures. For the log, and for tests."""
        with self._lock:
            return len(self._appeared), len(self._left)
