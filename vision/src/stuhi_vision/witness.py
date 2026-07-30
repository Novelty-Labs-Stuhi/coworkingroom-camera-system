"""Who was seen leaving, so the camera that counts an exit can name it.

The two cameras answer different questions. The doorway camera sees the doorframe, so it can
tell a passage from someone crossing the room behind it, and which way they went -- but it
watches people leave from *behind*, and the back of a head is not recognisable. The room
camera sees a leaver walk straight at the lens, face first, and so knows exactly who it is --
but everyone in its view is at the near edge the whole time, so it is a poor judge of whether
anyone passed through at all.

So one counts and the other names. The room camera records each face it sees walking towards
it here; the doorway camera, on committing an exit it could not name itself, claims the most
recent one. A claim consumes the name, so two people leaving one after another cannot both be
recorded as the first.

This is deliberately a name and a timestamp, not an identity model of its own: the recogniser
has already done that work, and duplicating any of it here would give two answers to the
question of who someone is.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

# How long before an exit a leaver may have been seen by the room camera. They pass that
# camera first, walk to the door, and the doorway camera only commits once their track has
# been gone for a moment -- so the sighting always comes first, by a second or several.
# Wide enough for a slow walk; narrow enough that a name cannot survive until the next person.
_WINDOW_SECONDS = 15.0


@dataclass(frozen=True, slots=True)
class Leaver:
    """A face the room camera saw walking towards it."""

    name: str
    score: float
    timestamp: float


class LeavingWitness:
    """Names recently seen heading out, waiting to be attached to an exit."""

    def __init__(self, window_seconds: float = _WINDOW_SECONDS) -> None:
        self._window = window_seconds
        self._seen: list[Leaver] = []
        # Both cameras run their own pipeline thread: one writes here, the other claims.
        self._lock = threading.RLock()

    def note(self, name: str, score: float, timestamp: float) -> None:
        with self._lock:
            self._seen.append(Leaver(name=name, score=score, timestamp=timestamp))

    def claim(self, timestamp: float) -> str | None:
        """Take the name of whoever was seen leaving nearest this moment, if anyone was."""
        with self._lock:
            self._forget_stale(timestamp)
            if not self._seen:
                return None
            nearest = min(self._seen, key=lambda leaver: abs(leaver.timestamp - timestamp))
            if abs(nearest.timestamp - timestamp) > self._window:
                return None
            self._seen.remove(nearest)   # consumed: the next exit is a different person
            return nearest.name

    def waiting(self) -> list[str]:
        """Names noted and not yet claimed -- for the status page and for tests."""
        with self._lock:
            return [leaver.name for leaver in self._seen]

    def _forget_stale(self, now: float) -> None:
        self._seen = [
            leaver for leaver in self._seen if abs(now - leaver.timestamp) <= self._window
        ]
