"""Commit a doorway crossing into the occupancy ledger.

A crossing is the only thing that marks someone in or out. When one fires, the
Doorkeeper takes that track's accumulated session and:

* rejects it if the track is too young (anti-flicker persistence gate);
* on the way IN, records the recognised name (or a fresh guest label) plus the stored
  entry embedding;
* on the way OUT, asks the ledger to link the body embedding to whoever is inside.
"""

from __future__ import annotations

from .domain import Crossing, Direction
from .ledger import Ledger
from .sessions import SessionManager


class Doorkeeper:
    """Turns crossings into ledger entries/exits, gated by track persistence."""

    def __init__(self, sessions: SessionManager, ledger: Ledger, min_track_age: int) -> None:
        self._sessions = sessions
        self._ledger = ledger
        self._min_track_age = min_track_age
        self._guests = 0

    def commit(self, crossing: Crossing) -> tuple[Direction, str | None] | None:
        """Apply a crossing; returns (direction, name) or ``None`` if it was rejected."""
        session = self._sessions.pop(crossing.track_id)
        if session is None or session.age < self._min_track_age:
            return None  # flicker / not a confident person pass-through
        if crossing.direction is Direction.IN:
            return Direction.IN, self._enter(session, crossing.timestamp)
        return Direction.OUT, self._exit(session, crossing.timestamp)

    def _enter(self, session, timestamp: float) -> str:
        name = session.name or self._new_guest()
        embedding = session.entry_embedding
        if embedding is None:
            embedding = session.body_embedding  # no clear-face frame; use the sharpest body
        self._ledger.enter(name, embedding, timestamp)
        return name

    def _exit(self, session, timestamp: float) -> str | None:
        return self._ledger.exit(session.body_embedding, timestamp)

    def _new_guest(self) -> str:
        self._guests += 1
        return f"guest-{self._guests}"
