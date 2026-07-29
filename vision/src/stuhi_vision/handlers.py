"""Commit a doorway crossing into the occupancy ledger.

A crossing is the only thing that marks someone in or out. When one fires, the
Doorkeeper takes that track's accumulated session and:

* rejects it if the track is too young (anti-flicker persistence gate);
* on the way IN, resolves the identity from the track's running best face match (or a
  fresh guest label when nothing matched) and records it with the entry embedding;
* on the way OUT, asks the ledger to link the body embedding to whoever is inside, since
  a person walking away shows no face.

It returns a :class:`~.domain.Sighting` carrying the evidence -- score, outcome, face crop
-- so the crossing can be announced and, if unrecognised, labelled later.
"""

from __future__ import annotations

from .domain import Crossing, Direction, Outcome, Sighting
from .identity import Decision
from .ledger import Ledger
from .sessions import SessionManager, TrackSession


class Doorkeeper:
    """Turns crossings into ledger entries/exits, gated by track persistence."""

    def __init__(
        self,
        sessions: SessionManager,
        ledger: Ledger,
        min_track_age: int,
        camera: str = "",
    ) -> None:
        self._sessions = sessions
        self._ledger = ledger
        self._min_track_age = min_track_age
        self._camera = camera
        self._guests = 0

    def commit(self, crossing: Crossing) -> Sighting | None:
        """Apply a crossing; returns the sighting, or ``None`` if it was rejected."""
        session = self._sessions.pop(crossing.track_id)
        if session is None or session.age < self._min_track_age:
            return None  # flicker / not a confident person pass-through

        decision = session.identity.decide()
        if crossing.direction is Direction.IN:
            name = self._enter(session, decision, crossing.timestamp)
        else:
            name = self._exit(session, decision, crossing.timestamp)

        return Sighting(
            timestamp=crossing.timestamp,
            direction=crossing.direction,
            name=name,
            score=decision.score,
            outcome=decision.outcome,
            face_embedding=session.face_embedding,
            face_crop=session.face_crop,
        )

    def _enter(self, session: TrackSession, decision: Decision, timestamp: float) -> str:
        name = decision.name if decision.outcome is Outcome.NAMED else self._new_guest()
        embedding = session.entry_embedding
        if embedding is None:
            embedding = session.body_embedding  # no face frame; use the sharpest body
        self._ledger.enter(name, embedding, timestamp, self._camera)
        return name

    def _exit(self, session: TrackSession, decision: Decision, timestamp: float) -> str | None:
        """Leave, named by the face when this camera could see one.

        A camera facing people as they leave recognises them exactly as one facing people
        arriving does. Passing that name to the ledger is the point of having a camera per
        direction; previously it was computed and then thrown away, and the exit fell back
        to a body-embedding match that usually resolved to nothing.
        """
        named = decision.name if decision.outcome is Outcome.NAMED else None
        return self._ledger.exit(session.body_embedding, timestamp, self._camera, named)

    def _new_guest(self) -> str:
        self._guests += 1
        return f"guest-{self._guests}"
