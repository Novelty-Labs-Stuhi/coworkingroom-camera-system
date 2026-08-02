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

import logging
from datetime import datetime

from .domain import Crossing, Direction, Outcome, Sighting
from .identity import Decision
from .ledger import Ledger
from .sessions import SessionManager, TrackSession
from .witness import LeavingWitness

_log = logging.getLogger(__name__)


class Doorkeeper:
    """Turns crossings into ledger entries/exits, gated by track persistence."""

    def __init__(
        self,
        sessions: SessionManager,
        ledger: Ledger,
        min_track_age: int,
        camera: str = "",
        witness: LeavingWitness | None = None,
        gallery=None,
        gallery_dir=None,
    ) -> None:
        self._sessions = sessions
        self._ledger = ledger
        self._min_track_age = min_track_age
        self._camera = camera
        self._witness = witness
        # Where a new identity's face goes. Without it an unrecognised person is unrecognisable
        # again next time, so they can never be matched on the way out.
        self._gallery = gallery
        self._gallery_dir = gallery_dir

    def commit(self, crossing: Crossing) -> Sighting | None:
        """Apply a crossing; returns the sighting, or ``None`` if it was rejected."""
        session = self._sessions.pop(crossing.track_id)
        if session is None or session.age < self._min_track_age:
            # Said out loud, because a rejection here is indistinguishable in the record from
            # a passage that was never recognised at all, and the two need different fixes.
            seen = "no session" if session is None else f"seen {session.age} frames"
            _log.info(
                "%s crossing not counted: track %s, %s", self._camera, crossing.track_id, seen
            )
            return None

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
        """Somebody came in. Recognised or not, they get an identity that lasts."""
        if decision.outcome is Outcome.NAMED:
            name = decision.name
        else:
            name = self._someone_new(session, timestamp)
        embedding = session.entry_embedding
        if embedding is None:
            embedding = session.body_embedding  # no face frame; use the sharpest body
        self._ledger.enter(name, embedding, timestamp, self._camera)
        return name

    def _someone_new(self, session: TrackSession, timestamp: float) -> str:
        """A fresh identity for somebody nobody recognised -- and *enrol their face under it*.

        Two things were wrong with the old guest label. It came from a counter that starts at
        one in every process, so a restart re-issued names that already existed and two
        different people shared one. And the face was never enrolled, so the same person coming
        back was unrecognised again and got yet another name -- which is how dozens of guests
        end up in the room at once, none of them ever leaving, because an exit can only be
        matched to somebody the recogniser can find.

        Enrolled, an unnamed identity behaves like a person: the next arrival matches it, their
        exit can be attributed, and it can later be given a real name or merged into one.
        """
        name = f"guest-{datetime.fromtimestamp(timestamp).strftime('%m%d-%H%M%S')}"
        if session.face_embedding is not None and self._gallery is not None:
            self._gallery.add(name, session.face_embedding)
            self._gallery.save(self._gallery_dir)
            _log.info("%s enrolled %s as somebody new", self._camera, name)
        return name

    def _exit(self, session: TrackSession, decision: Decision, timestamp: float) -> str | None:
        """Leave, named by the best evidence available, in order of how direct it is.

        The camera that can see the doorframe -- and so is the one that can tell an exit from
        background traffic -- is watching people leave from behind. Its own face match is
        therefore usually empty, and a body embedding rarely resolves. The room camera saw
        that same person walk at it face-first moments earlier, so its name is asked for
        second: better evidence than a body embedding, and worse than a face seen here.
        """
        named = decision.name if decision.outcome is Outcome.NAMED else None
        if named is None and self._witness is not None:
            named = self._witness.claim(timestamp)
            if named is not None:
                # Printed because this is the one step no single camera can verify: whether
                # the handover actually happened is otherwise invisible in the log.
                _log.info("exit named %s by the other camera", named)
        # The face goes too, even when it named nobody at the door: the ledger asks a much
        # easier question of it -- which of the people inside is this -- and a face too poor
        # to win against the whole gallery can win outright among three known candidates.
        return self._ledger.exit(
            session.body_embedding,
            timestamp,
            self._camera,
            named,
            face_embedding=session.face_embedding,
        )


class Identifier:
    """A camera whose job is *who*, not how many.

    Same shape as :class:`Doorkeeper` -- it takes a crossing and returns a sighting -- but it
    writes nothing to the ledger. It only records the name for the doorway camera to claim.

    That division is what stops one passage being counted twice. Both cameras see every
    passage, so if both committed, one person leaving would be two exits; and this camera is
    the weaker judge of whether a passage happened at all, since everybody in its view is at
    the near edge and it has no doorframe to go by. What it is unmatched at is recognising the
    face of somebody walking towards it.

    The sighting still comes back, so the clip and the face crop reach the review queue: an
    unrecognised leaver is exactly the footage worth labelling.
    """

    def __init__(
        self,
        sessions: SessionManager,
        witness: LeavingWitness,
        min_track_age: int,
        camera: str = "",
    ) -> None:
        self._sessions = sessions
        self._witness = witness
        self._min_track_age = min_track_age
        self._camera = camera

    def commit(self, crossing: Crossing) -> Sighting | None:
        session = self._sessions.pop(crossing.track_id)
        if session is None or session.age < self._min_track_age:
            return None

        decision = session.identity.decide()
        if crossing.direction is Direction.OUT and decision.outcome is Outcome.NAMED:
            self._witness.note(decision.name, decision.score, crossing.timestamp)
            _log.info(
                "%s saw %s leaving (%.2f)", self._camera, decision.name, decision.score
            )

        return Sighting(
            timestamp=crossing.timestamp,
            direction=crossing.direction,
            name=decision.name if decision.outcome is Outcome.NAMED else None,
            score=decision.score,
            outcome=decision.outcome,
            face_embedding=session.face_embedding,
            face_crop=session.face_crop,
        )
