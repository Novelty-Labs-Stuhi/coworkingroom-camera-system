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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .domain import Crossing, Direction, Outcome, Sighting
from .identity import Decision
from .ledger import Ledger
from .sessions import SessionManager, TrackSession
from .witness import LeavingWitness

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Enrolment:
    """Where a newly invented identity is written.

    The three travel together and are meaningless apart: a face enrolled without its name
    recorded as provisional looks like somebody a human chose, and a name recorded without
    the face enrolled cannot be matched again. Grouping them says that, and keeps the
    Doorkeeper's signature about what it does rather than about where things are kept.
    """

    # Where an unnamed identity's own face is kept. Deliberately *not* the named gallery:
    # an uncertain face must be unable to reach the references a real person is recognised
    # by, and a separate store enforces that by construction rather than by remembering to.
    strangers: object | None = None
    directory: Path | None = None
    provisional: object | None = None


class Doorkeeper:
    """Turns crossings into ledger entries/exits, gated by track persistence."""

    def __init__(
        self,
        sessions: SessionManager,
        ledger: Ledger,
        min_track_age: int,
        camera: str = "",
        witness: LeavingWitness | None = None,
        enrolment: Enrolment | None = None,
    ) -> None:
        self._sessions = sessions
        self._ledger = ledger
        self._min_track_age = min_track_age
        self._camera = camera
        self._witness = witness
        # Where a new identity's face goes. Without it an unrecognised person is unrecognisable
        # again next time, so they can never be matched on the way out.
        # Where an unrecognised arrival's identity is written. Without it they are
        # unrecognisable again next time, so they can never be matched on the way out.
        self._enrolment = enrolment or Enrolment()

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
        introduced = False
        if crossing.direction is Direction.IN:
            name, introduced = self._enter(session, decision, crossing.timestamp)
        else:
            name = self._exit(session, decision, crossing.timestamp)

        return Sighting(
            timestamp=crossing.timestamp,
            direction=crossing.direction,
            name=name,
            score=decision.score,
            outcome=decision.outcome,
            introduced=introduced,
            face_embedding=session.face_embedding,
            face_crop=session.face_crop,
        )

    def _enter(
        self, session: TrackSession, decision: Decision, timestamp: float
    ) -> tuple[str, bool]:
        """Somebody came in. Recognised or not, they get an identity that lasts.

        Returns the name and whether this crossing *invented* it -- the one moment worth
        interrupting a human for, and the only one that happens exactly once per person.
        """
        if decision.outcome is Outcome.NAMED:
            name = decision.name
            introduced = False
        else:
            name, introduced = self._someone_new(session, timestamp)
        embedding = session.entry_embedding
        if embedding is None:
            embedding = session.body_embedding  # no face frame; use the sharpest body
        self._ledger.enter(name, embedding, timestamp, self._camera)
        return name, introduced

    def _someone_new(self, session: TrackSession, timestamp: float) -> tuple[str, bool]:
        """Somebody the recogniser could not name: met before, or genuinely new?

        Asked in that order, because the two answers have very different consequences. If this
        is a stranger we have already met, they keep the identity they were given -- so their
        exit can be attributed and their entry closes, which is what keeps the count right. It
        is only when nobody is recognisable that a new identity is created, and it is created
        rather than forced onto the nearest name the gallery happens to hold.

        The bar for "same stranger" is lower than the bar for "this is Ilari" on purpose. Being
        cautious here bought nothing and cost an identity per visit: a stranger who came through
        eight times became eight people, none of whom could ever be recognised, and the room
        filled with occupants whose exits could never be matched.

        Nothing here touches the named gallery. Returns the name and whether it is new.

        Two things were wrong with the old guest label. It came from a counter that starts at
        one in every process, so a restart re-issued names that already existed and two
        different people shared one. And the face was never enrolled, so the same person coming
        back was unrecognised again and got yet another name -- which is how dozens of guests
        end up in the room at once, none of them ever leaving, because an exit can only be
        matched to somebody the recogniser can find.

        Enrolled, an unnamed identity behaves like a person: the next arrival matches it, their
        exit can be attributed, and it can later be given a real name or merged into one.
        """
        strangers = self._enrolment.strangers
        if strangers is not None:
            met_before = strangers.recognise(session.face_embedding)
            if met_before is not None:
                _log.info("%s recognised %s, somebody still unnamed", self._camera, met_before)
                return met_before, False

        name = f"guest-{datetime.fromtimestamp(timestamp).strftime('%m%d-%H%M%S')}"
        if strangers is not None:
            strangers.remember(name, session.face_embedding)
        if self._enrolment.provisional is not None:
            self._enrolment.provisional.add(name)
        _log.info("%s met somebody new: %s", self._camera, name)
        return name, True

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
