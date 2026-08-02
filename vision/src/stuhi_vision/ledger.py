"""The occupancy ledger -- the live set of who is currently inside.

Entries are named by the face recogniser and stored with one body embedding (the crop
from the clearest-face frame). An exit is linked to whoever is inside, in order of how
much the evidence is worth:

* the name a camera read off a face and passed in;
* **the exit's own face, matched against the people inside only.** This is a much easier
  question than the one the recogniser answers at the door. There, a face competes with
  everybody ever enrolled and has to clear a threshold that keeps strangers out. Here the
  answer is almost certainly one of two or three people the ledger already knows are in the
  room, so a face too poor to win open-set can still win outright among three -- which is
  what makes a low-resolution camera usable for exits;
* the body embedding: one candidate inside -> elimination, several -> the closest occupant
  wins only if it clears a threshold AND beats the runner-up by a margin.

Anything unresolved is recorded as unattributed rather than guessed.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, replace

import numpy as np

from .domain import Direction, Event, EventSink
from .recognition.embeddings import cosine

# An exit nobody could be matched to. Recorded rather than dropped, so the ledger's own
# history shows the gap instead of the occupancy count quietly climbing forever.
UNATTRIBUTED = "unknown"

_log = logging.getLogger(__name__)


@dataclass(slots=True)
class Occupant:
    name: str
    entered_at: float
    body_embedding: np.ndarray | None  # None when no body crop was embeddable at entry


class Ledger:
    """In-memory occupancy, journalled to an :class:`EventSink` as it changes."""

    def __init__(
        self,
        sink: EventSink,
        exit_similarity: float,
        exit_margin: float,
        faces=None,
        face_similarity: float = 0.22,
    ) -> None:
        self._sink = sink
        self._similarity = exit_similarity
        self._margin = exit_margin
        # Ranks a face against every enrolled name (the gallery's own ``rank``). Only the
        # people inside are considered, so the threshold can be far lower than the door's:
        # beating two or three known candidates is a much weaker claim than beating everybody.
        self._faces = faces
        self._face_similarity = face_similarity
        self._inside: dict[str, Occupant] = {}
        # Occupancy is one fact about one room, but with a camera per direction two
        # pipelines commit into it from their own threads. Without this, an entry and an
        # exit landing together could read a half-updated set of occupants.
        self._lock = threading.RLock()

    @property
    def occupancy(self) -> list[str]:
        with self._lock:
            return sorted(self._inside)

    def enter(
        self,
        name: str,
        body_embedding: np.ndarray | None,
        timestamp: float,
        camera: str = "",
    ) -> None:
        with self._lock:
            self._inside[name] = Occupant(name, timestamp, body_embedding)
            self._sink.record(Event(timestamp, name, Direction.IN, camera))

    def rename(self, old: str, new: str) -> bool:
        """Follow a corrected name for somebody currently inside. False if they are not.

        Occupancy is keyed by name, so a rename that skipped it would leave the old spelling
        inside for ever: their exit would arrive under the new name, match nobody, and the
        count would never come back down.
        """
        with self._lock:
            occupant = self._inside.pop(old, None)
            if occupant is None:
                return False
            self._inside[new] = replace(occupant, name=new)
            return True

    def exit(
        self,
        body_embedding: np.ndarray | None,
        timestamp: float,
        camera: str = "",
        name: str | None = None,
        face_embedding: np.ndarray | None = None,
    ) -> str | None:
        """Remove an occupant and journal the exit. Returns who left, if it can be told.

        ``name`` is the identity the *face* gave, when this camera could see one -- which is
        the case a camera facing outward is there for. It is trusted over the body-embedding
        match, which exists only for cameras that see people leaving from behind.

        An exit that cannot be attributed is still recorded, under ``unknown``: silently
        writing nothing meant occupancy only ever grew, so the count drifted upward
        permanently and no amount of walking out could correct it.
        """
        with self._lock:
            # What this exit's own evidence says, with no help from the room. Kept even when
            # the pool later overrules it, because the two disagreeing is the case worth
            # checking: it means the exit looked like one person and the room said another.
            natural = name if name in self._inside else None
            named_by = "face" if natural else ""

            resolved = natural
            if resolved is None:
                resolved = self._among_occupants(face_embedding)
                named_by = "pool" if resolved else named_by
            if resolved is None:
                resolved = self._only_occupant()
                named_by = "only" if resolved else named_by
            if resolved is None:
                resolved = self._attribute(body_embedding)
                named_by = "body" if resolved else named_by
            if resolved is None:
                resolved = self._longest_inside()
                named_by = "longest" if resolved else named_by
            if resolved is not None:
                del self._inside[resolved]
            self._sink.record(
                Event(
                    timestamp,
                    resolved or UNATTRIBUTED,
                    Direction.OUT,
                    camera,
                    named_by=named_by or "nobody",
                    natural=natural or "",
                )
            )
            return resolved

    def _longest_inside(self) -> str | None:
        """Whoever has been in longest, when nothing else could name the exit.

        Somebody walked out: that much is established by the doorway before this is asked.
        Refusing to say who leaves them inside for ever, and a room that fills and never empties
        is what this replaced -- dozens of people credited with every hour of every day, which
        is both wrong and obviously wrong.

        The longest-present is the safest guess. They have had the most opportunity to leave
        unseen, and an unclosed entry distorts their figures most, while somebody who arrived a
        minute ago is the least likely to be walking out now. It remains a guess: the crossing
        is recorded as named by "longest", so hours resting on it can be told from the rest.
        """
        if not self._inside:
            return None
        who = min(self._inside, key=lambda name: self._inside[name].entered_at)
        _log.info("exit matched nobody; closing the longest visit: %s", who)
        return who

    def _among_occupants(self, face: np.ndarray | None) -> str | None:
        """Which of the people inside this face belongs to, if any of them clearly.

        Deliberately a low bar: the field is two or three people who are known to be in the
        room, not everybody ever enrolled. What still has to hold is that one of them beats
        the others by a margin -- a face that suits two occupants equally names neither.
        """
        if face is None or self._faces is None or not self._inside:
            return None
        inside = [match for match in self._faces(face) if match.name in self._inside]
        if not inside:
            return None
        best = inside[0]
        if best.score < self._face_similarity:
            return None
        runner_up = inside[1].score if len(inside) > 1 else 0.0
        return best.name if best.score - runner_up >= self._margin else None

    def _only_occupant(self) -> str | None:
        """The one person inside, when there is only one. Not a match -- an elimination.

        Kept separate from the body comparison so the record says which it was. Filed as a body
        match, an exit with no body embedding at all looked like evidence it never had.
        """
        return next(iter(self._inside)) if len(self._inside) == 1 else None

    def _attribute(self, query: np.ndarray | None) -> str | None:
        if not self._inside:
            return None
        if query is None:
            return None  # cannot match a missing body against several candidates
        ranked = sorted(
            (
                (cosine(query, occ.body_embedding), occ.name)
                for occ in self._inside.values()
                if occ.body_embedding is not None
            ),
            reverse=True,
        )
        if not ranked:
            return None
        best_score, best_name = ranked[0]
        runner_up = ranked[1][0] if len(ranked) > 1 else -1.0
        if best_score >= self._similarity and (best_score - runner_up) >= self._margin:
            return best_name
        return None  # too weak or too ambiguous
