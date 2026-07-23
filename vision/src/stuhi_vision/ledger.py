"""The occupancy ledger -- the live set of who is currently inside.

Entries are named by the face recogniser and stored with one body embedding (the crop
from the clearest-face frame). Exits carry no face, so an exit is linked to whoever is
inside by comparing its body embedding to each occupant's:

* one candidate inside -> elimination: it is them;
* several inside -> the closest occupant wins, but only if it clears a similarity
  threshold AND beats the runner-up by a margin. Otherwise the exit is left
  unattributed rather than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .domain import Direction, Event, EventSink
from .recognition.embeddings import cosine


@dataclass(slots=True)
class Occupant:
    name: str
    entered_at: float
    body_embedding: np.ndarray | None  # None when no body crop was embeddable at entry


class Ledger:
    """In-memory occupancy, journalled to an :class:`EventSink` as it changes."""

    def __init__(self, sink: EventSink, exit_similarity: float, exit_margin: float) -> None:
        self._sink = sink
        self._similarity = exit_similarity
        self._margin = exit_margin
        self._inside: dict[str, Occupant] = {}

    @property
    def occupancy(self) -> list[str]:
        return sorted(self._inside)

    def enter(self, name: str, body_embedding: np.ndarray | None, timestamp: float) -> None:
        self._inside[name] = Occupant(name, timestamp, body_embedding)
        self._sink.record(Event(timestamp, name, Direction.IN))

    def exit(self, body_embedding: np.ndarray | None, timestamp: float) -> str | None:
        """Attribute an exit to an occupant and remove them. ``None`` if unresolved."""
        name = self._attribute(body_embedding)
        if name is None:
            return None
        del self._inside[name]
        self._sink.record(Event(timestamp, name, Direction.OUT))
        return name

    def _attribute(self, query: np.ndarray | None) -> str | None:
        if not self._inside:
            return None
        if len(self._inside) == 1:
            return next(iter(self._inside))  # elimination: only one candidate
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
