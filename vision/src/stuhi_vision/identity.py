"""Who a track is: the running best face match over its whole time in view.

The rule is deliberately one number. Every frame a face is seen it is scored against the
gallery, and the track keeps the *highest* score it has ever achieved. There is no
separate clarity gate: for the correct person a clearer, more frontal face simply scores
higher, so the running maximum already prefers the best look.

Three outcomes, decided only when the person crosses the doorway:

``NAMED``
    the best score cleared the threshold *and* beat the runner-up by a margin.
``UNKNOWN``
    a face was seen but nothing in the gallery matched well enough -- a stranger, or
    someone not enrolled yet. The crop is kept so the face can be labelled and enrolled.
``UNIDENTIFIED``
    no usable face was ever seen, so no claim is made either way.

The margin matters as much as the threshold: a known person is distinctly closest to
themselves, whereas a stranger tends to be mediocre against everyone with no clear
winner. Requiring a gap is what separates "this is a stranger" from "this is a bad look
at someone I know".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .domain import Outcome
from .recognition.embeddings import Match

_NO_SCORE = -2.0  # below the -1..1 cosine range, so the first observation always wins


@dataclass(frozen=True, slots=True)
class Decision:
    """The identity verdict for one track."""

    outcome: Outcome
    name: str | None
    score: float
    margin: float


@dataclass(slots=True)
class RunningIdentity:
    """Accumulates the best face match seen across a track's lifetime."""

    threshold: float
    margin_min: float
    best_name: str | None = None
    best_score: float = _NO_SCORE
    best_margin: float = 0.0
    saw_face: bool = False

    def observe(self, ranked: Sequence[Match]) -> bool:
        """Fold in one frame's gallery ranking. True when this became the best so far.

        Call only when a face was actually detected; ``ranked`` may still be empty, which
        is what an un-enrolled gallery looks like. Returning whether it improved lets the
        caller keep the embedding and crop from precisely the winning frame.
        """
        self.saw_face = True
        top = ranked[0] if ranked else None
        score = top.score if top is not None else _NO_SCORE + 1.0
        runner_up = ranked[1].score if len(ranked) > 1 else -1.0
        if score <= self.best_score:
            return False
        self.best_score = score
        self.best_name = top.name if top is not None else None
        self.best_margin = score - runner_up if top is not None else 0.0
        return True

    def decide(self) -> Decision:
        """Resolve the track into a final verdict."""
        if not self.saw_face:
            return Decision(Outcome.UNIDENTIFIED, None, 0.0, 0.0)
        named = (
            self.best_name is not None
            and self.best_score >= self.threshold
            and self.best_margin >= self.margin_min
        )
        outcome = Outcome.NAMED if named else Outcome.UNKNOWN
        return Decision(
            outcome=outcome,
            name=self.best_name if named else None,
            score=max(self.best_score, -1.0),
            margin=self.best_margin,
        )
