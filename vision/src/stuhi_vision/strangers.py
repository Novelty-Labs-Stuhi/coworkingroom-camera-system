"""Faces of people nobody has named yet, kept where they cannot be mistaken for evidence.

Three rules have to hold at once, and they pull against each other:

1. an uncertain face must never join the references a *named* person is recognised by --
   that is the one error in this system that compounds, because a stranger's face under a
   real name makes every future match against that name worse;
2. the count must stay right, which means somebody who comes back must be recognised as the
   same person, or their exit cannot be attributed and their entry never closes;
3. somebody genuinely new must get an identity that can be named later, not be forced onto
   the closest person the recogniser happens to know.

Satisfying (2) needs the stranger's own face to be matchable. Satisfying (1) forbids putting
it where the named people live. So there are **two stores**, not one store with a flag: the
named gallery, and this. A flag is something every read path has to remember to honour, and
forgetting once silently poisons the thing rule (1) exists to protect. Two stores cannot be
got wrong by omission -- an uncertain face is not *discouraged* from the named gallery, it is
structurally unable to reach it.

The bar for "is this the same stranger as before" is deliberately **lower** than the bar for
"is this Ilari". The two mistakes are not comparable. Naming the wrong real person puts two
people wrong at once -- one credited with hours they did not work, one missing from the record.
Merging two strangers costs nobody their hours, shows up as one card to name instead of two,
and the merge log makes it reversible. Being cautious here buys nothing and costs the identity
explosion that made the room fill with people who could never be recognised.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .recognition.gallery import FaceGallery

_log = logging.getLogger(__name__)

# How alike two sightings must be to be treated as the same unnamed person. Lower than
# ``face_match``, because the cost of being wrong is one card to name rather than two people's
# figures. The margin still applies: a face equally like two strangers joins neither.
SIMILARITY = 0.30
MARGIN = 0.04


class Strangers:
    """The provisional face set: identities the system invented, and how to find them again."""

    def __init__(
        self,
        directory: Path,
        similarity: float = SIMILARITY,
        margin: float = MARGIN,
    ) -> None:
        self._directory = Path(directory)
        self._similarity = similarity
        self._margin = margin
        self._faces = FaceGallery.load(self._directory)

    @property
    def names(self) -> list[str]:
        return self._faces.names

    def references_for(self, name: str) -> list:
        return self._faces.references_for(name)

    def recognise(self, embedding) -> str | None:
        """Which unnamed person this is, if it is clearly one we have already met.

        ``None`` means nobody recognisable -- which is the answer that earns a new identity,
        rather than being forced onto whoever happened to rank first.
        """
        if embedding is None:
            return None
        ranked = self._faces.rank(embedding)
        if not ranked:
            return None
        best = ranked[0]
        runner_up = ranked[1].score if len(ranked) > 1 else 0.0
        if best.score < self._similarity or best.score - runner_up < self._margin:
            return None
        return best.name

    def remember(self, name: str, embedding) -> None:
        """Keep this face as the one that identifies a newly invented identity."""
        if embedding is None:
            return
        self._faces.add(name, embedding)
        self._faces.save(self._directory)

    def forget(self, name: str) -> None:
        """Drop an identity's faces -- it has been named, or it has expired."""
        removed = self._faces.drop(name)
        if removed:
            # Saving sweeps the file for a name that no longer has any vectors, so the face
            # cannot come back on the next reload and put the card back on the page.
            self._faces.save(self._directory)
            _log.info("dropped %d provisional face(s) for %s", removed, name)
