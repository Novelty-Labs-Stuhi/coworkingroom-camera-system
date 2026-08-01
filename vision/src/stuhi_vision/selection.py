"""Which of a person's faces recognition actually uses.

Every labelled sighting keeps its face vector for ever -- that is the record. But a gallery
that matches against *all* of them gets worse as it grows: an old haircut, a bad frame kept
because nobody looked at it, and a mislabel all pull matches towards themselves, and the more
references somebody has the more of that noise they carry.

So the ones used for matching are chosen, in two steps:

* **The newest few.** People change -- hair, beards, glasses, a year of weather -- and a face
  from months ago is evidence about somebody who no longer looks like that. Anything older
  than the window is not used, and does not count towards the average either.
* **The closest to their average, within that window.** A face far from the rest of somebody's
  is either a bad capture or the wrong person, and either way it should not be matched against.
  The ones dropped here *do* still shape the average: they are that person, however awkward the
  frame, and excluding them from the average would let a tight clique of similar frames define
  the person and push every honest variation out.

A person can overrule both, and their choice wins. Automatic selection is a default, not a
verdict: somebody who has looked at a frame knows things the numbers do not -- that it is the
only picture of a colleague with their new beard, or that it is the back of a head.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .enrolment import Reference
from .recognition.embeddings import cosine, normalize

# How many of a person's most recent faces are considered at all. Fifty is generous: a person
# passing a door a few times a day reaches it in a fortnight, so the window is recent history
# rather than a hard cap on how much is remembered.
NEWEST = 50
# What fraction of those are used for matching, closest to the average first. The rest are
# still that person -- they shape the average -- but are not matched against.
KEEP = 0.80


@dataclass(frozen=True, slots=True)
class Chosen:
    """What one person's faces are being used for, and why."""

    name: str
    used: tuple[str, ...]        # sighting ids matched against
    too_old: tuple[str, ...]     # outside the window: no say in anything
    too_odd: tuple[str, ...]     # in the window, shaped the average, not matched against
    pinned: tuple[str, ...]      # used because a person said so
    barred: tuple[str, ...]      # not used because a person said so

    @property
    def readable(self) -> str:
        return (
            f"{self.name}: matching against {len(self.used)}"
            f" ({len(self.too_odd)} too unlike, {len(self.too_old)} too old"
            f", {len(self.pinned)} pinned, {len(self.barred)} barred)"
        )


def choose(
    references: list[Reference],
    ages: dict[str, float],
    decided: dict[str, bool] | None = None,
    newest: int = NEWEST,
    keep: float = KEEP,
) -> dict[str, Chosen]:
    """Pick each person's matching set from everything enrolled under them.

    ``ages`` is when each sighting happened, and ``decided`` holds the choices a person has
    made: True to use a face whatever the numbers say, False to never use it.
    """
    decided = decided or {}
    by_name: dict[str, list[Reference]] = {}
    for reference in references:
        by_name.setdefault(reference.name, []).append(reference)
    return {
        name: _for_one(name, found, ages, decided, newest, keep)
        for name, found in by_name.items()
    }


def _for_one(
    name: str,
    found: list[Reference],
    ages: dict[str, float],
    decided: dict[str, bool],
    newest: int,
    keep: float,
) -> Chosen:
    found.sort(key=lambda reference: ages.get(reference.sighting_id, 0.0), reverse=True)
    window, older = found[:newest], found[newest:]

    # The average comes from the whole window, including the faces about to be dropped for
    # being unlike it. Otherwise the tightest group of similar frames defines the person and
    # every honest variation looks like an outlier.
    average = _average([reference.embedding for reference in window])
    ranked = sorted(
        window,
        key=lambda reference: cosine(reference.embedding, average) if average is not None else 0.0,
        reverse=True,
    )
    wanted = _cut(ranked, average, keep)
    close, odd = ranked[:wanted], ranked[wanted:]

    pinned = tuple(
        reference.sighting_id
        for reference in found
        if decided.get(reference.sighting_id) is True
    )
    barred = tuple(
        reference.sighting_id
        for reference in found
        if decided.get(reference.sighting_id) is False
    )
    used = [reference.sighting_id for reference in close if reference.sighting_id not in barred]
    used += [sighting for sighting in pinned if sighting not in used]
    return Chosen(
        name=name,
        used=tuple(used),
        too_old=tuple(r.sighting_id for r in older if r.sighting_id not in pinned),
        too_odd=tuple(r.sighting_id for r in odd if r.sighting_id not in pinned),
        pinned=pinned,
        barred=barred,
    )


def _cut(ranked: list[Reference], average: np.ndarray | None, keep: float) -> int:
    """How many of the ranked faces to keep: the fraction, never splitting equals.

    The fraction alone would throw away a face that is exactly as good as one it kept -- four
    identical captures and a keep of four fifths discards one of them at random, losing a
    perfectly good reference to arithmetic. So the cut is extended over anything as close to
    the average as the last face kept.
    """
    if not ranked or average is None:
        return len(ranked)
    wanted = max(1, round(len(ranked) * keep))
    scores = [float(cosine(reference.embedding, average)) for reference in ranked]
    while wanted < len(ranked) and scores[wanted] >= scores[wanted - 1] - 1e-9:
        wanted += 1
    return wanted


def _average(embeddings: list[np.ndarray]) -> np.ndarray | None:
    """The middle of a set of faces, as a unit vector so cosines stay comparable."""
    if not embeddings:
        return None
    return normalize(np.mean(np.stack(embeddings), axis=0))


def distances(references: list[Reference], ages: dict[str, float], newest: int = NEWEST):
    """How like their own average each face is, for showing a person why one was dropped.

    Measured against the same average the choosing uses -- the window's -- so a number shown
    beside a face explains the decision that was actually made about it.
    """
    by_name: dict[str, list[Reference]] = {}
    for reference in references:
        by_name.setdefault(reference.name, []).append(reference)

    scores: dict[str, float] = {}
    for found in by_name.values():
        found.sort(key=lambda reference: ages.get(reference.sighting_id, 0.0), reverse=True)
        average = _average([reference.embedding for reference in found[:newest]])
        if average is None:
            continue
        for reference in found:
            scores[reference.sighting_id] = float(cosine(reference.embedding, average))
    return scores
