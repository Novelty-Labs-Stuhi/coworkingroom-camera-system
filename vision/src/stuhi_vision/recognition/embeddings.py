"""Vector maths for identity matching -- pure numpy, no models.

An "embedding" is just an L2-normalised vector. Comparing two people is then a
cosine similarity (a dot product), and recognising someone is a nearest-neighbour
search against a gallery of known vectors. That is the whole trick.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

_EPS = 1e-8


def normalize(vector: np.ndarray) -> np.ndarray:
    """Scale a vector to unit length so dot products become cosine similarities."""
    return vector / (float(np.linalg.norm(vector)) + _EPS)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors, in [-1, 1]. Higher means more alike."""
    return float(np.dot(normalize(a), normalize(b)))


@dataclass(frozen=True, slots=True)
class Match:
    name: str
    score: float


def rank(query: np.ndarray, gallery: Mapping[str, Sequence[np.ndarray]]) -> list[Match]:
    """Every gallery name scored against ``query``, best first.

    Unlike :func:`nearest` no threshold is applied: the caller decides what counts as a
    match. Returning the runner-up too is what lets a caller demand a *margin*, which is
    how a stranger (close to nobody in particular) is told apart from a known person
    (distinctly closest to themselves).
    """
    matches = [
        Match(name=name, score=max((cosine(query, ref) for ref in references), default=-1.0))
        for name, references in gallery.items()
    ]
    return sorted(matches, key=lambda match: match.score, reverse=True)


def nearest(
    query: np.ndarray,
    gallery: Mapping[str, Sequence[np.ndarray]],
    threshold: float,
) -> Match | None:
    """Return the closest name in ``gallery`` whose similarity clears ``threshold``.

    Each name may have several reference vectors (multiple views / visits); we score
    a name by its *best* matching vector. Returns ``None`` when nobody is close
    enough -- important so an unknown person is not force-matched to whoever is
    merely least-far.
    """
    best: Match | None = None
    for name, references in gallery.items():
        score = max((cosine(query, ref) for ref in references), default=-1.0)
        if score >= threshold and (best is None or score > best.score):
            best = Match(name=name, score=score)
    return best
