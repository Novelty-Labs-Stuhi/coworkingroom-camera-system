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
