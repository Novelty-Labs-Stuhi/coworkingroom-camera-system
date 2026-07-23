"""The enrolled-faces gallery: names -> reference face embeddings.

Persisted as one ``<name>.npy`` per person (a stack of that person's reference
vectors) under the gallery directory. Matching is a nearest-neighbour search, so a
face is only accepted as a known person when it clears the similarity threshold.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .embeddings import Match, nearest


class FaceGallery:
    """A mutable, disk-backed collection of labelled face embeddings."""

    def __init__(self, references: dict[str, list[np.ndarray]] | None = None) -> None:
        self._references: dict[str, list[np.ndarray]] = references or {}

    @classmethod
    def load(cls, directory: Path) -> FaceGallery:
        references: dict[str, list[np.ndarray]] = {}
        if directory.exists():
            for path in sorted(directory.glob("*.npy")):
                stacked = np.load(path)
                references[path.stem] = [row for row in stacked]
        return cls(references)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for name, vectors in self._references.items():
            np.save(directory / f"{name}.npy", np.stack(vectors))

    def add(self, name: str, embedding: np.ndarray) -> None:
        self._references.setdefault(name, []).append(embedding)

    def match(self, embedding: np.ndarray, threshold: float) -> Match | None:
        return nearest(embedding, self._references, threshold)

    @property
    def names(self) -> list[str]:
        return sorted(self._references)
