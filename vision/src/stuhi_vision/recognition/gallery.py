"""The enrolled-faces gallery: names -> reference face embeddings.

Persisted as one ``<name>.npy`` per person (a stack of that person's reference
vectors) under the gallery directory. Matching is a nearest-neighbour search, so a
face is only accepted as a known person when it clears the similarity threshold.

The gallery grows at runtime: a face nobody recognised is enrolled by labelling it (see
``review.ReviewQueue``), so the next person through the door is matched against it.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from .embeddings import Match, nearest, rank


class FaceGallery:
    """A mutable, disk-backed collection of labelled face embeddings.

    Labels arrive from outside the pipeline -- the Telegram poll thread and the web UI --
    while the pipeline is reading the gallery to recognise faces. Every access therefore
    takes a lock: mutating a name's list while ``rank`` iterates it would raise, or worse,
    silently compare against a half-updated set.
    """

    def __init__(self, references: dict[str, list[np.ndarray]] | None = None) -> None:
        self._references: dict[str, list[np.ndarray]] = references or {}
        # What matching actually compares against, when a choice has been made about it. Every
        # enrolled face is still kept -- that is the record, and a face dropped today may be
        # wanted tomorrow -- but a gallery that matches against all of them gets worse as it
        # grows: an old haircut and a mislabel pull matches towards themselves for ever.
        self._matching: dict[str, list[np.ndarray]] | None = None
        self._lock = threading.RLock()

    @classmethod
    def load(cls, directory: Path) -> FaceGallery:
        references: dict[str, list[np.ndarray]] = {}
        if directory.exists():
            for path in sorted(directory.glob("*.npy")):
                stacked = np.load(path)
                references[path.stem] = [row for row in stacked]
        return cls(references)

    def save(self, directory: Path) -> None:
        """Write every name's vectors, and remove files for names that no longer exist.

        Stale files go **first**, before anything is written. Writing first loses data on a
        case-insensitive filesystem: correcting "yehor" to "Yehor" wrote Yehor.npy into what is
        the same file, and the sweep then saw a directory entry still spelled yehor.npy, decided
        that name was gone, and deleted the person entirely. Removing first cannot do that,
        because by the time anything is written nothing stale is left to sweep.
        """
        with self._lock:
            directory.mkdir(parents=True, exist_ok=True)
            # A corrected label can empty a name out entirely; drop its file so a reload does
            # not resurrect someone who no longer has any reference vectors.
            for path in directory.glob("*.npy"):
                if path.stem not in self._references:
                    path.unlink()
            for name, vectors in self._references.items():
                np.save(directory / f"{name}.npy", np.stack(vectors))

    def add(self, name: str, embedding: np.ndarray) -> None:
        with self._lock:
            self._references.setdefault(name, []).append(embedding)

    def rename(self, old: str, new: str) -> int:
        """Move every reference from one name to another, merging if the new name exists.

        For a misspelling this is the only correct repair: the face vectors are right, only the
        string is wrong. Re-labelling each sighting by hand would recompute nothing and risk
        losing references. Returns how many moved; 0 if the old name is not there.
        """
        with self._lock:
            vectors = self._references.pop(old, None)
            if not vectors:
                return 0
            self._references.setdefault(new, []).extend(vectors)
            return len(vectors)

    def discard(self, name: str, embedding: np.ndarray) -> bool:
        """Remove one reference vector from ``name``; used when a label is corrected."""
        with self._lock:
            vectors = self._references.get(name)
            if not vectors:
                return False
            for index, vector in enumerate(vectors):
                if np.array_equal(vector, embedding):
                    del vectors[index]
                    if not vectors:
                        del self._references[name]
                    return True
            return False

    def contains(self, name: str, embedding: np.ndarray) -> bool:
        """True when this exact reference vector is already enrolled under ``name``.

        Guards against the same face being counted twice when it arrives by two different
        routes -- the chat and the web UI both labelling the same sighting.
        """
        with self._lock:
            return any(
                np.array_equal(vector, embedding) for vector in self._references.get(name, ())
            )

    def match(self, embedding: np.ndarray, threshold: float) -> Match | None:
        """The best name for this face, over the matching set rather than everything kept."""
        with self._lock:
            source = self._matching if self._matching is not None else self._references
            return nearest(embedding, source, threshold)

    def use_only(self, matching: dict[str, list[np.ndarray]] | None) -> None:
        """Match against these vectors from now on; ``None`` means every enrolled face.

        Chosen elsewhere (:mod:`~stuhi_vision.selection`), because choosing needs to know when
        each face was seen and what a person has decided about it -- neither of which belongs
        in a store of vectors.
        """
        with self._lock:
            self._matching = matching

    @property
    def matching_counts(self) -> dict[str, int]:
        """How many of each person's faces are actually matched against."""
        with self._lock:
            source = self._matching if self._matching is not None else self._references
            return {name: len(vectors) for name, vectors in source.items() if vectors}

    def rank(self, embedding: np.ndarray) -> list[Match]:
        """Every name scored against ``embedding``, best first, over the matching set."""
        with self._lock:
            source = self._matching if self._matching is not None else self._references
            return rank(embedding, source)

    def counts(self) -> dict[str, int]:
        """How many reference vectors are enrolled per name."""
        with self._lock:
            return {name: len(vectors) for name, vectors in sorted(self._references.items())}

    def references_for(self, name: str) -> list:
        """Every reference held under one name. A copy, so callers cannot mutate the set."""
        with self._lock:
            return list(self._references.get(name, []))

    @property
    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._references)
