"""Tests for the vector-matching core (pure numpy, no models)."""

from __future__ import annotations

import numpy as np

from stuhi_vision.recognition.embeddings import cosine, nearest


def test_cosine_identical_is_one() -> None:
    v = np.array([1.0, 2.0, 3.0])
    assert cosine(v, v) == np.float32(1.0) or abs(cosine(v, v) - 1.0) < 1e-6


def test_cosine_orthogonal_is_zero() -> None:
    assert abs(cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0]))) < 1e-6


def test_nearest_picks_best_above_threshold() -> None:
    gallery = {
        "alice": [np.array([1.0, 0.0, 0.0])],
        "bob": [np.array([0.0, 1.0, 0.0])],
    }
    query = np.array([0.9, 0.1, 0.0])
    match = nearest(query, gallery, threshold=0.5)
    assert match is not None
    assert match.name == "alice"


def test_nearest_returns_none_when_nobody_is_close() -> None:
    gallery = {"alice": [np.array([1.0, 0.0, 0.0])]}
    query = np.array([0.0, 0.0, 1.0])  # orthogonal -> similarity 0
    assert nearest(query, gallery, threshold=0.5) is None


def test_nearest_uses_best_of_several_references() -> None:
    gallery = {"alice": [np.array([0.0, 1.0]), np.array([1.0, 0.0])]}
    query = np.array([0.95, 0.05])
    match = nearest(query, gallery, threshold=0.5)
    assert match is not None and match.name == "alice"
