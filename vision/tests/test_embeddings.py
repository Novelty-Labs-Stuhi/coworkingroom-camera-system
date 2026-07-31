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


def test_the_labelling_still_is_the_face_with_room_around_it() -> None:
    """A body-sized picture with a small face in it is not something you can put a name to."""
    import numpy as np

    from stuhi_vision.domain import Box
    from stuhi_vision.sessions import _portrait

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[100:140, 300:340] = 255            # a 40x40 "face"
    face = Box(300, 100, 340, 140)

    portrait = _portrait(image, face)

    # Padded well beyond the face's own edges: hair, ears and jaw are much of what a person is
    # recognised by, and a tight crop of a mis-detected box would cut the face in half.
    assert portrait.shape[0] > 40 and portrait.shape[1] > 40
    assert portrait.shape[0] < image.shape[0]   # still a zoom, not the whole frame
    assert portrait.max() == 255                # the face itself is in there


def test_a_face_at_the_frame_edge_still_produces_a_still() -> None:
    """Padding runs off the picture for somebody at the edge; that must not empty the crop."""
    import numpy as np

    from stuhi_vision.domain import Box
    from stuhi_vision.sessions import _portrait

    image = np.full((480, 640, 3), 120, dtype=np.uint8)
    portrait = _portrait(image, Box(0, 0, 30, 30))

    assert portrait.size > 0
