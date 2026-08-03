"""What recognition is allowed to match against, and what silently fell out of it."""

from __future__ import annotations

import numpy as np

from stuhi_vision.domain import Direction, Outcome, Sighting
from stuhi_vision.recognition.gallery import FaceGallery
from stuhi_vision.review import ReviewQueue


def _face(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _queue(tmp_path):
    gallery = FaceGallery()
    return ReviewQueue(tmp_path / "review", gallery, tmp_path / "gallery"), gallery


def _file(review, name, at, seed):
    return review.record(
        Sighting(
            timestamp=at,
            direction=Direction.IN,
            name=name,
            score=0.5,
            outcome=Outcome.NAMED,
            face_embedding=_face(*seed),
        )
    )


def test_an_identity_the_system_enrolled_is_matched_against(tmp_path) -> None:
    """Its whole purpose is that the same person is matched to it next time."""
    review, gallery = _queue(tmp_path)
    gallery.add("guest-0803", _face(1.0, 0.0, 0.0))

    review.refresh_matching()

    assert gallery.matching_counts.get("guest-0803") == 1


def test_a_returning_stranger_is_recognised_instead_of_becoming_a_new_person(tmp_path) -> None:
    review, gallery = _queue(tmp_path)
    gallery.add("guest-0803", _face(1.0, 0.0, 0.0))
    review.refresh_matching()

    best = gallery.rank(_face(0.99, 0.14, 0.0))[0]

    assert best.name == "guest-0803"
    assert best.score > 0.9


def test_a_name_enrolled_from_photos_is_matched_against(tmp_path) -> None:
    """The CLI enrol route leaves no review record either."""
    review, gallery = _queue(tmp_path)
    gallery.add("ilari", _face(1.0, 0.0, 0.0))
    gallery.add("ilari", _face(0.98, 0.2, 0.0))

    review.refresh_matching()

    assert gallery.matching_counts.get("ilari") == 2


def test_a_reviewed_name_still_uses_only_its_chosen_faces(tmp_path) -> None:
    """The selection rules must keep applying where there is something to select from."""
    review, gallery = _queue(tmp_path)
    good = _file(review, None, 100.0, (1.0, 0.0, 0.0))
    review.label(good, "ilari")
    barred = _file(review, None, 200.0, (0.0, 1.0, 0.0))
    review.label(barred, "ilari")
    review.use_face(barred, False)

    counts = gallery.matching_counts

    # The barred face is still enrolled, and still not matched against.
    assert counts["ilari"] == 1
    assert len(gallery.references_for("ilari")) == 2


def test_a_labelled_name_is_not_quietly_replaced_by_its_raw_gallery_faces(tmp_path) -> None:
    review, gallery = _queue(tmp_path)
    first = _file(review, None, 100.0, (1.0, 0.0, 0.0))
    review.label(first, "ilari")

    review.refresh_matching()

    assert gallery.matching_counts["ilari"] == 1
