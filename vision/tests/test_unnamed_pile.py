"""The "needs you" pile offers one card per unnamed person, not one per sighting of them."""

from __future__ import annotations

import numpy as np
import pytest

from stuhi_vision.domain import Direction, Outcome, Sighting
from stuhi_vision.provisional import Provisional
from stuhi_vision.recognition.gallery import FaceGallery
from stuhi_vision.review import ReviewQueue


@pytest.fixture
def queue(tmp_path):
    gallery = FaceGallery()
    provisional = Provisional(tmp_path / "provisional")
    review = ReviewQueue(
        tmp_path / "review", gallery, tmp_path / "gallery", provisional=provisional
    )
    return review, gallery, provisional


def _seen(review, name: str, at: float, score: float) -> str:
    """One crossing of ``name``, with a face so it can be labelled."""
    return review.record(
        Sighting(
            timestamp=at,
            direction=Direction.IN,
            name=name,
            score=score,
            outcome=Outcome.NAMED,
            face_embedding=np.array([1.0, 0.0, 0.0]),
        )
    )


def _enrol(gallery: FaceGallery, provisional: Provisional, name: str) -> None:
    """As the pipeline does it: the face into the gallery, the invented name recorded."""
    gallery.add(name, np.array([1.0, 0.0, 0.0]))
    provisional.add(name)


def test_a_stranger_seen_ten_times_offers_one_card(queue) -> None:
    """The answer to any one of them names the person, so nine are the same question."""
    review, gallery, provisional = queue
    _enrol(gallery, provisional, "guest-0803-090000")
    for visit in range(10):
        _seen(review, "guest-0803-090000", 1000.0 + visit, score=0.3)

    assert len(review.unnamed()) == 1


def test_the_clearest_face_is_the_one_offered(queue) -> None:
    """A name given from a poor frame is the mistake this pile exists to prevent."""
    review, gallery, provisional = queue
    _enrol(gallery, provisional, "guest-a")
    _seen(review, "guest-a", 1000.0, score=0.11)
    best = _seen(review, "guest-a", 1001.0, score=0.62)
    _seen(review, "guest-a", 1002.0, score=0.40)

    assert [r.sighting_id for r in review.unnamed()] == [best]


def test_a_person_a_human_named_is_not_offered(queue) -> None:
    review, gallery, provisional = queue
    _enrol(gallery, provisional, "guest-a")
    gallery.add("arsenii", np.array([0.0, 1.0, 0.0]))
    _seen(review, "guest-a", 1000.0, score=0.3)
    _seen(review, "arsenii", 1001.0, score=0.3)

    assert [r.name for r in review.unnamed()] == ["guest-a"]


def test_labelling_one_sighting_clears_the_person_from_the_pile(queue) -> None:
    """Otherwise the stranger's other nine sightings put the same card straight back."""
    review, gallery, provisional = queue
    _enrol(gallery, provisional, "guest-a")
    first = _seen(review, "guest-a", 1000.0, score=0.5)
    _seen(review, "guest-a", 1001.0, score=0.3)
    _seen(review, "guest-a", 1002.0, score=0.2)

    review.label(first, "arsenii")

    assert review.unnamed() == []


def test_longest_unnamed_first(queue) -> None:
    """A week of unnamed crossings is a worse gap in the figures than a minute of them."""
    review, gallery, provisional = queue
    for name, at in (("guest-new", 5000.0), ("guest-old", 1000.0), ("guest-mid", 3000.0)):
        _enrol(gallery, provisional, name)
        _seen(review, name, at, score=0.3)

    assert [r.name for r in review.unnamed()] == ["guest-old", "guest-mid", "guest-new"]


def test_a_rejected_clip_is_never_offered(queue) -> None:
    review, gallery, provisional = queue
    _enrol(gallery, provisional, "guest-a")
    poor = _seen(review, "guest-a", 1000.0, score=0.9)
    usable = _seen(review, "guest-a", 1001.0, score=0.2)

    review.dismiss(poor, "back of a head")

    assert [r.sighting_id for r in review.unnamed()] == [usable]


def test_an_identity_merged_away_stops_being_asked_about(queue) -> None:
    """Work that can never be completed must not sit on the page for ever."""
    review, gallery, provisional = queue
    _enrol(gallery, provisional, "guest-a")
    _seen(review, "guest-a", 1000.0, score=0.3)
    gallery.discard("guest-a", np.array([1.0, 0.0, 0.0]))

    assert review.unnamed() == []


def test_the_pile_appears_among_the_others(queue) -> None:
    review, gallery, provisional = queue
    _enrol(gallery, provisional, "guest-a")
    _seen(review, "guest-a", 1000.0, score=0.3)

    piles = review.piles()

    assert [r.name for r in piles["unnamed"]] == ["guest-a"]


def test_without_a_provisional_set_the_pile_is_simply_empty(tmp_path) -> None:
    """The queue still works with no notion of invented names -- it just offers nothing."""
    gallery = FaceGallery()
    review = ReviewQueue(tmp_path / "review", gallery, tmp_path / "gallery")
    _seen(review, "guest-a", 1000.0, score=0.3)

    assert review.unnamed() == []
