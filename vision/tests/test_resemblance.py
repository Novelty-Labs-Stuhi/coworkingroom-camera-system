"""Identities nobody named, offered as one question instead of hundreds of clips."""

from __future__ import annotations

import numpy as np

from stuhi_vision.recognition.gallery import FaceGallery
from stuhi_vision.resemblance import Resemblance, suggestions


def _face(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


ILARI = _face(1.0, 0.0, 0.0)
ARSENII = _face(0.0, 1.0, 0.0)


def _gallery(**people) -> FaceGallery:
    gallery = FaceGallery()
    for name, face in people.items():
        gallery.add(name.replace("_", "-"), face)
    return gallery


def test_a_provisional_identity_that_clearly_matches_is_offered() -> None:
    gallery = _gallery(ilari=ILARI, arsenii=ARSENII, guest_a=_face(0.99, 0.14, 0.0))

    offered = suggestions(gallery, {"guest-a"})

    assert [(p.provisional, p.resembles) for p in offered] == [("guest-a", "ilari")]


def test_a_face_between_two_people_is_never_offered() -> None:
    """Equally like two people names neither -- the margin is what refuses it."""
    gallery = _gallery(ilari=ILARI, arsenii=ARSENII, guest_a=_face(0.71, 0.71, 0.0))

    assert suggestions(gallery, {"guest-a"}) == []


def test_a_weak_match_is_never_offered() -> None:
    gallery = _gallery(ilari=ILARI, guest_a=_face(0.4, 0.0, 0.92))

    assert suggestions(gallery, {"guest-a"}) == []


def test_two_provisional_identities_are_never_offered_to_each_other() -> None:
    """Folding two strangers together names nobody and loses both."""
    gallery = _gallery(guest_a=_face(1.0, 0.0, 0.0), guest_b=_face(0.99, 0.14, 0.0))

    assert suggestions(gallery, {"guest-a", "guest-b"}) == []


def test_a_named_person_is_never_proposed_for_absorption() -> None:
    gallery = _gallery(ilari=ILARI, arsenii=_face(0.99, 0.14, 0.0))

    offered = suggestions(gallery, set())

    assert offered == []


def test_the_offer_says_how_many_crossings_it_would_correct() -> None:
    """The point of accepting: hundreds of unreviewed clips corrected by one click."""
    gallery = _gallery(ilari=ILARI, guest_a=_face(0.99, 0.14, 0.0))

    offered = suggestions(gallery, {"guest-a"}, counts={"guest-a": 42})

    assert offered[0].crossings == 42
    assert "42 crossing(s)" in offered[0].readable


def test_an_identity_with_no_face_is_skipped_not_guessed_at() -> None:
    gallery = _gallery(ilari=ILARI)

    assert suggestions(gallery, {"guest-never-had-a-face"}) == []


def test_nothing_is_offered_when_nobody_has_been_named_yet() -> None:
    gallery = _gallery(guest_a=_face(1.0, 0.0, 0.0))

    assert suggestions(gallery, {"guest-a"}) == []


def test_the_strongest_resemblance_is_offered_first() -> None:
    gallery = _gallery(
        ilari=ILARI,
        arsenii=ARSENII,
        guest_a=_face(0.95, 0.31, 0.0),
        guest_b=_face(0.999, 0.045, 0.0),
    )

    offered = suggestions(gallery, {"guest-a", "guest-b"})

    assert [p.provisional for p in offered] == ["guest-b", "guest-a"]


def test_nothing_is_applied_by_computing_a_suggestion() -> None:
    """It proposes; it never merges. Doing so automatically means doing it at scale."""
    gallery = _gallery(ilari=ILARI, guest_a=_face(0.99, 0.14, 0.0))

    suggestions(gallery, {"guest-a"})

    assert gallery.names == ["guest-a", "ilari"]


def test_a_resemblance_reads_clearly() -> None:
    pair = Resemblance("guest-a", "ilari", 0.81, 0.42, 7)

    assert pair.readable == "guest-a looks like ilari (0.81 vs 0.42 next, 7 crossing(s))"
