"""Identities nobody named, offered as one question instead of hundreds of clips."""

from __future__ import annotations

import numpy as np

from stuhi_vision.recognition.gallery import FaceGallery
from stuhi_vision.resemblance import Resemblance, suggestions
from stuhi_vision.strangers import Strangers


def _face(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


ILARI = _face(1.0, 0.0, 0.0)
ARSENII = _face(0.0, 1.0, 0.0)


def _stores(tmp_path, named=None, guests=None):
    """The two stores as they now are: people with names, and faces of people without."""
    gallery = FaceGallery()
    for name, face in (named or {}).items():
        gallery.add(name, face)
    strangers = Strangers(tmp_path / "strangers")
    for name, face in (guests or {}).items():
        strangers.remember(name, face)
    return gallery, strangers


def test_a_provisional_identity_that_clearly_matches_is_offered(tmp_path) -> None:
    gallery, strangers = _stores(
        tmp_path,
        named={"ilari": ILARI, "arsenii": ARSENII},
        guests={"guest-a": _face(0.99, 0.14, 0.0)},
    )

    offered = suggestions(gallery, strangers, {"guest-a"})

    assert [(p.provisional, p.resembles) for p in offered] == [("guest-a", "ilari")]


def test_a_face_between_two_people_is_never_offered(tmp_path) -> None:
    """Equally like two people names neither -- the margin is what refuses it."""
    gallery, strangers = _stores(
        tmp_path,
        named={"ilari": ILARI, "arsenii": ARSENII},
        guests={"guest-a": _face(0.71, 0.71, 0.0)},
    )

    assert suggestions(gallery, strangers, {"guest-a"}) == []


def test_a_weak_match_is_never_offered(tmp_path) -> None:
    gallery, strangers = _stores(
        tmp_path, named={"ilari": ILARI}, guests={"guest-a": _face(0.4, 0.0, 0.92)}
    )

    assert suggestions(gallery, strangers, {"guest-a"}) == []


def test_two_provisional_identities_are_never_offered_to_each_other(tmp_path) -> None:
    """This pile exists to put a *name* on somebody; two strangers name nobody."""
    gallery, strangers = _stores(
        tmp_path,
        guests={"guest-a": _face(1.0, 0.0, 0.0), "guest-b": _face(0.99, 0.14, 0.0)},
    )

    assert suggestions(gallery, strangers, {"guest-a", "guest-b"}) == []


def test_a_named_person_is_never_proposed_for_absorption(tmp_path) -> None:
    gallery, strangers = _stores(
        tmp_path, named={"ilari": ILARI, "arsenii": _face(0.99, 0.14, 0.0)}
    )

    assert suggestions(gallery, strangers, set()) == []


def test_the_offer_says_how_many_crossings_it_would_correct(tmp_path) -> None:
    """The point of accepting: every unreviewed clip of that person, fixed by one click."""
    gallery, strangers = _stores(
        tmp_path, named={"ilari": ILARI}, guests={"guest-a": _face(0.99, 0.14, 0.0)}
    )

    offered = suggestions(gallery, strangers, {"guest-a"}, counts={"guest-a": 42})

    assert offered[0].crossings == 42
    assert "42 crossing(s)" in offered[0].readable


def test_an_identity_with_no_face_is_skipped_not_guessed_at(tmp_path) -> None:
    gallery, strangers = _stores(tmp_path, named={"ilari": ILARI})

    assert suggestions(gallery, strangers, {"guest-never-had-a-face"}) == []


def test_nothing_is_offered_when_nobody_has_been_named_yet(tmp_path) -> None:
    gallery, strangers = _stores(tmp_path, guests={"guest-a": _face(1.0, 0.0, 0.0)})

    assert suggestions(gallery, strangers, {"guest-a"}) == []


def test_the_strongest_resemblance_is_offered_first(tmp_path) -> None:
    gallery, strangers = _stores(
        tmp_path,
        named={"ilari": ILARI, "arsenii": ARSENII},
        guests={"guest-a": _face(0.95, 0.31, 0.0), "guest-b": _face(0.999, 0.045, 0.0)},
    )

    offered = suggestions(gallery, strangers, {"guest-a", "guest-b"})

    assert [p.provisional for p in offered] == ["guest-b", "guest-a"]


def test_computing_a_suggestion_moves_nothing(tmp_path) -> None:
    """It proposes; it never merges. Doing so automatically means doing it at scale."""
    gallery, strangers = _stores(
        tmp_path, named={"ilari": ILARI}, guests={"guest-a": _face(0.99, 0.14, 0.0)}
    )

    suggestions(gallery, strangers, {"guest-a"})

    assert gallery.names == ["ilari"]
    assert strangers.names == ["guest-a"]


def test_an_unnamed_face_never_competes_with_a_named_person(tmp_path) -> None:
    """The rule that stops the one error which compounds, checked at the boundary itself."""
    gallery, strangers = _stores(
        tmp_path, named={"ilari": ILARI}, guests={"guest-a": _face(0.5, 0.5, 0.7)}
    )

    # The unnamed face is kept, and kept out of the ranking a named person is judged by --
    # so no number of strangers can crowd Ilari out of his own matches.
    assert strangers.names == ["guest-a"]
    assert "guest-a" not in gallery.names
    assert gallery.rank(_face(0.5, 0.5, 0.7))[0].name == "ilari"


def test_a_resemblance_reads_clearly() -> None:
    pair = Resemblance("guest-a", "ilari", 0.81, 0.42, 7)

    assert pair.readable == "guest-a looks like ilari (0.81 vs 0.42 next, 7 crossing(s))"
