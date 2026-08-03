"""One person's faces: which recognition uses, why, and overruling it."""

from __future__ import annotations

import numpy as np
from starlette.testclient import TestClient

from stuhi_vision.domain import Direction, Outcome, Sighting
from stuhi_vision.recognition.gallery import FaceGallery
from stuhi_vision.review import ReviewQueue
from stuhi_vision.web.app import create_app


def _frame_for(client, sighting_id: str) -> dict:
    frames = client.get("/api/person/art/faces").json()["frames"]
    return next(frame for frame in frames if frame["id"] == sighting_id)


def _alike(name: str, count: int, drift: float = 0.004):
    """Faces of one person, near-identical but distinct.

    Distinct matters: the gallery counts one sighting once by comparing *vectors*, so
    bit-identical faces would collapse into a single reference and no real camera produces
    those anyway.
    """
    return [(name, [1.0, drift * index, 0.0]) for index in range(count)]


def _setup(tmp_path, faces: list[tuple[str, list[float]]]):
    gallery = FaceGallery()
    review = ReviewQueue(tmp_path / "review", gallery, tmp_path / "gallery")
    ids = []
    for index, (name, vector) in enumerate(faces):
        sighting_id = review.record(
            Sighting(
                timestamp=1_785_000_000.0 + index,
                direction=Direction.IN,
                name=None,
                score=0.5,
                outcome=Outcome.UNKNOWN,
                face_embedding=np.array(vector, dtype=float),
                face_crop=None,
            )
        )
        review.label(sighting_id, name)
        ids.append(sighting_id)
    return TestClient(create_app(review)), review, gallery, ids


def test_the_faces_say_whether_they_are_used_and_why(tmp_path) -> None:
    alike = _alike("art", 8)
    odd = [("art", [0.0, 1.0, 0.0]), ("art", [0.0, 0.0, 1.0])]
    client, _, _, ids = _setup(tmp_path, alike + odd)

    body = client.get("/api/person/art/faces").json()

    assert body["name"] == "art"
    assert body["kept"] == 10
    assert body["in_use"] == 8
    excluded = [frame for frame in body["frames"] if not frame["in_use"]]
    assert {frame["id"] for frame in excluded} == set(ids[8:])
    assert all(frame["why"] == "unlike the average" for frame in excluded)


def test_recognition_only_matches_against_the_chosen_faces(tmp_path) -> None:
    """The point of the whole thing: a face left out must not decide who somebody is."""
    alike = _alike("art", 8)
    stranger = [("art", [0.0, 1.0, 0.0]), ("art", [0.0, 0.0, 1.0])]
    _, _, gallery, _ = _setup(tmp_path, alike + stranger)

    assert gallery.counts() == {"art": 10}  # everything is still kept
    assert gallery.matching_counts == {"art": 8}  # not everything is used

    # A face exactly like the two that were dropped scores nothing now.
    ranked = gallery.rank(np.array([0.0, 1.0, 0.0]))
    assert ranked[0].score < 0.5


def test_asking_for_a_face_overrules_the_rule(tmp_path) -> None:
    alike = _alike("art", 8)
    beard = [("art", [0.0, 1.0, 0.0])]
    client, _, gallery, ids = _setup(tmp_path, alike + beard)
    # Nine faces, four fifths of which is seven: the beard and the least-alike of the rest.
    assert gallery.matching_counts == {"art": 7}

    response = client.post("/api/use-face", json={"sighting_id": ids[-1], "wanted": True})

    assert response.status_code == 200
    assert gallery.matching_counts == {"art": 8}
    frame = _frame_for(client, ids[-1])
    assert frame["in_use"] is True
    assert frame["why"] == "you asked for it"


def test_excluding_a_face_keeps_it_but_stops_it_being_used(tmp_path) -> None:
    client, _, gallery, _ = _setup(tmp_path, _alike("art", 4))
    assert gallery.matching_counts == {"art": 3}  # four fifths of four
    # One the rule *is* using, or excluding it would prove nothing: the rule had already
    # dropped the oldest of four on its own.
    in_use = next(
        f["id"] for f in client.get("/api/person/art/faces").json()["frames"] if f["in_use"]
    )

    client.post("/api/use-face", json={"sighting_id": in_use, "wanted": False})

    assert gallery.counts() == {"art": 4}  # the record is untouched
    assert gallery.matching_counts == {"art": 2}
    frame = _frame_for(client, in_use)
    assert frame["in_use"] is False
    assert frame["why"] == "you excluded it"


def test_a_decision_can_be_handed_back_to_the_rule(tmp_path) -> None:
    client, _, gallery, _ = _setup(tmp_path, _alike("art", 4))
    in_use = next(
        f["id"] for f in client.get("/api/person/art/faces").json()["frames"] if f["in_use"]
    )
    client.post("/api/use-face", json={"sighting_id": in_use, "wanted": False})
    assert gallery.matching_counts == {"art": 2}

    client.post("/api/use-face", json={"sighting_id": in_use, "wanted": None})

    assert gallery.matching_counts == {"art": 3}  # the rule decides again
    frame = _frame_for(client, in_use)
    assert frame["decided"] is None


def test_the_two_orders_are_both_offered(tmp_path) -> None:
    alike = _alike("art", 8)
    odd = [("art", [0.0, 1.0, 0.0])]
    client, _, _, ids = _setup(tmp_path, alike + odd)

    latest = client.get("/api/person/art/faces?sort=latest").json()["frames"]
    used = client.get("/api/person/art/faces?sort=used").json()["frames"]

    assert latest[0]["id"] == ids[-1]  # newest, which is the odd one here
    assert used[0]["in_use"] is True  # in use first
    assert used[-1]["id"] == ids[-1]  # the unused one sinks to the bottom


def test_a_name_nobody_has_returns_an_empty_gallery(tmp_path) -> None:
    client, _, _, _ = _setup(tmp_path, _alike("art", 1))

    body = client.get("/api/person/nobody/faces").json()

    assert body == {
        "name": "nobody",
        "frames": [],
        "in_use": 0,
        "kept": 0,
        "confirmed": 0,
        "guessed": 0,
    }


def test_frames_the_system_linked_itself_are_listed_as_unconfirmed(tmp_path) -> None:
    """An identity the system invented carries its name in ``name``, not ``labelled_as``.

    Listing only the labelled ones left fifty of seventy-one people on a live board with an empty
    gallery and no portrait while their frames sat on disk.
    """
    client, review, _, _ = _setup(tmp_path, [])
    # What the pipeline files for somebody it could not recognise: a name it invented for them.
    guessed = review.record(
        Sighting(
            timestamp=1_785_000_000.0,
            direction=Direction.IN,
            name="guest-0803-070244",
            score=0.2,
            outcome=Outcome.UNKNOWN,
            face_embedding=np.array([1.0, 0.0, 0.0]),
            face_crop=None,
        )
    )

    body = client.get("/api/person/guest-0803-070244/faces").json()

    assert [frame["id"] for frame in body["frames"]] == [guessed]
    assert body["kept"] == 1
    assert body["confirmed"] == 0
    assert body["guessed"] == 1
    assert body["frames"][0]["link"] == "guessed"
    assert body["frames"][0]["confirmed"] is False
    # It teaches the named gallery nothing until somebody says whose face it is.
    assert body["frames"][0]["in_use"] is False
    assert "guess" in body["frames"][0]["why"]


def test_naming_a_guessed_frame_makes_it_confirmed(tmp_path) -> None:
    client, review, _, _ = _setup(tmp_path, [])
    guessed = review.record(
        Sighting(
            timestamp=1_785_000_000.0,
            direction=Direction.IN,
            name="guest-0803-070244",
            score=0.2,
            outcome=Outcome.UNKNOWN,
            face_embedding=np.array([1.0, 0.0, 0.0]),
            face_crop=None,
        )
    )

    client.post("/api/label", json={"sighting_id": guessed, "name": "ilari"})

    assert client.get("/api/person/guest-0803-070244/faces").json()["kept"] == 0
    named = client.get("/api/person/ilari/faces").json()
    assert named["confirmed"] == 1 and named["guessed"] == 0
    assert named["frames"][0]["link"] == "labelled"


def test_a_frame_set_aside_as_nobody_does_not_come_back_under_the_guessed_name(tmp_path) -> None:
    """Somebody looked at it and said it belongs to no one. Re-listing it argues with them."""
    client, review, _, _ = _setup(tmp_path, [])
    aside = review.record(
        Sighting(
            timestamp=1_785_000_000.0,
            direction=Direction.IN,
            name="guest-0803-070244",
            score=0.2,
            outcome=Outcome.UNKNOWN,
            face_embedding=np.array([1.0, 0.0, 0.0]),
            face_crop=None,
        )
    )
    review.label(aside, "unknown")

    assert client.get("/api/person/guest-0803-070244/faces").json()["kept"] == 0
