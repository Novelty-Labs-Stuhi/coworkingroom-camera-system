"""The labelling UI drives the same queue as the chat, and counts a sighting once."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from stuhi_vision.domain import Direction, Outcome, Sighting
from stuhi_vision.recognition.gallery import FaceGallery
from stuhi_vision.review import ReviewQueue
from stuhi_vision.web import create_app


@pytest.fixture
def setup(tmp_path):
    gallery = FaceGallery()
    review = ReviewQueue(tmp_path / "review", gallery, tmp_path / "gallery")
    sighting_id = review.record(
        Sighting(
            timestamp=1_760_000_000.0,
            direction=Direction.IN,
            name=None,
            score=0.12,
            outcome=Outcome.UNKNOWN,
            face_embedding=np.array([1.0, 0.0, 0.0]),
        )
    )
    return TestClient(create_app(review)), review, gallery, sighting_id


def test_the_page_renders(setup) -> None:
    client, *_ = setup
    response = client.get("/")
    assert response.status_code == 200
    assert "Who came through?" in response.text


def test_pending_sightings_are_listed(setup) -> None:
    client, _review, _gallery, sighting_id = setup
    body = client.get("/api/sightings").json()

    assert [r["id"] for r in body["pending"]] == [sighting_id]
    assert body["people"] == {}


def test_labelling_enrols_and_moves_it_to_labelled(setup) -> None:
    client, _review, gallery, sighting_id = setup

    response = client.post("/api/label", json={"sighting_id": sighting_id, "name": "ilari"})

    assert response.status_code == 200
    assert response.json() == {"outcome": "enrolled", "people": {"ilari": 1}}
    assert gallery.counts() == {"ilari": 1}

    body = client.get("/api/sightings").json()
    assert body["pending"] == []
    assert [r["labelled_as"] for r in body["labelled"]] == ["ilari"]


def test_labelling_twice_counts_once(setup) -> None:
    client, _review, gallery, sighting_id = setup
    payload = {"sighting_id": sighting_id, "name": "ilari"}

    client.post("/api/label", json=payload)
    second = client.post("/api/label", json=payload)

    assert second.json()["outcome"] == "unchanged"
    assert gallery.counts() == {"ilari": 1}


def test_a_label_from_the_chat_is_not_duplicated_by_the_ui(setup) -> None:
    # Both routes share one queue; the same sighting must contribute one reference.
    client, review, gallery, sighting_id = setup
    review.label(sighting_id, "ilari")  # as the Telegram poller would

    response = client.post("/api/label", json={"sighting_id": sighting_id, "name": "ilari"})

    assert response.json()["outcome"] == "unchanged"
    assert gallery.counts() == {"ilari": 1}


def test_correcting_replaces_rather_than_accumulates(setup) -> None:
    client, _review, gallery, sighting_id = setup
    client.post("/api/label", json={"sighting_id": sighting_id, "name": "ilari"})

    response = client.post("/api/label", json={"sighting_id": sighting_id, "name": "mark"})

    assert response.json()["outcome"] == "corrected"
    assert gallery.counts() == {"mark": 1}


def test_a_blank_name_is_refused(setup) -> None:
    client, _review, gallery, sighting_id = setup
    response = client.post("/api/label", json={"sighting_id": sighting_id, "name": "   "})

    assert response.status_code == 400
    assert gallery.counts() == {}


def test_an_unknown_sighting_is_a_404(setup) -> None:
    client, *_ = setup
    response = client.post("/api/label", json={"sighting_id": "nope", "name": "ilari"})
    assert response.status_code == 404


def test_missing_media_is_a_404(setup) -> None:
    client, _review, _gallery, sighting_id = setup
    assert client.get(f"/media/{sighting_id}.mp4").status_code == 404


def test_the_camera_state_says_how_old_its_picture_is(tmp_path) -> None:
    """"Current view" is a claim, so the page is given the number to back it up."""
    import numpy as np
    from starlette.testclient import TestClient

    from stuhi_vision.latest import LatestFrames
    from stuhi_vision.web.app import create_app
    from stuhi_vision.zones import ZoneStore

    frames = LatestFrames(lambda image: b"jpeg-bytes")
    frames.put("door-in", np.zeros((4, 4, 3), dtype=np.uint8))

    review = ReviewQueue(tmp_path / "review", FaceGallery(), tmp_path / "gallery")
    client = TestClient(create_app(review, frames=frames, zones=ZoneStore(tmp_path)))
    state = client.get("/api/cameras").json()["cameras"][0]

    assert state["name"] == "door-in"
    assert 0.0 <= state["frame_age"] < 5.0
    assert client.get("/frame/door-in.jpg").content == b"jpeg-bytes"


def test_the_pages_version_their_stylesheet_so_a_fix_is_not_invisible(tmp_path) -> None:
    """A cached stylesheet made a deployed CSS fix look like no fix at all.

    The server was serving the corrected file and the page was still broken on screen. A token
    that changes with the files is what makes a deploy visible without anybody remembering to
    hard-refresh.
    """
    from starlette.testclient import TestClient

    from stuhi_vision.recognition.gallery import FaceGallery
    from stuhi_vision.review import ReviewQueue
    from stuhi_vision.web.app import create_app

    review = ReviewQueue(tmp_path / "review", FaceGallery(), tmp_path / "gallery")
    client = TestClient(create_app(review))

    for page in ("/", "/zones"):
        body = client.get(page).text
        assert "/static/app.css?v=" in body
        assert "?v={{" not in body   # the token is rendered, not left as a placeholder
