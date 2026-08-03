"""The profile: the name, the bio, the picture, the year of days, and one day opened up."""

from __future__ import annotations

import json
import time
from datetime import date, timedelta

import numpy as np
from starlette.testclient import TestClient

from stuhi_vision.domain import Direction, Event, Outcome, Sighting
from stuhi_vision.presence import day_starting, office_day
from stuhi_vision.recognition.gallery import FaceGallery
from stuhi_vision.review import ReviewQueue
from stuhi_vision.store import EventStore
from stuhi_vision.web.app import create_app
from stuhi_vision.web.faces import representative


def _client(tmp_path, passages=(), history_wanted: bool = True):
    """A client whose counting began long ago, so the figures are about the crossings."""
    (tmp_path / "review").mkdir(parents=True, exist_ok=True)
    (tmp_path / "review" / "stats-epoch.json").write_text(
        json.dumps({"from": 0.0}), encoding="utf-8"
    )
    history = None
    if history_wanted:
        history = EventStore(tmp_path / "events.db")
        for name, at, direction in passages:
            history.record(
                Event(
                    timestamp=at,
                    name=name,
                    direction=Direction.IN if direction == "in" else Direction.OUT,
                    camera="door-in",
                )
            )
    review = ReviewQueue(tmp_path / "review", FaceGallery(), tmp_path / "gallery", history)
    return TestClient(create_app(review, history=history)), review, history


def _face(review, name: str, vector: list[float], at: float, with_crop: bool = True) -> str:
    """A labelled sighting, with a picture beside it unless a test wants one without."""
    sighting_id = review.record(
        Sighting(
            timestamp=at,
            direction=Direction.IN,
            name=None,
            score=0.5,
            outcome=Outcome.UNKNOWN,
            face_embedding=np.array(vector, dtype=float),
            face_crop=object() if with_crop else None,
        ),
        encode_jpeg=(lambda _crop: b"not really a jpeg") if with_crop else None,
    )
    review.label(sighting_id, name)
    return sighting_id


def _hours_ago(hours: float) -> float:
    return time.time() - hours * 3600


def _as_date(iso: str) -> date:
    return date.fromisoformat(iso)


# --- the header -----------------------------------------------------------------------------


def test_a_profile_has_a_name_no_bio_and_the_figures(tmp_path) -> None:
    """Everybody starts with an empty bio, which is normal rather than missing data."""
    client, _, history = _client(
        tmp_path, [("ilari", _hours_ago(5), "in"), ("ilari", _hours_ago(2), "out")]
    )

    profile = client.get("/api/profile/ilari").json()

    assert profile["name"] == "ilari"
    assert profile["bio"] == ""
    assert profile["totals"]["week"]["seconds"] >= 3 * 3600 - 5
    assert profile["inside"] is False
    assert profile["visits"] == 1
    history.close()


def test_somebody_still_inside_says_so_on_their_profile(tmp_path) -> None:
    client, _, history = _client(tmp_path, [("art", _hours_ago(1), "in")])

    assert client.get("/api/profile/art").json()["inside"] is True
    history.close()


def test_the_picture_is_the_face_closest_to_the_average(tmp_path) -> None:
    client, review, history = _client(tmp_path)
    # Eight faces of one person, and one that is nothing like them: the odd one is what the
    # rule drops, and it must not become the portrait.
    for index in range(8):
        _face(review, "art", [1.0, 0.004 * index, 0.0], 1_785_000_000.0 + index)
    odd = _face(review, "art", [0.0, 1.0, 0.0], 1_785_000_100.0)

    profile = client.get("/api/profile/art").json()
    frames = client.get("/api/person/art/faces?sort=used").json()["frames"]

    assert profile["picture"] != odd
    assert profile["picture"] == frames[0]["id"]     # in use, closest to the average
    # Nine kept, and four fifths of nine is seven of them matched against.
    assert profile["faces"] == {"in_use": 7, "kept": 9}
    history.close()


def test_a_person_with_no_saved_picture_has_no_portrait(tmp_path) -> None:
    """Real people can have reference vectors and no crop, and a portrait is not invented."""
    client, review, history = _client(tmp_path)
    _face(review, "art", [1.0, 0.0, 0.0], 1_785_000_000.0, with_crop=False)

    profile = client.get("/api/profile/art").json()

    assert profile["picture"] is None
    assert profile["faces"]["kept"] == 1
    history.close()


def test_a_face_in_use_is_preferred_over_a_marginally_closer_unused_one() -> None:
    """A profile should not be represented by a picture the system has stopped believing in."""
    frames = [
        {"id": "old", "in_use": False, "closeness": 0.99, "has_picture": True},
        {"id": "current", "in_use": True, "closeness": 0.90, "has_picture": True},
    ]

    assert representative(frames) == "current"


def test_a_face_with_no_picture_is_never_the_portrait() -> None:
    frames = [
        {"id": "vector-only", "in_use": True, "closeness": 0.99, "has_picture": False},
        {"id": "has-one", "in_use": False, "closeness": 0.10, "has_picture": True},
    ]

    assert representative(frames) == "has-one"
    assert representative([]) is None


# --- the bio --------------------------------------------------------------------------------


def test_a_bio_can_be_written_and_comes_back_on_the_profile(tmp_path) -> None:
    client, _, history = _client(tmp_path)

    saved = client.post("/api/bio", json={"name": "ilari", "bio": "Sits by the window."})

    assert saved.status_code == 200
    assert client.get("/api/profile/ilari").json()["bio"] == "Sits by the window."
    history.close()


def test_a_bio_that_does_not_fit_is_refused_with_a_reason(tmp_path) -> None:
    client, _, history = _client(tmp_path)

    refused = client.post("/api/bio", json={"name": "ilari", "bio": "x" * 501})

    assert refused.status_code == 400
    assert "500" in refused.json()["detail"]
    history.close()


def test_correcting_the_name_carries_the_bio_with_it(tmp_path) -> None:
    """Spelling somebody's name properly must not cost them their profile."""
    at = _hours_ago(4)
    client, review, history = _client(tmp_path, [("ilari", at, "in")])
    _face(review, "ilari", [1.0, 0.0, 0.0], at)
    client.post("/api/bio", json={"name": "ilari", "bio": "Sits by the window."})

    renamed = client.post("/api/rename", json={"old": "ilari", "new": "Ilari"})

    assert renamed.status_code == 200
    assert client.get("/api/profile/Ilari").json()["bio"] == "Sits by the window."
    assert client.get("/api/profile/ilari").json()["bio"] == ""
    # And the hours moved too, or one person would read as two half-present ones.
    assert client.get("/api/profile/Ilari").json()["visits"] == 1
    history.close()


# --- the chart ------------------------------------------------------------------------------


def test_the_chart_has_a_box_for_every_day_ending_today(tmp_path) -> None:
    client, _, history = _client(
        tmp_path, [("ilari", _hours_ago(6), "in"), ("ilari", _hours_ago(1), "out")]
    )

    chart = client.get("/api/profile/ilari/activity?weeks=4").json()

    today = office_day(time.time())
    # Four weeks back, then pulled to a Monday so every column is a whole week.
    expected = 28 + (today - timedelta(days=27)).weekday()
    assert chart["until"] == today.isoformat()
    assert len(chart["days"]) == expected
    assert chart["days"][0]["date"] == chart["from"]
    assert chart["days"][-1]["date"] == chart["until"]
    # Contiguous and whole weeks, or the boxes would sit in the wrong columns.
    assert len({day["date"] for day in chart["days"]}) == len(chart["days"])
    assert timedelta(days=expected - 1) == today - _as_date(chart["from"])
    history.close()


def test_a_day_in_the_room_is_shaded_and_a_day_away_is_not(tmp_path) -> None:
    client, _, history = _client(
        tmp_path, [("ilari", _hours_ago(6), "in"), ("ilari", _hours_ago(1), "out")]
    )

    days = {day["date"]: day for day in client.get("/api/profile/ilari/activity").json()["days"]}
    today = office_day(time.time())

    assert days[today.isoformat()]["level"] > 0
    assert days[today.isoformat()]["readable"]
    assert days[(today - timedelta(days=3)).isoformat()]["level"] == 0
    history.close()


def test_the_chart_marks_the_days_before_counting_began(tmp_path) -> None:
    """An empty box then means nobody was looking, which is not the same as nobody being here."""
    (tmp_path / "review").mkdir(parents=True, exist_ok=True)
    (tmp_path / "review" / "stats-epoch.json").write_text(
        json.dumps({"from": day_starting(office_day(time.time()))}), encoding="utf-8"
    )
    history = EventStore(tmp_path / "events.db")
    review = ReviewQueue(tmp_path / "review", FaceGallery(), tmp_path / "gallery", history)
    client = TestClient(create_app(review, history=history))

    days = {day["date"]: day for day in client.get("/api/profile/ilari/activity").json()["days"]}
    today = office_day(time.time())

    assert days[today.isoformat()]["counted"] is True
    assert days[(today - timedelta(days=5)).isoformat()]["counted"] is False
    history.close()


# --- one day --------------------------------------------------------------------------------


def test_a_day_lists_the_crossings_with_the_frame_each_came_from(tmp_path) -> None:
    """The frames are the point: a wrong box on the chart is a wrong label on a frame."""
    entered, left = _hours_ago(6), _hours_ago(1)
    client, review, history = _client(
        tmp_path, [("ilari", entered, "in"), ("ilari", left, "out")]
    )
    frame = _face(review, "ilari", [1.0, 0.0, 0.0], entered)
    today = office_day(time.time()).isoformat()

    day = client.get(f"/api/profile/ilari/day/{today}").json()

    assert [crossing["direction"] for crossing in day["crossings"]] == ["in", "out"]
    assert day["crossings"][0]["sighting_id"] == frame
    assert day["crossings"][0]["has_picture"] is True
    assert day["crossings"][0]["labelled_as"] == "ilari"
    # The exit had no sighting recorded, and says so rather than offering a dead control.
    assert day["crossings"][1]["sighting_id"] is None
    assert day["readable"]
    history.close()


def test_a_day_shows_only_that_person_and_only_that_day(tmp_path) -> None:
    today = office_day(time.time())
    client, _, history = _client(
        tmp_path,
        [
            ("ilari", _hours_ago(3), "in"),
            ("art", _hours_ago(3), "in"),
            ("ilari", day_starting(today - timedelta(days=2)) + 9 * 3600, "in"),
        ],
    )

    day = client.get(f"/api/profile/ilari/day/{today.isoformat()}").json()

    assert len(day["crossings"]) == 1
    assert day["today"] is True
    history.close()


def test_a_frame_relabelled_here_moves_the_crossing_off_this_day(tmp_path) -> None:
    """Correcting a label on the timeline has to move the hours, or the chart keeps a lie."""
    entered = _hours_ago(4)
    client, review, history = _client(tmp_path, [("guest-9", entered, "in")])
    frame = _face(review, "guest-9", [1.0, 0.0, 0.0], entered)
    today = office_day(time.time()).isoformat()
    assert len(client.get(f"/api/profile/guest-9/day/{today}").json()["crossings"]) == 1

    corrected = client.post("/api/label", json={"sighting_id": frame, "name": "ilari"})

    assert corrected.status_code == 200
    assert client.get(f"/api/profile/guest-9/day/{today}").json()["crossings"] == []
    assert len(client.get(f"/api/profile/ilari/day/{today}").json()["crossings"]) == 1
    history.close()


def test_a_date_that_is_not_a_date_is_refused(tmp_path) -> None:
    client, _, history = _client(tmp_path)

    refused = client.get("/api/profile/ilari/day/last-tuesday")

    assert refused.status_code == 400
    history.close()


def test_a_profile_still_works_where_no_crossings_are_recorded(tmp_path) -> None:
    """A labelling-only deployment has no history, and a profile is still worth showing."""
    client, review, _ = _client(tmp_path, history_wanted=False)
    _face(review, "art", [1.0, 0.0, 0.0], 1_785_000_000.0)
    client.post("/api/bio", json={"name": "art", "bio": "Here anyway."})

    profile = client.get("/api/profile/art").json()
    chart = client.get("/api/profile/art/activity?weeks=2").json()
    day = client.get(f"/api/profile/art/day/{office_day(time.time()).isoformat()}").json()

    assert profile["bio"] == "Here anyway."
    assert profile["picture"] is not None
    assert profile["totals"]["all"]["seconds"] == 0
    assert all(box["level"] == 0 for box in chart["days"])
    assert day["crossings"] == []


def test_the_profile_page_is_served(tmp_path) -> None:
    client, _, history = _client(tmp_path)

    page = client.get("/profile?name=ilari")

    assert page.status_code == 200
    assert "profile.js" in page.text
    assert "calendar.js" in page.text
    history.close()
