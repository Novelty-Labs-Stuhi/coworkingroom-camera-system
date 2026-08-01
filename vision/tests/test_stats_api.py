"""The leaderboard, a person's own figures, and who the record thinks is still inside."""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
from starlette.testclient import TestClient

from stuhi_vision.domain import Direction, Event, Outcome, Sighting
from stuhi_vision.recognition.gallery import FaceGallery
from stuhi_vision.review import ReviewQueue
from stuhi_vision.store import EventStore
from stuhi_vision.web.app import create_app


def _client(tmp_path, passages, counting_since: float | None = None):
    """A client whose counting began long ago, unless a test says otherwise.

    The epoch is what stops the boards presenting months of crossings recorded while the rules
    were changing hourly. Every test about *figures* wants it out of the way; the tests about
    the epoch itself set it deliberately.
    """
    import json

    (tmp_path / "review").mkdir(parents=True, exist_ok=True)
    (tmp_path / "review" / "stats-epoch.json").write_text(
        json.dumps({"from": counting_since if counting_since is not None else 0.0}),
        encoding="utf-8",
    )
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


def _hours_ago(hours: float) -> float:
    return time.time() - hours * 3600


def test_the_board_ranks_by_time_in_the_room(tmp_path) -> None:
    client, _, history = _client(
        tmp_path,
        [
            ("ilari", _hours_ago(9), "in"), ("ilari", _hours_ago(1), "out"),
            ("art", _hours_ago(5), "in"), ("art", _hours_ago(4), "out"),
        ],
    )

    board = client.get("/api/leaderboard?window=week").json()

    assert [row["name"] for row in board["standings"]] == ["ilari", "art"]
    assert board["standings"][0]["seconds"] > board["standings"][1]["seconds"]
    assert "h" in board["standings"][0]["readable"]
    history.close()


def test_somebody_still_inside_is_shown_as_such_and_counts_up_to_now(tmp_path) -> None:
    client, _, history = _client(tmp_path, [("art", _hours_ago(2), "in")])

    board = client.get("/api/leaderboard?window=day").json()

    assert board["standings"][0]["still_inside"] is True
    assert board["standings"][0]["seconds"] > 3600
    history.close()


def test_going_back_a_window_shows_a_different_period(tmp_path) -> None:
    client, _, history = _client(
        tmp_path,
        [("ilari", _hours_ago(24 * 9), "in"), ("ilari", _hours_ago(24 * 9 - 3), "out")],
    )

    this_week = client.get("/api/leaderboard?window=week&offset=0").json()
    last_week = client.get("/api/leaderboard?window=week&offset=1").json()

    assert this_week["standings"] == []          # nine days ago is not this week
    assert [row["name"] for row in last_week["standings"]] == ["ilari"]
    history.close()


def test_a_streak_board_counts_days_not_hours(tmp_path) -> None:
    passages = []
    for days_back in (0, 1, 2):
        at = _hours_ago(24 * days_back + 6)
        passages += [("ilari", at, "in"), ("ilari", at + 1800, "out")]
    client, _, history = _client(tmp_path, passages)

    board = client.get("/api/leaderboard?window=streak").json()

    assert board["standings"][0]["name"] == "ilari"
    assert "day" in board["standings"][0]["readable"]
    history.close()


def test_a_person_page_has_their_own_figures_and_visits(tmp_path) -> None:
    client, _, history = _client(
        tmp_path, [("ilari", _hours_ago(4), "in"), ("ilari", _hours_ago(2), "out")]
    )

    figures = client.get("/api/person/ilari").json()

    assert figures["name"] == "ilari"
    assert figures["totals"]["week"]["seconds"] >= 7000
    assert figures["inside"] is False
    assert len(figures["visits"]) == 1
    history.close()


def test_the_inside_list_is_what_a_correction_may_need_to_reach(tmp_path) -> None:
    """An exit can only belong to somebody inside, so an impossible one means an entry is wrong."""
    entered = _hours_ago(3)
    client, review, history = _client(
        tmp_path,
        [("guest-2", entered, "in"), ("art", _hours_ago(2), "in"), ("art", _hours_ago(1), "out")],
    )
    # A sighting for that entry, so the clip can be found from the list.
    review.record(
        Sighting(
            timestamp=entered,
            direction=Direction.IN,
            name="guest-2",
            score=0.2,
            outcome=Outcome.UNKNOWN,
            face_embedding=np.array([1.0, 0.0, 0.0]),
            face_crop=None,
        )
    )

    listed = client.get("/api/inside").json()["inside"]

    assert [row["name"] for row in listed] == ["guest-2"]
    assert listed[0]["sighting_id"] is not None      # so its clip can be opened and corrected
    assert "h" in listed[0]["for"] or "m" in listed[0]["for"]
    history.close()


def test_nothing_is_counted_before_counting_begins(tmp_path) -> None:
    """The figures start when they were switched on, not at the start of the log.

    Otherwise months of crossings recorded while the rules were being changed hourly would be
    presented as somebody's record.
    """
    client, _, history = _client(
        tmp_path,
        [("ilari", _hours_ago(48), "in"), ("ilari", _hours_ago(40), "out")],
        counting_since=_hours_ago(24),
    )

    board = client.get("/api/leaderboard?window=all").json()

    assert board["standings"] == []   # it all happened before counting began
    history.close()


def test_a_window_says_nothing_until_one_of_it_has_passed(tmp_path) -> None:
    """A week's board on its second day looks like a week's while being a day's."""
    client, _, history = _client(
        tmp_path,
        [("ilari", _hours_ago(3), "in"), ("ilari", _hours_ago(1), "out")],
        counting_since=_hours_ago(6),
    )

    week = client.get("/api/leaderboard?window=week").json()
    running = client.get("/api/leaderboard?window=all").json()

    assert week["ready"] is False and week["standings"] == []
    assert week["ready_at"] > time.time()          # and says when it will mean something
    # The running total is honest from the first minute, being explicitly "so far".
    assert running["ready"] is True
    assert [row["name"] for row in running["standings"]] == ["ilari"]
    history.close()


def test_before_counting_starts_there_is_nothing_at_all(tmp_path) -> None:
    """Switched on at 04:30 today, asked at half past three: every board is empty."""
    client, _, history = _client(
        tmp_path,
        [("ilari", _hours_ago(3), "in")],
        counting_since=time.time() + 3600,
    )

    for window in ("streak", "day", "week", "month", "year", "all"):
        board = client.get(f"/api/leaderboard?window={window}").json()
        assert board["ready"] is False, window
        assert board["standings"] == [], window
    history.close()


def test_the_board_is_derived_so_a_correction_changes_it(tmp_path) -> None:
    """The whole reason for deriving rather than storing totals."""
    at = _hours_ago(6)
    client, review, history = _client(
        tmp_path, [("guest-9", at, "in"), ("guest-9", at + 3600, "out")]
    )
    sighting_id = review.record(
        Sighting(
            timestamp=at,
            direction=Direction.IN,
            name="guest-9",
            score=0.2,
            outcome=Outcome.UNKNOWN,
            face_embedding=np.array([1.0, 0.0, 0.0]),
            face_crop=None,
        )
    )

    assert [r["name"] for r in client.get("/api/leaderboard?window=day").json()["standings"]] == [
        "guest-9"
    ]

    review.label(sighting_id, "ilari")

    named = [r["name"] for r in client.get("/api/leaderboard?window=day").json()["standings"]]
    assert "ilari" in named
    history.close()


def test_the_day_boundary_is_half_past_four(tmp_path) -> None:
    """A visit at one in the morning belongs to the day that is ending."""
    from stuhi_vision.presence import office_day

    late = datetime.now().replace(hour=1, minute=30).timestamp()
    assert office_day(late) != datetime.fromtimestamp(late).date()
