"""Tests for the occupancy ledger and exit attribution (pure, no models)."""

from __future__ import annotations

import numpy as np

from stuhi_vision.domain import Event
from stuhi_vision.ledger import Ledger


class FakeSink:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def record(self, event: Event) -> None:
        self.events.append(event)


def _ledger(similarity: float = 0.5, margin: float = 0.1) -> Ledger:
    return Ledger(FakeSink(), exit_similarity=similarity, exit_margin=margin)


def test_single_occupant_exit_by_elimination() -> None:
    ledger = _ledger()
    ledger.enter("alice", np.array([1.0, 0.0]), timestamp=1.0)
    # A body nothing like alice's still resolves: she is the only candidate.
    assert ledger.exit(np.array([0.0, 1.0]), timestamp=2.0) == "alice"
    assert ledger.occupancy == []


def test_exit_matches_best_pair_with_clear_margin() -> None:
    ledger = _ledger()
    ledger.enter("alice", np.array([1.0, 0.0, 0.0]), timestamp=1.0)
    ledger.enter("bob", np.array([0.0, 1.0, 0.0]), timestamp=2.0)
    assert ledger.exit(np.array([0.9, 0.1, 0.0]), timestamp=3.0) == "alice"
    assert ledger.occupancy == ["bob"]


def test_exit_unresolved_when_ambiguous() -> None:
    ledger = _ledger(similarity=0.5, margin=0.1)
    ledger.enter("alice", np.array([1.0, 0.0]), timestamp=1.0)
    ledger.enter("bob", np.array([0.9, 0.1]), timestamp=2.0)  # very close to alice
    who = ledger.exit(np.array([1.0, 0.0]), timestamp=3.0)  # matches both nearly equally
    assert who is None
    assert ledger.occupancy == ["alice", "bob"]  # nobody removed on a coin-flip


def test_exit_unresolved_when_below_threshold() -> None:
    ledger = _ledger(similarity=0.95, margin=0.05)
    ledger.enter("alice", np.array([1.0, 0.0]), timestamp=1.0)
    ledger.enter("bob", np.array([0.0, 1.0]), timestamp=2.0)
    who = ledger.exit(np.array([0.7, 0.7]), timestamp=3.0)  # ~0.71 to each, under 0.95
    assert who is None


def test_events_are_journalled() -> None:
    sink = FakeSink()
    ledger = Ledger(sink, exit_similarity=0.5, exit_margin=0.1)
    ledger.enter("alice", np.array([1.0, 0.0]), timestamp=1.0)
    ledger.exit(np.array([1.0, 0.0]), timestamp=2.0)
    assert [(e.name, e.direction.value) for e in sink.events] == [("alice", "in"), ("alice", "out")]


def test_an_unattributable_exit_is_still_recorded() -> None:
    # Writing nothing meant occupancy only ever grew: the count drifted upward permanently
    # and no amount of walking out could correct it. The gap belongs in the history.
    sink = FakeSink()
    ledger = Ledger(sink, exit_similarity=0.95, exit_margin=0.05)

    assert ledger.exit(None, timestamp=1.0) is None
    assert [(e.name, e.direction.value) for e in sink.events] == [("unknown", "out")]


def test_a_face_seen_on_the_way_out_names_the_exit() -> None:
    # A camera facing people as they leave recognises them exactly as one facing arrivals
    # does. That name is trusted over the body match, which exists only for the other case.
    sink = FakeSink()
    ledger = Ledger(sink, exit_similarity=0.95, exit_margin=0.05)
    ledger.enter("alice", np.array([1.0, 0.0]), timestamp=1.0, camera="door-in")
    ledger.enter("bob", np.array([0.0, 1.0]), timestamp=1.5, camera="door-in")

    # No body embedding at all, which would previously have resolved to nobody.
    assert ledger.exit(None, timestamp=2.0, camera="door-out", name="bob") == "bob"
    assert ledger.occupancy == ["alice"]
    assert sink.events[-1].camera == "door-out"


def test_a_face_naming_somebody_not_inside_falls_back_to_the_body_match() -> None:
    ledger = Ledger(FakeSink(), exit_similarity=0.5, exit_margin=0.05)
    ledger.enter("alice", np.array([1.0, 0.0]), timestamp=1.0)

    # "carol" was never inside, so the claim is ignored and elimination applies instead.
    assert ledger.exit(np.array([1.0, 0.0]), timestamp=2.0, name="carol") == "alice"


def test_events_carry_the_camera_that_saw_them() -> None:
    sink = FakeSink()
    ledger = Ledger(sink, exit_similarity=0.5, exit_margin=0.05)
    ledger.enter("alice", None, timestamp=1.0, camera="door-in")

    assert sink.events[0].camera == "door-in"


def _ranker(scores: dict[str, float]):
    """Stand in for the gallery: fixed scores, best first, as rank() returns them."""
    from stuhi_vision.recognition.embeddings import Match

    ordered = sorted(scores.items(), key=lambda pair: -pair[1])
    return lambda embedding: [Match(name=name, score=score) for name, score in ordered]


def test_an_exit_is_named_from_the_people_inside_by_a_face_too_weak_for_the_door() -> None:
    """The whole point: a poor face only has to beat the people in the room.

    At the door a face competes with everybody enrolled and must clear a threshold that keeps
    strangers out. Leaving, the answer is almost certainly one of two or three known occupants,
    so a low-resolution camera can still settle it.
    """
    sink = FakeSink()
    ledger = Ledger(
        sink,
        exit_similarity=0.9,
        exit_margin=0.05,
        faces=_ranker({"alice": 0.28, "bob": 0.11, "stranger": 0.95}),
        face_similarity=0.22,
    )
    ledger.enter("alice", np.array([1.0, 0.0]), timestamp=1.0)
    ledger.enter("bob", np.array([0.0, 1.0]), timestamp=2.0)

    # 0.28 would be "unknown" at the door, and the strongest match overall is somebody who is
    # not even in the room -- so only the occupants may be considered.
    assert ledger.exit(None, timestamp=3.0, face_embedding=np.array([1.0])) == "alice"
    assert ledger.occupancy == ["bob"]


def test_a_face_that_suits_two_occupants_equally_names_neither() -> None:
    sink = FakeSink()
    ledger = Ledger(
        sink,
        exit_similarity=0.9,
        exit_margin=0.05,
        faces=_ranker({"alice": 0.30, "bob": 0.29}),
        face_similarity=0.22,
    )
    ledger.enter("alice", np.array([1.0, 0.0]), timestamp=1.0)
    ledger.enter("bob", np.array([0.0, 1.0]), timestamp=2.0)

    assert ledger.exit(None, timestamp=3.0, face_embedding=np.array([1.0])) is None
    assert ledger.occupancy == ["alice", "bob"]      # nobody removed on a coin-flip


def test_a_name_a_camera_read_still_wins_over_the_pool() -> None:
    sink = FakeSink()
    ledger = Ledger(
        sink,
        exit_similarity=0.9,
        exit_margin=0.05,
        faces=_ranker({"alice": 0.40, "bob": 0.10}),
        face_similarity=0.22,
    )
    ledger.enter("alice", np.array([1.0, 0.0]), timestamp=1.0)
    ledger.enter("bob", np.array([0.0, 1.0]), timestamp=2.0)

    who = ledger.exit(None, timestamp=3.0, name="bob", face_embedding=np.array([1.0]))
    assert who == "bob"


def test_the_pool_is_ignored_when_nobody_is_inside() -> None:
    ledger = Ledger(
        FakeSink(),
        exit_similarity=0.9,
        exit_margin=0.05,
        faces=_ranker({"alice": 0.99}),
        face_similarity=0.22,
    )
    assert ledger.exit(None, timestamp=3.0, face_embedding=np.array([1.0])) is None
