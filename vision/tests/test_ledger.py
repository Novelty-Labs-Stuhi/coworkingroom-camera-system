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
