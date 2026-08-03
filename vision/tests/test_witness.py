"""One camera counts the passage, the other names who it was.

The doorway camera can tell an exit from background traffic, because a person passing occludes
the doorframe -- but it sees them from behind. The room camera sees the same person walk at it
face-first. These tests cover the handover between the two, and that only one of them counts.
"""

from __future__ import annotations

import numpy as np

from stuhi_vision.domain import Crossing, Direction, Outcome
from stuhi_vision.handlers import Doorkeeper, Enrolment, Identifier
from stuhi_vision.ledger import Ledger
from stuhi_vision.witness import LeavingWitness


class FakeSink:
    def __init__(self) -> None:
        self.events = []

    def record(self, event) -> None:
        self.events.append(event)


class FakeIdentity:
    def __init__(self, name: str | None, score: float = 0.8) -> None:
        self._name = name
        self._score = score

    def decide(self):
        from stuhi_vision.identity import Decision

        if self._name is None:
            return Decision(name=None, score=self._score, margin=0.0, outcome=Outcome.UNKNOWN)
        return Decision(name=self._name, score=self._score, margin=0.2, outcome=Outcome.NAMED)


class FakeSession:
    """Just the parts a handler reads off a finished track."""

    def __init__(self, name: str | None, age: int = 5) -> None:
        self.identity = FakeIdentity(name)
        self.age = age
        self.body_embedding = np.array([1.0, 0.0])
        self.entry_embedding = np.array([1.0, 0.0])
        self.face_embedding = None
        self.face_crop = None


class FakeSessions:
    def __init__(self, session) -> None:
        self._session = session

    def pop(self, track_id: int):
        return self._session


def _crossing(direction: Direction, timestamp: float = 100.0) -> Crossing:
    return Crossing(track_id=1, direction=direction, timestamp=timestamp)


def _ledger(sink: FakeSink) -> Ledger:
    return Ledger(sink, exit_similarity=0.99, exit_margin=0.5)


def test_the_identifying_camera_counts_nothing() -> None:
    sink = FakeSink()
    ledger = _ledger(sink)
    ledger.enter("alice", np.array([1.0, 0.0]), timestamp=1.0)
    witness = LeavingWitness()

    identifier = Identifier(FakeSessions(FakeSession("alice")), witness, min_track_age=2)
    sighting = identifier.commit(_crossing(Direction.OUT))

    # It saw her leave and said so, but the count is not its business: both cameras see every
    # passage, so a second committer would make one exit into two.
    assert witness.waiting() == ["alice"]
    assert ledger.occupancy == ["alice"]
    assert sighting is not None and sighting.name == "alice"


def test_an_exit_seen_from_behind_is_named_by_the_other_camera() -> None:
    sink = FakeSink()
    ledger = _ledger(sink)
    ledger.enter("alice", np.array([1.0, 0.0]), timestamp=1.0)
    witness = LeavingWitness()

    Identifier(FakeSessions(FakeSession("alice")), witness, min_track_age=2).commit(
        _crossing(Direction.OUT, timestamp=98.0)
    )
    # The doorway camera saw only the back of a head, so its own decision names nobody.
    door = Doorkeeper(
        FakeSessions(FakeSession(None)), ledger, min_track_age=2, witness=witness
    )
    sighting = door.commit(_crossing(Direction.OUT, timestamp=100.0))

    assert ledger.occupancy == []          # alice is out, by name
    assert sink.events[-1].name == "alice"
    assert sighting is not None


def test_two_people_leaving_are_not_both_recorded_as_the_first() -> None:
    sink = FakeSink()
    ledger = _ledger(sink)
    for name in ("alice", "bob"):
        ledger.enter(name, np.array([1.0, 0.0]), timestamp=1.0)
    witness = LeavingWitness()

    for name, seen in (("alice", 98.0), ("bob", 99.0)):
        Identifier(FakeSessions(FakeSession(name)), witness, min_track_age=2).commit(
            _crossing(Direction.OUT, timestamp=seen)
        )

    door = Doorkeeper(
        FakeSessions(FakeSession(None)), ledger, min_track_age=2, witness=witness
    )
    door.commit(_crossing(Direction.OUT, timestamp=98.2))   # nearest to alice's sighting
    door.commit(_crossing(Direction.OUT, timestamp=99.4))   # nearest to bob's

    assert [event.name for event in sink.events[-2:]] == ["alice", "bob"]
    assert ledger.occupancy == []


def test_a_name_does_not_survive_long_enough_to_claim_a_later_exit() -> None:
    witness = LeavingWitness(window_seconds=10.0)
    witness.note("alice", 0.9, timestamp=100.0)

    assert witness.claim(timestamp=105.0) == "alice"
    witness.note("bob", 0.9, timestamp=200.0)
    # An exit half an hour later is not bob walking out again; it is somebody unknown.
    assert witness.claim(timestamp=2000.0) is None


def test_a_face_seen_by_the_counting_camera_wins_over_the_other_camera() -> None:
    sink = FakeSink()
    ledger = _ledger(sink)
    for name in ("alice", "bob"):
        ledger.enter(name, np.array([1.0, 0.0]), timestamp=1.0)
    witness = LeavingWitness()
    witness.note("bob", 0.9, timestamp=99.0)

    # This camera recognised alice itself. Direct evidence beats a name passed between
    # cameras, and the unclaimed one stays available for the exit it belongs to.
    door = Doorkeeper(
        FakeSessions(FakeSession("alice")), ledger, min_track_age=2, witness=witness
    )
    door.commit(_crossing(Direction.OUT, timestamp=100.0))

    assert sink.events[-1].name == "alice"
    assert witness.waiting() == ["bob"]


def test_an_unrecognised_arrival_gets_an_identity_that_lasts(tmp_path) -> None:
    """Enrolled, so the same person coming back is matched instead of becoming somebody else.

    The old label came from a counter that restarts at one in every process -- so a restart
    re-issued names two different people then shared -- and the face was never enrolled, so
    every return visit created yet another identity. That is how a room fills with guests who
    never leave: an exit can only be matched to somebody the recogniser can find.
    """
    import numpy as np

    from stuhi_vision.recognition.gallery import FaceGallery
    from stuhi_vision.strangers import Strangers

    gallery = FaceGallery()
    strangers = Strangers(tmp_path / "strangers")
    ledger = _ledger(FakeSink())
    face = np.array([1.0, 0.0], dtype=np.float32)
    session = FakeSession(None)
    session.face_embedding = face

    door = Doorkeeper(
        FakeSessions(session),
        ledger,
        min_track_age=2,
        enrolment=Enrolment(strangers, tmp_path / "gallery"),
    )
    sighting = door.commit(_crossing(Direction.IN, timestamp=1_785_600_000.0))

    assert sighting is not None
    name = sighting.name
    assert name.startswith("guest-")
    assert strangers.names == [name]             # findable next time
    # ...but nowhere near the people who have names: an uncertain face must not be able to
    # compete with Ilari's, which is why the two stores are separate rather than one flagged.
    assert gallery.names == []
    assert ledger.occupancy == [name]
    # Derived from the moment, so a restart cannot hand the same name to somebody else.
    assert name != "guest-1"
