"""One label should correct every clip of that person -- and never invent a phantom.

Most crossings will never be looked at by a human. What makes labelling worth doing at all is
that naming somebody *once* repairs the record of every time they came through, including the
clips nobody will ever open.
"""

from __future__ import annotations

import numpy as np

from stuhi_vision.domain import Direction, Event, Outcome, Sighting
from stuhi_vision.merges import MergeLog
from stuhi_vision.presence import Passage, visits_from
from stuhi_vision.provisional import Provisional
from stuhi_vision.recognition.gallery import FaceGallery
from stuhi_vision.review import ReviewQueue
from stuhi_vision.store import EventStore


def _face(seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    vector = generator.normal(size=8).astype(np.float32)
    return vector / np.linalg.norm(vector)


def _built(tmp_path):
    store = EventStore(tmp_path / "occupancy.db")
    provisional = Provisional(tmp_path / "provisional")
    merges = MergeLog(tmp_path / "merges.jsonl")
    review = ReviewQueue(
        tmp_path / "review",
        FaceGallery(),
        tmp_path / "gallery",
        history=store,
        provisional=provisional,
        merges=merges,
    )
    return review, store, provisional, merges


def _crossing(review, name, at, direction, seed, score=0.5):
    """File a sighting and record the matching occupancy event, as the pipeline would."""
    sighting = Sighting(
        timestamp=at,
        direction=direction,
        name=name,
        score=score,
        outcome=Outcome.NAMED,
        face_embedding=_face(seed),
    )
    return review.record(sighting)


def _visit(review, store, name, entered, left, seed=0):
    sighting_id = _crossing(review, name, entered, Direction.IN, seed)
    store.record(Event(entered, name, Direction.IN))
    _crossing(review, name, left, Direction.OUT, seed + 100)
    store.record(Event(left, name, Direction.OUT))
    return sighting_id


def _names(store) -> list[str]:
    return [name for name, _, _ in store.passages()]


def test_naming_an_invented_identity_corrects_every_crossing_it_made(tmp_path) -> None:
    """The leverage: one label, and a week of unreviewed clips are right."""
    review, store, provisional, _ = _built(tmp_path)
    provisional.add("guest-0803")
    first = _visit(review, store, "guest-0803", 100.0, 200.0, seed=1)
    _visit(review, store, "guest-0803", 300.0, 400.0, seed=2)

    review.label(first, "ilari")

    assert _names(store) == ["ilari"] * 4


def test_labelling_never_leaves_half_a_visit_behind(tmp_path) -> None:
    """Renaming one crossing of a pair makes an entry that never closes: a phantom occupant."""
    review, store, provisional, _ = _built(tmp_path)
    provisional.add("guest-0803")
    entry = _visit(review, store, "guest-0803", 100.0, 200.0, seed=1)

    review.label(entry, "ilari")

    visits = visits_from([Passage(n, at, d) for n, at, d in store.passages()])
    assert [(v.name, v.left) for v in visits] == [("ilari", 200.0)]
    assert all(visit.left is not None for visit in visits)  # nobody is left inside


def test_correcting_a_recognised_person_touches_only_that_crossing(tmp_path) -> None:
    """Arsenii is still Arsenii everywhere else; the claim is about this passage alone."""
    review, store, _, _ = _built(tmp_path)
    _visit(review, store, "arsenii", 100.0, 200.0, seed=1)
    mistaken = _crossing(review, "arsenii", 300.0, Direction.IN, seed=3)
    store.record(Event(300.0, "arsenii", Direction.IN))

    review.label(mistaken, "ilari")

    assert sorted(_names(store)) == ["arsenii", "arsenii", "ilari"]


def test_the_merge_is_recorded_so_it_can_be_taken_back(tmp_path) -> None:
    review, store, provisional, merges = _built(tmp_path)
    provisional.add("guest-0803")
    entry = _visit(review, store, "guest-0803", 100.0, 200.0, seed=1)

    review.label(entry, "ilari")

    recorded = merges.all()
    assert [(m.absorbed, m.into) for m in recorded] == [("guest-0803", "ilari")]
    assert len(recorded[0].events) == 2  # both halves of the visit


def test_an_undone_merge_restores_exactly_the_crossings_it_moved(tmp_path) -> None:
    """Ilari's own visits must not be dragged back with the mistake."""
    from stuhi_vision.merges import undo

    review, store, provisional, merges = _built(tmp_path)
    _visit(review, store, "ilari", 10.0, 20.0, seed=9)      # genuinely Ilari
    provisional.add("guest-0803")
    entry = _visit(review, store, "guest-0803", 100.0, 200.0, seed=1)
    review.label(entry, "ilari")

    undo(store, merges.latest_for("guest-0803"))

    assert sorted(_names(store)) == ["guest-0803", "guest-0803", "ilari", "ilari"]


def test_a_named_identity_stops_being_offered_for_naming(tmp_path) -> None:
    review, store, provisional, _ = _built(tmp_path)
    provisional.add("guest-0803")
    entry = _visit(review, store, "guest-0803", 100.0, 200.0, seed=1)

    review.label(entry, "ilari")

    assert not provisional.holds("guest-0803")


def test_labelling_still_works_with_no_history_or_merge_log(tmp_path) -> None:
    """The label is the thing being asked for; the figures are a consequence, not a gate."""
    review = ReviewQueue(tmp_path / "review", FaceGallery(), tmp_path / "gallery")
    sighting_id = _crossing(review, "guest-0803", 100.0, Direction.IN, seed=1)

    assert review.label(sighting_id, "ilari").succeeded
