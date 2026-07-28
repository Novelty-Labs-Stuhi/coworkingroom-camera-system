"""Labelling a sighting enrols it; labelling again corrects it."""

from __future__ import annotations

import numpy as np

from stuhi_vision.domain import Direction, Outcome, Sighting
from stuhi_vision.recognition.gallery import FaceGallery
from stuhi_vision.review import ReviewQueue


def _sighting(embedding: np.ndarray | None = None, **overrides) -> Sighting:
    defaults = {
        "timestamp": 1_760_000_000.0,
        "direction": Direction.IN,
        "name": None,
        "score": 0.21,
        "outcome": Outcome.UNKNOWN,
        "face_embedding": embedding if embedding is not None else np.array([1.0, 0.0, 0.0]),
        "face_crop": None,
    }
    return Sighting(**{**defaults, **overrides})


def _queue(tmp_path, gallery: FaceGallery | None = None) -> tuple[ReviewQueue, FaceGallery]:
    gallery = gallery or FaceGallery()
    queue = ReviewQueue(tmp_path / "review", gallery, tmp_path / "gallery")
    return queue, gallery


def test_labelling_enrols_the_face(tmp_path) -> None:
    queue, gallery = _queue(tmp_path)
    sighting_id = queue.record(_sighting())

    assert queue.label(sighting_id, "ilari") is True
    assert gallery.counts() == {"ilari": 1}
    assert (tmp_path / "gallery" / "ilari.npy").exists()


def test_relabelling_moves_the_face_and_leaves_no_trace(tmp_path) -> None:
    queue, gallery = _queue(tmp_path)
    sighting_id = queue.record(_sighting())

    queue.label(sighting_id, "ilari")
    queue.label(sighting_id, "mark")  # correction

    assert gallery.counts() == {"mark": 1}
    assert "ilari" not in gallery.names
    # The emptied name's file must go too, or a reload resurrects the wrong label.
    assert not (tmp_path / "gallery" / "ilari.npy").exists()


def test_relabelling_to_the_same_name_does_not_duplicate(tmp_path) -> None:
    queue, gallery = _queue(tmp_path)
    sighting_id = queue.record(_sighting())

    queue.label(sighting_id, "ilari")
    queue.label(sighting_id, "ilari")

    assert gallery.counts() == {"ilari": 1}


def test_unknown_id_is_refused(tmp_path) -> None:
    queue, _ = _queue(tmp_path)
    assert queue.label("2020-01-01_00-00-00", "ilari") is False


def test_sighting_without_a_face_cannot_be_enrolled(tmp_path) -> None:
    queue, gallery = _queue(tmp_path)
    sighting_id = queue.record(_sighting(face_embedding=None, outcome=Outcome.UNIDENTIFIED))

    assert queue.label(sighting_id, "ilari") is False
    assert gallery.counts() == {}


def test_pending_lists_only_enrollable_unlabelled_sightings(tmp_path) -> None:
    queue, _ = _queue(tmp_path)
    unknown_id = queue.record(_sighting())
    queue.record(_sighting(face_embedding=None, outcome=Outcome.UNIDENTIFIED))

    assert [record.sighting_id for record in queue.pending()] == [unknown_id]

    queue.label(unknown_id, "ilari")
    assert queue.pending() == []


def test_records_survive_a_restart(tmp_path) -> None:
    queue, gallery = _queue(tmp_path)
    sighting_id = queue.record(_sighting())
    queue.label(sighting_id, "ilari")

    reopened = ReviewQueue(tmp_path / "review", gallery, tmp_path / "gallery")
    assert reopened.pending() == []  # already labelled, not offered again


def test_simultaneous_sightings_get_distinct_ids(tmp_path) -> None:
    queue, _ = _queue(tmp_path)
    first = queue.record(_sighting())
    second = queue.record(_sighting())
    assert first != second
