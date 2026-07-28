"""Labelling by replying with a name, and naming a whole group in crossing order."""

from __future__ import annotations

import numpy as np

from stuhi_vision.domain import Direction, Outcome, Sighting
from stuhi_vision.notify.telegram import parse_names
from stuhi_vision.recognition.gallery import FaceGallery
from stuhi_vision.review import LabelOutcome, ReviewQueue


def _sighting(seed: int) -> Sighting:
    # Distinct embeddings, or "already enrolled" would collapse different people into one.
    embedding = np.zeros(4, dtype=np.float32)
    embedding[seed % 4] = 1.0
    return Sighting(
        timestamp=1_760_000_000.0 + seed,
        direction=Direction.IN,
        name=None,
        score=0.1,
        outcome=Outcome.UNKNOWN,
        face_embedding=embedding,
    )


def _queue(tmp_path):
    gallery = FaceGallery()
    return ReviewQueue(tmp_path / "review", gallery, tmp_path / "gallery"), gallery


def test_a_bare_name_is_accepted() -> None:
    assert parse_names("ilari") == ["ilari"]
    assert parse_names("  ilari  ") == ["ilari"]


def test_a_comma_list_names_a_group() -> None:
    assert parse_names("a, b, c") == ["a", "b", "c"]
    assert parse_names("a,b,c") == ["a", "b", "c"]


def test_brackets_are_optional_decoration() -> None:
    # People write the list either way; the brackets carry no meaning.
    assert parse_names("[a, b, c]") == ["a", "b", "c"]


def test_names_may_contain_spaces() -> None:
    assert parse_names("ilari sell, mark p") == ["ilari sell", "mark p"]


def test_commands_are_not_treated_as_names() -> None:
    # Otherwise a mistyped command would enrol a face called "pending".
    assert parse_names("/pending") == []
    assert parse_names("") == []


def test_a_burst_is_labelled_in_crossing_order(tmp_path) -> None:
    queue, gallery = _queue(tmp_path)
    ids = [queue.record(_sighting(index), position=index + 1, burst=7) for index in range(3)]

    results = queue.label_burst(ids[1], ["a", "b", "c"])

    assert [outcome for _id, outcome in results] == [LabelOutcome.ENROLLED] * 3
    assert [sighting_id for sighting_id, _o in results] == ids  # first crosser first
    assert gallery.counts() == {"a": 1, "b": 1, "c": 1}


def test_replying_to_any_member_labels_the_whole_group(tmp_path) -> None:
    queue, _ = _queue(tmp_path)
    ids = [queue.record(_sighting(index), position=index + 1, burst=3) for index in range(2)]

    # It should not matter which of the group's videos was replied to.
    assert [i for i, _o in queue.label_burst(ids[1], ["x", "y"])] == ids


def test_too_few_names_leaves_the_rest_unlabelled(tmp_path) -> None:
    queue, gallery = _queue(tmp_path)
    for index in range(3):
        queue.record(_sighting(index), position=index + 1, burst=5)
    first = queue.burst_members(queue.pending()[0].sighting_id)[0].sighting_id

    queue.label_burst(first, ["a"])

    # A miscounted group must not attach a name to the wrong person.
    assert gallery.counts() == {"a": 1}


def test_a_lone_sighting_is_its_own_burst(tmp_path) -> None:
    queue, gallery = _queue(tmp_path)
    sighting_id = queue.record(_sighting(1))

    queue.label_burst(sighting_id, ["ilari", "ignored"])

    assert gallery.counts() == {"ilari": 1}
