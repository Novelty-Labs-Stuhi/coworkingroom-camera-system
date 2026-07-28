"""Auditing the gallery: which enrolled faces look wrong, and who needs more examples."""

from __future__ import annotations

import numpy as np

from stuhi_vision.enrolment import Reference, audit
from stuhi_vision.recognition.embeddings import normalize


def _vector(*components: float) -> np.ndarray:
    return normalize(np.array(components, dtype=np.float32))


def _cluster(name: str, count: int, spread: float = 0.02) -> list[Reference]:
    """``count`` near-identical faces for one person -- a healthy enrolment."""
    return [
        Reference(f"{name}-{index}", name, _vector(1.0, index * spread, 0.0))
        for index in range(count)
    ]


def test_a_tight_cluster_has_no_suspects() -> None:
    assert audit(_cluster("ilari", 5)).suspects == []


def test_a_face_from_someone_else_is_flagged() -> None:
    references = [*_cluster("ilari", 4), Reference("stranger", "ilari", _vector(0.0, 0.0, 1.0))]

    report = audit(references)

    assert [suspect.sighting_id for suspect in report.suspects] == ["stranger"]
    assert report.suspects[0].reason == "not the same face"
    assert report.suspects[0].similarity < report.suspects[0].average


def test_a_merely_unlike_face_is_flagged_with_the_softer_reason() -> None:
    # Same person, but a poor capture: similar enough to clear the absolute floor, far
    # enough from the others to be worth a look.
    references = [*_cluster("ilari", 4), Reference("odd", "ilari", _vector(1.0, 1.15, 0.0))]

    report = audit(references)

    assert [s.sighting_id for s in report.suspects] == ["odd"]
    assert report.suspects[0].reason == "unlike this person's other faces"


def test_two_references_cannot_produce_a_relative_outlier() -> None:
    # Each sits the same distance from their midpoint by construction, so a relative
    # comparison is meaningless -- which is exactly why "thin" is reported separately.
    references = [
        Reference("a", "ilari", _vector(1.0, 0.0, 0.0)),
        Reference("b", "ilari", _vector(1.0, 0.6, 0.0)),
    ]

    report = audit(references)

    assert [s.reason for s in report.suspects] == []
    assert report.thin == {"ilari": 2}


def test_two_references_still_catch_a_completely_wrong_face() -> None:
    references = [
        Reference("a", "ilari", _vector(1.0, 0.0, 0.0)),
        Reference("b", "ilari", _vector(0.0, 0.0, 1.0)),
    ]

    assert [s.reason for s in audit(references).suspects] == [
        "not the same face",
        "not the same face",
    ]


def test_people_with_too_few_faces_are_listed() -> None:
    references = [*_cluster("ilari", 5), *_cluster("mark", 1), *_cluster("sam", 2)]

    report = audit(references)

    assert report.thin == {"mark": 1, "sam": 2}
    assert "ilari" not in report.thin


def test_suspects_come_worst_first() -> None:
    references = [
        *_cluster("ilari", 4),
        Reference("bad", "ilari", _vector(0.0, 0.0, 1.0)),
        Reference("worse", "ilari", _vector(0.0, 0.0, -1.0)),
    ]

    similarities = [s.similarity for s in audit(references).suspects]

    assert similarities == sorted(similarities)


def test_each_person_is_judged_against_their_own_faces() -> None:
    # Two tight but mutually distant clusters: neither should contaminate the other.
    references = [
        *_cluster("ilari", 3),
        Reference("m1", "mark", _vector(0.0, 1.0, 0.0)),
        Reference("m2", "mark", _vector(0.02, 1.0, 0.0)),
        Reference("m3", "mark", _vector(0.0, 1.0, 0.02)),
    ]

    assert audit(references).suspects == []
