"""Which of a person's faces recognition uses, and why each one was left out."""

from __future__ import annotations

import numpy as np

from stuhi_vision.enrolment import Reference
from stuhi_vision.selection import choose, distances


def _face(sighting: str, name: str, vector: list[float]) -> Reference:
    return Reference(sighting_id=sighting, name=name, embedding=np.array(vector, dtype=float))


def _clique(name: str, count: int, start: int = 0, drift: float = 0.0):
    """A set of near-identical faces, optionally drifting away from each other."""
    return [
        _face(f"{name}-{index}", name, [1.0, drift * index, 0.0])
        for index in range(start, start + count)
    ]


def _ages(references, first: float = 1_000.0):
    """One second apart, in the order given, so "newest" is the end of the list."""
    return {
        reference.sighting_id: first + index for index, reference in enumerate(references)
    }


def test_everything_recent_and_alike_is_used() -> None:
    faces = _clique("art", 4)
    chosen = choose(faces, _ages(faces))["art"]

    assert len(chosen.used) == 4
    assert chosen.too_old == () and chosen.too_odd == ()


def test_the_least_like_the_average_are_not_matched_against() -> None:
    """Eight alike and two nothing like them: the two are dropped from matching."""
    faces = [
        *_clique("art", 8),
        _face("art-odd-1", "art", [0.0, 1.0, 0.0]),
        _face("art-odd-2", "art", [0.0, 0.0, 1.0]),
    ]
    chosen = choose(faces, _ages(faces), keep=0.8)["art"]

    assert len(chosen.used) == 8
    assert set(chosen.too_odd) == {"art-odd-1", "art-odd-2"}


def test_the_dropped_ones_still_shape_the_average() -> None:
    """Excluding them from the average would let a tight clique define the person.

    Five identical faces and five spread out: if the average came only from what survives, the
    average would be the clique itself and every spread face would look like an outlier. Taken
    from the whole window, the average sits between them, and the spread faces are ranked on
    how far they really are rather than how unlike the clique they are.
    """
    clique = [_face(f"art-same-{i}", "art", [1.0, 0.0, 0.0]) for i in range(5)]
    spread = [_face(f"art-wide-{i}", "art", [1.0, 0.35 + i * 0.05, 0.0]) for i in range(5)]
    faces = clique + spread

    chosen = choose(faces, _ages(faces), keep=0.8)["art"]

    # Eight of ten survive, and the two dropped are the *furthest* of the spread -- not simply
    # "everything that is not the clique", which is what a clique-only average would give.
    assert len(chosen.used) == 8
    assert set(chosen.too_odd) == {"art-wide-4", "art-wide-3"}


def test_anything_older_than_the_window_is_not_used_and_has_no_say() -> None:
    """People change: a face from months ago is evidence about somebody who looked different."""
    old = _clique("art", 3, start=0)
    recent = _clique("art", 4, start=3)
    faces = old + recent

    chosen = choose(faces, _ages(faces), newest=4, keep=1.0)["art"]

    assert set(chosen.too_old) == {"art-0", "art-1", "art-2"}
    assert set(chosen.used) == {"art-3", "art-4", "art-5", "art-6"}


def test_a_person_can_insist_on_a_face_the_rule_would_drop() -> None:
    """Somebody who has looked knows things the numbers do not."""
    faces = [*_clique("art", 8), _face("art-beard", "art", [0.0, 1.0, 0.0])]
    chosen = choose(faces, _ages(faces), decided={"art-beard": True}, keep=0.8)["art"]

    assert "art-beard" in chosen.used
    assert "art-beard" in chosen.pinned
    assert "art-beard" not in chosen.too_odd


def test_a_person_can_bar_a_face_the_rule_would_keep() -> None:
    faces = _clique("art", 5)
    chosen = choose(faces, _ages(faces), decided={"art-2": False})["art"]

    assert "art-2" not in chosen.used
    assert chosen.barred == ("art-2",)


def test_an_insisted_face_survives_being_out_of_the_window() -> None:
    """The only picture of somebody with their new beard may also be their oldest."""
    faces = _clique("art", 6)
    chosen = choose(faces, _ages(faces), decided={"art-0": True}, newest=3)["art"]

    assert "art-0" in chosen.used
    assert "art-0" not in chosen.too_old


def test_nobody_is_left_with_nothing_to_match_against() -> None:
    """Rounding must never take the last face away: that person becomes unrecognisable."""
    faces = _clique("art", 1)
    assert len(choose(faces, _ages(faces), keep=0.1)["art"].used) == 1


def test_each_person_is_chosen_from_their_own_faces_only() -> None:
    faces = _clique("art", 3) + _clique("ilari", 2)
    chosen = choose(faces, _ages(faces))

    assert set(chosen) == {"art", "ilari"}
    assert len(chosen["ilari"].used) == 2


def test_the_distance_shown_is_the_one_the_decision_used() -> None:
    """A number beside a face has to explain the decision actually made about it."""
    faces = [*_clique("art", 4), _face("art-odd", "art", [0.0, 1.0, 0.0])]
    scores = distances(faces, _ages(faces))

    assert scores["art-odd"] < scores["art-0"]
    assert 0.0 <= scores["art-odd"] <= 1.0
