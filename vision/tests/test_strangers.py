"""Somebody unnamed must be recognisable again without competing with somebody named.

Three rules have to hold together, and each one alone breaks the system:

* an uncertain face in the named gallery makes every future match against that name worse;
* an unrecognisable stranger becomes a new person every visit, so their exit never closes and
  the count drifts;
* forcing a genuinely new person onto the nearest known name puts two people wrong at once.
"""

from __future__ import annotations

import numpy as np

from stuhi_vision.strangers import Strangers


def _face(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def test_a_stranger_met_before_is_recognised_again(tmp_path) -> None:
    """The rule that keeps the count right: their exit can close the entry they already have."""
    strangers = Strangers(tmp_path / "strangers")
    strangers.remember("guest-0803", _face(1.0, 0.0, 0.0))

    assert strangers.recognise(_face(0.97, 0.24, 0.0)) == "guest-0803"


def test_somebody_genuinely_new_is_not_forced_onto_the_nearest_stranger(tmp_path) -> None:
    strangers = Strangers(tmp_path / "strangers")
    strangers.remember("guest-0803", _face(1.0, 0.0, 0.0))

    assert strangers.recognise(_face(0.0, 1.0, 0.0)) is None


def test_a_face_equally_like_two_strangers_joins_neither(tmp_path) -> None:
    strangers = Strangers(tmp_path / "strangers")
    strangers.remember("guest-a", _face(1.0, 0.0, 0.0))
    strangers.remember("guest-b", _face(0.0, 1.0, 0.0))

    assert strangers.recognise(_face(0.71, 0.71, 0.0)) is None


def test_the_bar_here_is_lower_than_the_bar_for_a_named_person(tmp_path) -> None:
    """Deliberate: merging two strangers costs one card, naming the wrong person costs two.

    A score of 0.32 would be refused at the door -- ``face_match`` is 0.35 -- and is accepted
    here, which is the whole point. Being cautious about which stranger somebody is bought
    nothing and cost an identity per visit.
    """
    from stuhi_vision.strangers import SIMILARITY

    strangers = Strangers(tmp_path / "strangers")
    strangers.remember("guest-a", _face(1.0, 0.0, 0.0))

    marginal = _face(0.32, 0.947, 0.0)  # cosine ~0.32 against guest-a
    assert SIMILARITY < 0.35
    assert strangers.recognise(marginal) == "guest-a"


def test_no_faces_at_all_recognises_nobody(tmp_path) -> None:
    assert Strangers(tmp_path / "strangers").recognise(_face(1.0, 0.0, 0.0)) is None


def test_a_crossing_with_no_face_recognises_nobody(tmp_path) -> None:
    strangers = Strangers(tmp_path / "strangers")
    strangers.remember("guest-a", _face(1.0, 0.0, 0.0))

    assert strangers.recognise(None) is None


def test_remembering_nothing_is_harmless(tmp_path) -> None:
    strangers = Strangers(tmp_path / "strangers")

    strangers.remember("guest-a", None)

    assert strangers.names == []


def test_the_faces_survive_a_restart(tmp_path) -> None:
    """Otherwise every restart makes strangers of everybody again."""
    Strangers(tmp_path / "strangers").remember("guest-a", _face(1.0, 0.0, 0.0))

    assert Strangers(tmp_path / "strangers").recognise(_face(0.99, 0.14, 0.0)) == "guest-a"


def test_a_named_identity_can_be_forgotten(tmp_path) -> None:
    strangers = Strangers(tmp_path / "strangers")
    strangers.remember("guest-a", _face(1.0, 0.0, 0.0))

    strangers.forget("guest-a")

    assert strangers.names == []
    assert Strangers(tmp_path / "strangers").names == []  # and stays gone after a reload


def test_forgetting_somebody_unknown_is_harmless(tmp_path) -> None:
    Strangers(tmp_path / "strangers").forget("never-existed")


def test_a_returning_stranger_creates_no_second_identity(tmp_path) -> None:
    """The failure this exists to stop: eight visits became eight people, none recognisable."""
    strangers = Strangers(tmp_path / "strangers")
    first = _face(1.0, 0.02, 0.0)
    strangers.remember("guest-0803-070244", first)

    for visit in range(8):
        again = _face(1.0, 0.02 + visit * 0.01, 0.0)
        assert strangers.recognise(again) == "guest-0803-070244"

    assert strangers.names == ["guest-0803-070244"]
