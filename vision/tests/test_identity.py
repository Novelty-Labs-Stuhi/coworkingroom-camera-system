"""The running-best identity rule: threshold, margin, and the three outcomes."""

from __future__ import annotations

from stuhi_vision.domain import Outcome
from stuhi_vision.identity import RunningIdentity
from stuhi_vision.recognition.embeddings import Match


def _identity(threshold: float = 0.4, margin: float = 0.05) -> RunningIdentity:
    return RunningIdentity(threshold=threshold, margin_min=margin)


def test_no_face_is_unidentified() -> None:
    decision = _identity().decide()
    assert decision.outcome is Outcome.UNIDENTIFIED
    assert decision.name is None


def test_best_score_wins_over_the_whole_track() -> None:
    identity = _identity()
    identity.observe([Match("ilari", 0.42), Match("mark", 0.10)])
    identity.observe([Match("mark", 0.71), Match("ilari", 0.30)])
    identity.observe([Match("ilari", 0.50), Match("mark", 0.20)])

    decision = identity.decide()
    assert decision.outcome is Outcome.NAMED
    assert decision.name == "mark"  # highest score ever seen, not the latest frame
    assert decision.score == 0.71


def test_improvement_is_reported_so_the_caller_can_keep_that_frame() -> None:
    identity = _identity()
    assert identity.observe([Match("ilari", 0.30)]) is True
    assert identity.observe([Match("ilari", 0.20)]) is False  # worse look, ignore it
    assert identity.observe([Match("ilari", 0.55)]) is True


def test_below_threshold_is_unknown_not_a_guess() -> None:
    identity = _identity(threshold=0.4)
    identity.observe([Match("ilari", 0.31), Match("mark", 0.02)])

    decision = identity.decide()
    assert decision.outcome is Outcome.UNKNOWN
    assert decision.name is None  # a stranger is not force-matched to the closest name
    assert decision.score == 0.31  # the score is still reported, for tuning


def test_thin_margin_is_unknown_even_above_threshold() -> None:
    identity = _identity(threshold=0.4, margin=0.05)
    identity.observe([Match("ilari", 0.62), Match("mark", 0.60)])

    # Clears the threshold, but is nearly as close to someone else: no clear winner.
    assert identity.decide().outcome is Outcome.UNKNOWN


def test_single_enrolled_person_still_names() -> None:
    identity = _identity()
    identity.observe([Match("ilari", 0.55)])  # no runner-up to beat
    assert identity.decide().name == "ilari"


def test_empty_gallery_is_unknown_but_counts_as_seeing_a_face() -> None:
    identity = _identity()
    assert identity.observe([]) is True  # first face still captured, for enrolling later

    decision = identity.decide()
    assert decision.outcome is Outcome.UNKNOWN  # not UNIDENTIFIED -- there was a face
    assert decision.name is None
