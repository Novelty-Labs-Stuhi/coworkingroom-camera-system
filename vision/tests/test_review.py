"""Labelling a sighting enrols it; labelling again corrects it."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from stuhi_vision.domain import Direction, Outcome, Sighting
from stuhi_vision.recognition.gallery import FaceGallery
from stuhi_vision.review import LabelOutcome, ReviewQueue


def _sighting(
    embedding: np.ndarray | None = None,
    timestamp: float = 1_760_000_000.0,
    **overrides,
) -> Sighting:
    defaults = {
        "timestamp": timestamp,
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

    assert queue.label(sighting_id, "ilari") is LabelOutcome.ENROLLED
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
    assert queue.label("2020-01-01_00-00-00", "ilari") is LabelOutcome.UNKNOWN_ID


def test_sighting_without_a_face_cannot_be_enrolled(tmp_path) -> None:
    queue, gallery = _queue(tmp_path)
    sighting_id = queue.record(_sighting(face_embedding=None, outcome=Outcome.UNIDENTIFIED))

    assert queue.label(sighting_id, "ilari") is LabelOutcome.NO_FACE
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


def test_labelling_the_same_way_twice_adds_nothing(tmp_path) -> None:
    # One sighting must contribute exactly one reference however many times it is
    # labelled, or a repeated command quietly over-weights that one face.
    queue, gallery = _queue(tmp_path)
    sighting_id = queue.record(_sighting())

    assert queue.label(sighting_id, "ilari") is LabelOutcome.ENROLLED
    assert queue.label(sighting_id, "ilari") is LabelOutcome.UNCHANGED
    assert queue.label(sighting_id, "ilari") is LabelOutcome.UNCHANGED
    assert gallery.counts() == {"ilari": 1}


def test_correcting_a_label_is_reported_as_a_correction(tmp_path) -> None:
    queue, gallery = _queue(tmp_path)
    sighting_id = queue.record(_sighting())

    queue.label(sighting_id, "ilari")
    assert queue.label(sighting_id, "mark") is LabelOutcome.CORRECTED
    assert gallery.counts() == {"mark": 1}


def test_rejecting_a_sighting_removes_its_reference(tmp_path) -> None:
    # A back-of-head capture must stop being offered *and* stop dragging the person's
    # average; hiding the card while leaving the reference would degrade matches invisibly.
    queue, gallery = _queue(tmp_path)
    sighting_id = queue.record(_sighting())
    queue.label(sighting_id, "ilari")

    assert queue.dismiss(sighting_id) is LabelOutcome.DISMISSED
    assert gallery.counts() == {}
    assert queue.pending() == []


def test_rejecting_an_unlabelled_sighting_just_hides_it(tmp_path) -> None:
    queue, gallery = _queue(tmp_path)
    sighting_id = queue.record(_sighting())

    assert queue.dismiss(sighting_id) is LabelOutcome.DISMISSED
    assert queue.pending() == []
    assert gallery.counts() == {}


def test_a_rejected_sighting_stays_rejected_after_a_restart(tmp_path) -> None:
    queue, gallery = _queue(tmp_path)
    sighting_id = queue.record(_sighting())
    queue.dismiss(sighting_id)

    reopened = ReviewQueue(tmp_path / "review", gallery, tmp_path / "gallery")
    assert reopened.pending() == []


def test_references_tie_enrolled_faces_back_to_their_sightings(tmp_path) -> None:
    queue, _ = _queue(tmp_path)
    first = queue.record(_sighting())
    second = queue.record(_sighting(embedding=np.array([0.0, 1.0, 0.0])))
    queue.label(first, "ilari")
    queue.label(second, "mark")

    references = {r.sighting_id: r.name for r in queue.references()}

    assert references == {first: "ilari", second: "mark"}


def test_rejected_faces_are_left_out_of_the_audit(tmp_path) -> None:
    queue, _ = _queue(tmp_path)
    sighting_id = queue.record(_sighting())
    queue.label(sighting_id, "ilari")
    queue.dismiss(sighting_id)

    assert queue.references() == []


def test_group_sizes_count_everyone_who_crossed_together(tmp_path) -> None:
    queue, _ = _queue(tmp_path)
    queue.record(_sighting(), position=1, burst=4)
    queue.record(_sighting(), position=2, burst=4)
    queue.record(_sighting(), position=1, burst=5)

    assert queue.group_sizes() == {4: 2, 5: 1}


def test_unlabelling_returns_a_sighting_to_the_pending_list(tmp_path) -> None:
    queue, gallery = _queue(tmp_path)
    sighting_id = queue.record(_sighting())
    queue.label(sighting_id, "ilari")

    assert queue.unlabel(sighting_id) is LabelOutcome.UNLABELLED
    assert gallery.counts() == {}
    assert [r.sighting_id for r in queue.pending()] == [sighting_id]


def test_unlabelling_something_unlabelled_changes_nothing(tmp_path) -> None:
    queue, _ = _queue(tmp_path)
    sighting_id = queue.record(_sighting())
    assert queue.unlabel(sighting_id) is LabelOutcome.UNCHANGED


def test_composite_labels_are_reported(tmp_path) -> None:
    # These are not people: they came from free text passed through as a single name.
    queue, _ = _queue(tmp_path)
    good = queue.record(_sighting())
    bad = queue.record(_sighting(embedding=np.array([0.0, 1.0, 0.0])))
    queue.label(good, "ilari")
    queue.label(bad, "a, yehor")

    assert [r.sighting_id for r in queue.composite_labels()] == [bad]


def test_renaming_corrects_the_name_on_every_clip_and_in_the_gallery(tmp_path) -> None:
    """A misspelling is one mistake, not one per sighting."""
    review, gallery = _queue(tmp_path)
    first = review.record(_sighting(np.array([1.0, 0.0, 0.0])))
    second = review.record(_sighting(np.array([0.0, 1.0, 0.0])))
    review.label(first, "ilar")
    review.label(second, "ilar")

    moved = review.rename("ilar", "ilari")

    assert moved == 2
    assert review.counts() == {"ilari": 2}
    assert review.get(first).labelled_as == "ilari"
    assert gallery.names == ["ilari"]


def test_renaming_onto_an_existing_name_merges_them(tmp_path) -> None:
    """"ilari" and "Ilari" being the same person is exactly what a merge means."""
    review, _ = _queue(tmp_path)
    review.label(review.record(_sighting(np.array([1.0, 0.0, 0.0]))), "ilari")
    review.label(review.record(_sighting(np.array([0.0, 1.0, 0.0]))), "Ilari")

    review.rename("Ilari", "ilari")

    assert review.counts() == {"ilari": 2}


def test_renaming_survives_a_reload(tmp_path) -> None:
    review, _ = _queue(tmp_path)
    review.label(review.record(_sighting()), "yehor")
    review.rename("yehor", "Yehor")

    reloaded = ReviewQueue(
        tmp_path / "review", FaceGallery.load(tmp_path / "gallery"), tmp_path / "gallery"
    )
    assert reloaded.counts() == {"Yehor": 1}


def test_renaming_a_name_nobody_has_changes_nothing(tmp_path) -> None:
    review, _ = _queue(tmp_path)
    assert review.rename("nobody", "somebody") == 0
    assert review.rename("same", "same") == 0


def test_regrouping_rebuilds_groups_from_the_gaps_between_crossings(tmp_path) -> None:
    """The recorded groups were wrong; the timestamps were not."""
    review, _ = _queue(tmp_path)
    together = [review.record(_sighting(), position=index, burst=1) for index in (1, 2)]
    later = review.record(
        _sighting(timestamp=1_760_000_400.0), position=3, burst=1  # minutes later
    )

    changed = review.regroup(gap_seconds=3.0)

    groups = review.groups()
    assert changed >= 1
    # Groups are keyed by the moment they began, so a repaired group cannot collide with one
    # the pipeline issues after a restart.
    assert sorted(groups[1_760_000_000]) == sorted(together)
    assert groups[1_760_000_400] == [later]        # the straggler is its own group
    assert review.get(later).position == 1


def test_regrouping_is_recorded_so_it_survives_a_reload(tmp_path) -> None:
    review, _ = _queue(tmp_path)
    review.record(_sighting(), position=1, burst=1)
    review.record(_sighting(timestamp=1_760_000_900.0), position=2, burst=1)
    review.regroup(gap_seconds=3.0)

    reloaded = ReviewQueue(tmp_path / "review", FaceGallery(), tmp_path / "gallery")
    assert len(reloaded.groups()) == 2


def test_a_saved_label_never_returns_to_worth_rechecking(tmp_path) -> None:
    """Whatever the numbers say, somebody has judged it -- offering it back argues with them."""
    review, _ = _queue(tmp_path)
    once = review.record(_sighting())
    review.label(once, "ilar")           # used exactly once: what a typo looks like

    assert [record.sighting_id for record, _ in review.worth_rechecking()] == []


def test_a_name_used_once_is_worth_rechecking_until_somebody_saves_it(tmp_path) -> None:
    review, _ = _queue(tmp_path)
    typo = review.record(_sighting())
    review.label(typo, "ilar")
    # Undo the "checked" mark the way an import or an older record would look.
    review._records[typo] = type(review._records[typo])(
        **{**asdict(review._records[typo]), "checked": False}
    )

    flagged = review.worth_rechecking()
    assert [record.sighting_id for record, _ in flagged] == [typo]
    assert "used only once" in flagged[0][1]

    review.label(typo, "ilari")          # corrected and saved
    assert review.worth_rechecking() == []


def test_recent_names_are_offered_most_recently_used_first(tmp_path) -> None:
    review, _ = _queue(tmp_path)
    for offset, name in enumerate(("art", "ilari", "art", "yehor")):
        review.label(review.record(_sighting(timestamp=1_760_000_000.0 + offset)), name)

    # "art" was used again after "ilari", so it leads; each name appears once.
    assert review.recent_names() == ["yehor", "art", "ilari"]


def test_every_sighting_lands_in_a_pile(tmp_path) -> None:
    """The system labels everything it sees, so nothing may fall between the piles."""
    review, _ = _queue(tmp_path)
    unknown = review.record(_sighting())
    guessed = review.record(
        _sighting(timestamp=1_760_000_100.0, name="art", outcome=Outcome.NAMED)
    )
    done = review.record(_sighting(timestamp=1_760_000_200.0))
    review.label(done, "ilari")

    piles = review.piles()

    assert [r.sighting_id for r in piles["unchecked_unknown"]] == [unknown]
    assert [r.sighting_id for r in piles["unchecked_named"]] == [guessed]
    assert [r.sighting_id for r in piles["checked"]] == [done]


def test_somebody_saved_as_unknown_sits_with_the_unknowns(tmp_path) -> None:
    """A person saying "unknown" is a decision, and it needs a different action from a name."""
    review, _ = _queue(tmp_path)
    said_unknown = review.record(_sighting())
    review.label(said_unknown, "unknown")

    piles = review.piles()
    assert [r.sighting_id for r in piles["checked_unknown"]] == [said_unknown]
    assert [r.sighting_id for r in piles["checked"]] == [said_unknown]


def test_rejecting_keeps_the_reason_where_it_can_be_read(tmp_path) -> None:
    review, _ = _queue(tmp_path)
    bad = review.record(_sighting())

    review.dismiss(bad, "that is the door swinging, not a person")

    assert review.get(bad).rejected_because == "that is the door swinging, not a person"
    # Rejected is a decision too, so it leaves the queue rather than being offered again.
    assert [r.sighting_id for r in review.piles()["unchecked_unknown"]] == []


def test_saving_as_unknown_checks_it_without_enrolling_anybody(tmp_path) -> None:
    """"unknown" is not a person: a gallery entry by that name would match everybody."""
    review, gallery = _queue(tmp_path)
    nobody = review.record(_sighting())

    assert review.label(nobody, "unknown") is LabelOutcome.SET_ASIDE

    assert gallery.counts() == {}
    assert review.get(nobody).labelled_as == "unknown"
    assert review.get(nobody).checked is True
    # Out of the queue: otherwise the only way to clear an unrecognisable frame would be to
    # give it somebody's name.
    assert [r.sighting_id for r in review.piles()["unchecked_unknown"]] == []
    assert [r.sighting_id for r in review.piles()["checked_unknown"]] == [nobody]


def test_saving_as_unknown_withdraws_a_name_it_used_to_carry(tmp_path) -> None:
    review, gallery = _queue(tmp_path)
    mislabelled = review.record(_sighting())
    review.label(mislabelled, "ilari")
    assert gallery.counts() == {"ilari": 1}

    review.label(mislabelled, "unknown")

    assert gallery.counts() == {}   # it must stop influencing recognition


def test_a_sighting_with_no_face_can_still_be_set_aside(tmp_path) -> None:
    """Nothing to enrol, but it must not be stuck in the queue for ever."""
    review, _ = _queue(tmp_path)
    faceless = review.record(_sighting(embedding=None))
    (tmp_path / "review" / f"{faceless}.npy").unlink(missing_ok=True)

    assert review.label(faceless, "somebody") is LabelOutcome.NO_FACE
    assert review.label(faceless, "unknown") is LabelOutcome.SET_ASIDE


class FakeHistory:
    """The event log, only as much of it as a corrected label touches."""

    def __init__(self) -> None:
        self.renamed: list[tuple[float, str, str]] = []

    def rename_crossing(self, at: float, direction: str, name: str, window: float = 1.0) -> int:
        self.renamed.append((at, direction, name))
        return 1


def test_a_corrected_label_reaches_the_crossing_it_is_about(tmp_path) -> None:
    """The time-in-the-room figures come from the crossings, not from the review records.

    Without this the totals keep whatever the system guessed at the time, and no amount of
    careful labelling would ever change them.
    """
    history = FakeHistory()
    review = ReviewQueue(tmp_path / "review", FaceGallery(), tmp_path / "gallery", history)
    sighting_id = review.record(_sighting())

    review.label(sighting_id, "ilari")

    assert history.renamed == [(1_760_000_000.0, "in", "ilari")]


def test_setting_aside_makes_the_crossing_unknown_too(tmp_path) -> None:
    """Leaving the guess there would credit somebody with hours nobody claims."""
    history = FakeHistory()
    review = ReviewQueue(tmp_path / "review", FaceGallery(), tmp_path / "gallery", history)
    guessed = review.record(_sighting(name="art", outcome=Outcome.NAMED))

    review.label(guessed, "unknown")

    assert history.renamed[-1] == (1_760_000_000.0, "in", "unknown")


def test_a_label_is_still_recorded_when_the_history_cannot_be_reached(tmp_path) -> None:
    """The label is what was asked for; the figures can be recomputed from the log later."""

    class Broken:
        def rename_crossing(self, *args, **kwargs):
            raise RuntimeError("database is locked")

    review = ReviewQueue(tmp_path / "review", FaceGallery(), tmp_path / "gallery", Broken())
    sighting_id = review.record(_sighting())

    assert review.label(sighting_id, "ilari") is LabelOutcome.ENROLLED
    assert review.get(sighting_id).labelled_as == "ilari"


def test_rejecting_with_a_name_records_who_it_was_without_enrolling_them(tmp_path) -> None:
    """Two different things: who came through, and what the recogniser should learn from.

    An unusable picture of a known person is still evidence they were there.
    """
    history = FakeHistory()
    gallery = FaceGallery()
    review = ReviewQueue(tmp_path / "review", gallery, tmp_path / "gallery", history)
    blurred = review.record(_sighting())

    review.dismiss(blurred, "motion blur, but that is clearly him", name="ilari")

    assert gallery.counts() == {}                       # nothing to learn from
    assert review.get(blurred).labelled_as is None      # not enrolled under anybody
    assert review.get(blurred).attributed_to == "ilari"  # but we know who it was
    assert history.renamed == [(1_760_000_000.0, "in", "ilari")]   # so the hours count


def test_rejecting_a_previously_enrolled_face_still_withdraws_it(tmp_path) -> None:
    review, gallery = _queue(tmp_path)
    mislabelled = review.record(_sighting())
    review.label(mislabelled, "ilari")

    review.dismiss(mislabelled, "back of a head", name="ilari")

    assert gallery.counts() == {}


def test_rejecting_with_no_name_leaves_the_crossing_alone(tmp_path) -> None:
    """A rejection that names nobody is not a claim about who came through."""
    history = FakeHistory()
    review = ReviewQueue(tmp_path / "review", FaceGallery(), tmp_path / "gallery", history)
    rubbish = review.record(_sighting())

    review.dismiss(rubbish, "that is the door swinging")

    assert history.renamed == []
