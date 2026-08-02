"""Folding one identity into another must be worth doing, and possible to take back."""

from __future__ import annotations

from pathlib import Path

from stuhi_vision.domain import Direction, Event
from stuhi_vision.merges import LABELLED, SUGGESTED, Merge, MergeLog, merge_identity, undo
from stuhi_vision.store import EventStore


def _store(tmp_path: Path) -> EventStore:
    return EventStore(tmp_path / "occupancy.db")


def _visit(store: EventStore, name: str, entered: float, left: float) -> None:
    store.record(Event(entered, name, Direction.IN))
    store.record(Event(left, name, Direction.OUT))


def _names(store: EventStore) -> list[str]:
    return [name for name, _, _ in store.passages()]


# --- the log --------------------------------------------------------------------------------


def test_a_merge_is_written_down_and_read_back(tmp_path) -> None:
    log = MergeLog(tmp_path / "merges.jsonl")

    log.record(Merge(at=100.0, absorbed="guest-a", into="ilari", because=LABELLED, events=[1, 2]))

    assert [m.readable for m in log.all()] == ["guest-a -> ilari (labelled, 2 crossing(s))"]


def test_the_log_only_ever_grows(tmp_path) -> None:
    log = MergeLog(tmp_path / "merges.jsonl")
    log.record(Merge(at=1.0, absorbed="a", into="x", because=LABELLED))
    log.record(Merge(at=2.0, absorbed="b", into="y", because=SUGGESTED))

    assert [(m.absorbed, m.into) for m in log.all()] == [("a", "x"), ("b", "y")]


def test_a_truncated_final_line_does_not_lose_the_earlier_merges(tmp_path) -> None:
    """A crash mid-write must not take the ability to undo everything before it."""
    path = tmp_path / "merges.jsonl"
    log = MergeLog(path)
    log.record(Merge(at=1.0, absorbed="a", into="x", because=LABELLED, events=[1]))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"at": 2.0, "absorbed": "b", "int')

    assert [m.absorbed for m in log.all()] == ["a"]


def test_a_missing_log_reads_as_no_merges(tmp_path) -> None:
    assert MergeLog(tmp_path / "never-written").all() == []


# --- merging --------------------------------------------------------------------------------


def test_a_merge_renames_every_crossing_of_that_identity(tmp_path) -> None:
    """One label, every past clip of that person corrected -- the whole point."""
    store = _store(tmp_path)
    _visit(store, "guest-a", 100.0, 200.0)
    _visit(store, "guest-a", 300.0, 400.0)
    log = MergeLog(tmp_path / "merges.jsonl")

    merge_identity(store, log, "guest-a", "ilari", LABELLED, now=500.0)

    assert _names(store) == ["ilari"] * 4


def test_both_halves_of_a_visit_move_together(tmp_path) -> None:
    """Renaming one crossing of a pair leaves an entry that never closes: a phantom."""
    from stuhi_vision.presence import Passage, visits_from

    store = _store(tmp_path)
    _visit(store, "guest-a", 100.0, 200.0)
    log = MergeLog(tmp_path / "merges.jsonl")

    merge_identity(store, log, "guest-a", "ilari", LABELLED, now=500.0)

    visits = visits_from([Passage(n, at, d) for n, at, d in store.passages()])
    assert [(v.name, v.left) for v in visits] == [("ilari", 200.0)]


def test_the_merge_is_logged_before_the_rows_change(tmp_path) -> None:
    """A recorded-but-unapplied merge is recoverable; the reverse is silent and permanent."""
    store = _store(tmp_path)
    _visit(store, "guest-a", 100.0, 200.0)
    seen = []

    class _Watching(MergeLog):
        def record(self, merge):
            seen.append(_names(store))   # what the store looked like when the log was written
            super().record(merge)

    merge_identity(store, _Watching(tmp_path / "m.jsonl"), "guest-a", "ilari", LABELLED, 500.0)

    assert seen == [["guest-a", "guest-a"]]


def test_merging_a_name_into_itself_is_refused(tmp_path) -> None:
    store = _store(tmp_path)
    _visit(store, "ilari", 100.0, 200.0)
    log = MergeLog(tmp_path / "merges.jsonl")

    assert merge_identity(store, log, "ilari", "ilari", LABELLED, now=500.0) is None
    assert log.all() == []


def test_an_identity_with_no_crossings_merges_harmlessly(tmp_path) -> None:
    store = _store(tmp_path)
    log = MergeLog(tmp_path / "merges.jsonl")

    merge = merge_identity(store, log, "guest-never-seen", "ilari", LABELLED, now=500.0)

    assert merge is not None and merge.events == []


# --- undoing --------------------------------------------------------------------------------


def test_a_merge_can_be_taken_back_exactly(tmp_path) -> None:
    store = _store(tmp_path)
    _visit(store, "guest-a", 100.0, 200.0)
    log = MergeLog(tmp_path / "merges.jsonl")
    merge = merge_identity(store, log, "guest-a", "ilari", LABELLED, now=500.0)

    undo(store, merge)

    assert _names(store) == ["guest-a", "guest-a"]


def test_undoing_leaves_the_target_s_own_crossings_alone(tmp_path) -> None:
    """The reason ids are recorded: undoing by name would drag Ilari's real visits back too."""
    store = _store(tmp_path)
    _visit(store, "ilari", 50.0, 60.0)          # Ilari's own visit, nothing to do with the merge
    _visit(store, "guest-a", 100.0, 200.0)
    log = MergeLog(tmp_path / "merges.jsonl")
    merge = merge_identity(store, log, "guest-a", "ilari", LABELLED, now=500.0)

    undo(store, merge)

    assert sorted(_names(store)) == ["guest-a", "guest-a", "ilari", "ilari"]


def test_the_log_can_find_the_merge_to_undo(tmp_path) -> None:
    store = _store(tmp_path)
    _visit(store, "guest-a", 100.0, 200.0)
    log = MergeLog(tmp_path / "merges.jsonl")
    merge_identity(store, log, "guest-a", "ilari", LABELLED, now=500.0)

    undo(store, log.latest_for("guest-a"))

    assert _names(store) == ["guest-a", "guest-a"]
