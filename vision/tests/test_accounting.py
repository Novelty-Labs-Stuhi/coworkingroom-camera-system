"""Does the count add up: entries with no exit, and exits the room renamed."""

from __future__ import annotations

from stuhi_vision.accounting import Crossing, renamed_exits, unmatched_entries, with_missed

DAY = 1_785_500_000.0


def _in(name: str, at: float) -> Crossing:
    return Crossing(timestamp=at, name=name, direction="in", named_by="face")


def _out(name: str, at: float, named_by: str = "face", natural: str = "") -> Crossing:
    return Crossing(
        timestamp=at, name=name, direction="out", named_by=named_by, natural=natural
    )


def test_a_person_who_came_and_went_is_not_flagged() -> None:
    crossings = [_in("art", DAY + 10), _out("art", DAY + 200)]

    assert unmatched_entries(crossings, DAY, DAY + 1000) == []


def test_an_entry_with_no_exit_is_flagged() -> None:
    """They stay inside for ever and the count climbs: this is the fault that matters."""
    crossings = [_in("art", DAY + 10), _in("ilari", DAY + 20), _out("ilari", DAY + 90)]

    open_entries = unmatched_entries(crossings, DAY, DAY + 1000)

    assert [entry.name for entry in open_entries] == ["art"]
    assert open_entries[0].entered_at == DAY + 10


def test_two_entries_in_a_row_leave_the_first_open() -> None:
    """Which is the point: an exit between them went unrecorded."""
    crossings = [_in("art", DAY + 10), _in("art", DAY + 500), _out("art", DAY + 900)]

    open_entries = unmatched_entries(crossings, DAY, DAY + 1000)

    assert [entry.entered_at for entry in open_entries] == [DAY + 10]


def test_the_passages_it_refused_are_offered_as_the_evidence() -> None:
    """The missing exit is usually right there, thrown away for a reason now visible."""
    open_entries = [
        entry for entry in unmatched_entries([_in("art", DAY + 10)], DAY, DAY + 1000)
    ]
    refused = [
        {"timestamp": DAY + 50, "frames": 3, "direction": "out", "counted": 0},
        {"timestamp": DAY - 500, "frames": 4, "direction": "out", "counted": 0},
    ]

    with_evidence = with_missed(open_entries, refused, until=DAY + 1000)

    # Only what happened after they came in: a refusal from before says nothing about this.
    assert [passage["timestamp"] for passage in with_evidence[0].missed] == [DAY + 50]
    assert "1 passage(s) were seen afterwards and not counted" in with_evidence[0].readable


def test_an_exit_the_room_renamed_is_paired_with_that_person_s_entry() -> None:
    """Both halves have to be right: confirming one without the other proves nothing."""
    crossings = [
        _in("art", DAY + 10),
        _out("art", DAY + 300, named_by="pool", natural="ilari"),
    ]

    pairs = renamed_exits(crossings, DAY, DAY + 1000)

    assert len(pairs) == 1
    assert pairs[0].given == "art"
    assert pairs[0].natural == "ilari"       # its own evidence disagreed
    assert pairs[0].entry_at == DAY + 10


def test_an_exit_named_by_its_own_face_is_not_a_pair() -> None:
    """Nothing was overruled, so there is nothing to check against the room."""
    crossings = [_in("art", DAY + 10), _out("art", DAY + 300, named_by="face")]

    assert renamed_exits(crossings, DAY, DAY + 1000) == []


def test_a_renamed_exit_with_no_entry_still_shows_up() -> None:
    """Worse, not better: the room named somebody who is not recorded as having come in."""
    crossings = [_out("art", DAY + 300, named_by="pool")]

    pairs = renamed_exits(crossings, DAY, DAY + 1000)

    assert pairs[0].entry_at is None


def test_only_the_asked_for_window_is_accounted(tmp_path=None) -> None:
    """The day starts at half past four; yesterday's crossings are yesterday's problem."""
    crossings = [_in("art", DAY - 10_000), _in("ilari", DAY + 10)]

    assert [entry.name for entry in unmatched_entries(crossings, DAY, DAY + 1000)] == ["ilari"]
