"""The room does not carry people across a restart wrongly, nor across a night at all."""

from __future__ import annotations

from datetime import datetime, timedelta

from stuhi_vision.continuity import (
    RESET,
    DayBoundary,
    open_visits,
    partition,
    start_of_day,
)
from stuhi_vision.domain import Direction
from stuhi_vision.ledger import Ledger
from stuhi_vision.presence import OFFICE


class _Sink:
    def __init__(self) -> None:
        self.events = []

    def record(self, event) -> None:
        self.events.append(event)


def _at(day: int, hour: int, minute: int = 0) -> float:
    """A timestamp in office time, so the 04:30 boundary lands where the test means it to."""
    return datetime(2026, 8, day, hour, minute, tzinfo=OFFICE).timestamp()


def _ledger(sink: _Sink | None = None) -> tuple[Ledger, _Sink]:
    sink = sink or _Sink()
    return Ledger(sink, exit_similarity=0.6, exit_margin=0.05), sink


# --- reading the record ------------------------------------------------------------------


def test_open_visits_replays_crossings_into_who_is_inside() -> None:
    inside = open_visits(
        [
            (100.0, "ilari", "in"),
            (200.0, "arsenii", "in"),
            (300.0, "ilari", "out"),
        ]
    )

    assert inside == {"arsenii": 200.0}


def test_a_second_entry_keeps_the_earlier_start() -> None:
    """Two entries in a row means an exit was missed; the visit is as old as the first."""
    inside = open_visits([(100.0, "ilari", "in"), (500.0, "ilari", "in")])

    assert inside == {"ilari": 100.0}


def test_an_exit_for_somebody_not_inside_changes_nothing() -> None:
    assert open_visits([(100.0, "ghost", "out")]) == {}


# --- what survives a restart -------------------------------------------------------------


def test_visits_from_today_are_restored_and_older_ones_closed() -> None:
    day_began = _at(3, 4, 30)
    restore, close = partition(
        {"today": _at(3, 9), "yesterday": _at(2, 9), "overnight": _at(3, 2)}, day_began
    )

    assert set(restore) == {"today"}
    # 02:00 is before 04:30, so it belongs to the day that is ending -- not still here.
    assert set(close) == {"yesterday", "overnight"}


def test_restore_puts_people_back_without_journalling_them_again() -> None:
    ledger, sink = _ledger()

    ledger.restore({"ilari": 100.0, "arsenii": 200.0})

    assert ledger.occupancy == ["arsenii", "ilari"]
    assert sink.events == []  # these entries were recorded when they happened


def test_a_restored_occupant_can_be_exited_normally() -> None:
    """The point of restoring: their exit closes their visit instead of matching nobody."""
    ledger, sink = _ledger()
    ledger.restore({"ilari": 100.0})

    assert ledger.exit(None, 300.0) == "ilari"
    assert ledger.occupancy == []
    assert sink.events[-1].direction is Direction.OUT


def test_restore_never_displaces_somebody_already_inside() -> None:
    ledger, _ = _ledger()
    ledger.enter("ilari", None, 500.0)

    ledger.restore({"ilari": 100.0})

    assert ledger.occupancy == ["ilari"]


# --- the night boundary ------------------------------------------------------------------


def test_the_boundary_closes_everyone_when_a_new_office_day_starts() -> None:
    ledger, sink = _ledger()
    ledger.enter("ilari", None, _at(3, 22))
    ledger.enter("arsenii", None, _at(3, 23))
    boundary = DayBoundary(ledger)
    boundary.check(_at(3, 23, 30))  # same office day: nothing yet

    closed = boundary.check(_at(4, 5))  # past 04:30 -> a new office day

    assert closed == ["arsenii", "ilari"]
    assert ledger.occupancy == []
    assert [e.named_by for e in sink.events if e.direction is Direction.OUT] == [RESET, RESET]


def test_the_boundary_fires_once_per_day_however_often_it_is_asked() -> None:
    ledger, _ = _ledger()
    ledger.enter("ilari", None, _at(3, 22))
    boundary = DayBoundary(ledger)
    boundary.check(_at(3, 23))

    first = boundary.check(_at(4, 5))
    ledger.enter("arsenii", None, _at(4, 9))
    again = boundary.check(_at(4, 10))

    assert first == ["ilari"]
    assert again == []              # already accounted for today
    assert ledger.occupancy == ["arsenii"]   # somebody who arrived after it is untouched


def test_before_half_past_four_is_still_the_previous_office_day() -> None:
    ledger, _ = _ledger()
    ledger.enter("ilari", None, _at(3, 22))
    boundary = DayBoundary(ledger)
    boundary.check(_at(3, 22, 30))

    assert boundary.check(_at(4, 3)) == []      # 03:00 -- yesterday, still working
    assert boundary.check(_at(4, 4, 45)) == ["ilari"]


def test_an_empty_room_at_the_boundary_writes_nothing() -> None:
    ledger, sink = _ledger()
    boundary = DayBoundary(ledger)
    boundary.check(_at(3, 12))

    assert boundary.check(_at(4, 12)) == []
    assert sink.events == []


def test_start_of_day_is_half_past_four_office_time() -> None:
    began = start_of_day(_at(3, 12))

    moment = datetime.fromtimestamp(began, OFFICE)
    assert (moment.hour, moment.minute) == (4, 30)
    assert moment.date() == datetime(2026, 8, 3).date()


def test_a_visit_spanning_the_boundary_is_closed_not_carried() -> None:
    """The case the whole module exists for: nobody is in the room for two days."""
    ledger, _ = _ledger()
    ledger.enter("ilari", None, _at(1, 10))
    boundary = DayBoundary(ledger)
    boundary.check(_at(1, 10))

    for day in (2, 3):
        boundary.check(_at(day, 5))

    assert ledger.occupancy == []


def test_the_boundary_survives_a_gap_of_several_days() -> None:
    """Downtime over a weekend must still clear the room, not skip the boundary."""
    ledger, _ = _ledger()
    ledger.enter("ilari", None, _at(1, 10))
    boundary = DayBoundary(ledger)
    boundary.check(_at(1, 10))

    later = (datetime(2026, 8, 1, 10, tzinfo=OFFICE) + timedelta(days=4)).timestamp()

    assert boundary.check(later) == ["ilari"]


def test_two_cameras_asking_at_once_fire_the_boundary_only_once() -> None:
    """Both pipelines tick from their own thread; the guarantee must be a real one."""
    import threading

    ledger, sink = _ledger()
    for guest in range(20):
        ledger.enter(f"guest-{guest}", None, _at(3, 22))
    boundary = DayBoundary(ledger)
    boundary.check(_at(3, 23))

    results, barrier = [], threading.Barrier(2)

    def ask() -> None:
        barrier.wait()
        results.append(boundary.check(_at(4, 5)))

    threads = [threading.Thread(target=ask) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    closed = [names for names in results if names]
    assert len(closed) == 1 and len(closed[0]) == 20
    assert len([e for e in sink.events if e.named_by == RESET]) == 20
