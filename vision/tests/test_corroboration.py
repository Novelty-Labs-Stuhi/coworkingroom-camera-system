"""The room camera's second opinion on a passage the doorway camera counted."""

from __future__ import annotations

from stuhi_vision.corroboration import Corroboration


def test_an_entry_is_confirmed_by_somebody_appearing_in_the_room() -> None:
    """Which is what walking in looks like from the camera inside."""
    room = Corroboration()
    room.appeared(at=100.0)

    seen = room.confirms_entry(98.5)

    assert seen is not None
    assert seen.at == 100.0


def test_an_exit_is_confirmed_by_somebody_walking_at_the_lens_and_out_of_frame() -> None:
    room = Corroboration()
    room.left_frame(at=95.0, grew=0.21)

    seen = room.confirms_exit(100.0)

    assert seen is not None
    assert seen.grew == 0.21


def test_an_entry_is_not_confirmed_by_somebody_leaving() -> None:
    """The two halves are different observations; either standing for the other proves nothing."""
    room = Corroboration()
    room.left_frame(at=100.0, grew=0.3)

    assert room.confirms_entry(100.0) is None


def test_an_exit_is_not_confirmed_by_somebody_appearing() -> None:
    room = Corroboration()
    room.appeared(at=100.0)

    assert room.confirms_exit(100.0) is None


def test_a_confirmation_is_consumed_so_it_cannot_serve_two_crossings() -> None:
    """Otherwise one person walking in would confirm every entry imagined for the next 15 s."""
    room = Corroboration()
    room.appeared(at=100.0)

    assert room.confirms_entry(100.0) is not None
    assert room.confirms_entry(100.5) is None


def test_two_people_arriving_need_two_appearances_between_them() -> None:
    room = Corroboration()
    room.appeared(at=100.0)
    room.appeared(at=103.0)

    assert room.confirms_entry(99.5) is not None
    assert room.confirms_entry(102.5) is not None
    assert room.confirms_entry(104.0) is None


def test_the_nearest_observation_is_the_one_claimed() -> None:
    """Two people seconds apart must not have their evidence swapped."""
    room = Corroboration()
    room.appeared(at=100.0)
    room.appeared(at=110.0)

    assert room.confirms_entry(109.0).at == 110.0
    assert room.confirms_entry(101.0).at == 100.0


def test_an_entry_looks_forward_because_they_cross_the_door_before_reaching_the_room() -> None:
    room = Corroboration()
    room.appeared(at=110.0)          # ten seconds after the crossing

    assert room.confirms_entry(100.0) is not None


def test_an_exit_looks_back_because_they_cross_the_room_before_reaching_the_door() -> None:
    room = Corroboration()
    room.left_frame(at=90.0, grew=0.2)   # ten seconds before the crossing

    assert room.confirms_exit(100.0) is not None


def test_an_observation_too_far_away_is_not_about_this_crossing() -> None:
    room = Corroboration()
    room.appeared(at=200.0)

    assert room.confirms_entry(100.0) is None


def test_nothing_seen_at_all_is_unconfirmed_rather_than_an_error() -> None:
    """A passage without a second witness is recorded, not refused: refusing an exit nobody
    corroborated would leave that person in the room for ever."""
    room = Corroboration()

    assert room.confirms_entry(100.0) is None
    assert room.confirms_exit(100.0) is None


def test_stale_observations_are_forgotten_so_the_lists_cannot_grow_all_day() -> None:
    room = Corroboration()
    for moment in range(0, 400, 2):
        room.appeared(at=float(moment))
    room.confirms_entry(400.0)

    waiting, _ = room.waiting
    assert waiting < 20, "only the recent past can be about the crossing being asked of"
