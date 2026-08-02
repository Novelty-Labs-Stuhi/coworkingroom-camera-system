"""An identity the system named itself is the one piece of work worth interrupting for."""

from __future__ import annotations

from stuhi_vision.provisional import Provisional


def test_a_name_the_system_invented_is_held(tmp_path) -> None:
    provisional = Provisional(tmp_path / "provisional")

    provisional.add("guest-0802-161045")

    assert provisional.holds("guest-0802-161045")
    assert not provisional.holds("arsenii")


def test_a_human_naming_it_releases_it(tmp_path) -> None:
    provisional = Provisional(tmp_path / "provisional")
    provisional.add("guest-0802-161045")

    provisional.claimed("guest-0802-161045")

    assert not provisional.holds("guest-0802-161045")


def test_it_survives_a_restart(tmp_path) -> None:
    """The whole point is that a name's origin is remembered, not re-derived."""
    path = tmp_path / "provisional"
    Provisional(path).add("guest-a")

    assert Provisional(path).holds("guest-a")


def test_releasing_a_name_it_never_held_is_harmless(tmp_path) -> None:
    provisional = Provisional(tmp_path / "provisional")

    provisional.claimed("somebody-else")

    assert provisional.names() == set()


def test_adding_twice_writes_one_entry(tmp_path) -> None:
    path = tmp_path / "provisional"
    provisional = Provisional(path)

    provisional.add("guest-a")
    provisional.add("guest-a")

    assert path.read_text().split() == ["guest-a"]


def test_a_name_that_is_a_real_person_is_never_held_by_accident(tmp_path) -> None:
    """Recorded, not pattern-matched: somebody actually called this is not provisional."""
    provisional = Provisional(tmp_path / "provisional")

    assert not provisional.holds("guest services")


def test_names_narrows_to_what_the_gallery_still_has(tmp_path) -> None:
    provisional = Provisional(tmp_path / "provisional")
    provisional.add("guest-a")
    provisional.add("guest-b")

    assert provisional.names(known=["guest-a", "arsenii"]) == {"guest-a"}


def test_narrowing_forgets_the_vanished_name_permanently(tmp_path) -> None:
    """An identity merged away elsewhere must not linger as work that cannot be done."""
    path = tmp_path / "provisional"
    provisional = Provisional(path)
    provisional.add("guest-a")
    provisional.add("guest-b")

    provisional.names(known=["guest-a"])

    assert Provisional(path).names() == {"guest-a"}


def test_a_missing_file_reads_as_nothing_held(tmp_path) -> None:
    assert Provisional(tmp_path / "never-written").names() == set()


def test_an_unwritable_path_does_not_raise(tmp_path) -> None:
    """Losing this degrades the page; it must not be able to stop a crossing being recorded."""
    blocked = tmp_path / "afile"
    blocked.write_text("not a directory")

    Provisional(blocked / "provisional").add("guest-a")  # must not raise
