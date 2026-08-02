"""A stamp that stops advancing is how a wedged pipeline is told from a healthy one."""

from __future__ import annotations

import pytest

from stuhi_vision.heartbeat import Heartbeat, stale


class _Clock:
    def __init__(self, at: float = 1000.0) -> None:
        self.at = at

    def __call__(self) -> float:
        return self.at


def test_the_first_beat_writes_the_current_second(tmp_path) -> None:
    clock = _Clock()
    beat = Heartbeat(tmp_path / "door-in", now=clock)

    beat.beat()

    assert (tmp_path / "door-in").read_text().strip() == "1000"


def test_beats_within_the_interval_do_not_rewrite_the_file(tmp_path) -> None:
    clock = _Clock()
    beat = Heartbeat(tmp_path / "cam", interval=1.0, now=clock)
    beat.beat()

    clock.at = 1000.5
    beat.beat()

    assert (tmp_path / "cam").read_text().strip() == "1000"


def test_the_stamp_advances_once_the_interval_has_passed(tmp_path) -> None:
    clock = _Clock()
    beat = Heartbeat(tmp_path / "cam", interval=1.0, now=clock)
    beat.beat()

    clock.at = 1002.0
    beat.beat()

    assert (tmp_path / "cam").read_text().strip() == "1002"


def test_the_directory_is_created_if_it_does_not_exist(tmp_path) -> None:
    beat = Heartbeat(tmp_path / "nested" / "deeper" / "cam", now=_Clock())

    beat.beat()

    assert (tmp_path / "nested" / "deeper" / "cam").exists()


def test_a_write_failure_never_reaches_the_pipeline(tmp_path) -> None:
    """The heartbeat exists to protect the pipeline; it must not be able to stop it."""
    blocked = tmp_path / "afile"
    blocked.write_text("not a directory")
    beat = Heartbeat(blocked / "cam", now=_Clock())

    beat.beat()  # must not raise


def test_a_write_failure_is_logged_once_not_every_second(tmp_path, caplog) -> None:
    blocked = tmp_path / "afile"
    blocked.write_text("not a directory")
    clock = _Clock()
    beat = Heartbeat(blocked / "cam", interval=1.0, now=clock)

    with caplog.at_level("WARNING"):
        for second in range(5):
            clock.at = 1000.0 + second * 2
            beat.beat()

    assert len([r for r in caplog.records if "heartbeat" in r.message]) == 1


# --- reading the stamp back ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("written", "now", "expected"),
    [
        (1000, 1005, False),   # five seconds old, well inside the threshold
        (1000, 1100, True),    # a hundred seconds old
        (1000, 1060, False),   # exactly at the threshold is not yet stale
    ],
)
def test_stale_compares_the_stamp_against_the_threshold(
    tmp_path, written, now, expected
) -> None:
    path = tmp_path / "cam"
    path.write_text(f"{written}\n")

    assert stale(path, older_than=60, now=now) is expected


def test_a_missing_stamp_is_stale(tmp_path) -> None:
    """Never having handled a frame is the condition being watched for, not an exemption."""
    assert stale(tmp_path / "never-written", older_than=60, now=1000) is True


def test_an_unreadable_stamp_is_stale(tmp_path) -> None:
    path = tmp_path / "cam"
    path.write_text("half-written garbage")

    assert stale(path, older_than=60, now=1000) is True
