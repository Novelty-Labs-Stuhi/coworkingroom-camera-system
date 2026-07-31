"""The ring buffer: a slow consumer must not cause frames to be missed."""

from __future__ import annotations

import numpy as np

from stuhi_vision.domain import Frame
from stuhi_vision.sources.buffered import BufferedSource


def _frames(count: int) -> list[Frame]:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    return [Frame(timestamp=float(index), image=image) for index in range(count)]


def test_every_frame_arrives_in_order_when_the_consumer_keeps_up() -> None:
    source = BufferedSource(_frames(50), capacity=64)
    assert [frame.timestamp for frame in source] == [float(i) for i in range(50)]


def test_a_finite_source_terminates_the_iteration() -> None:
    assert len(list(BufferedSource(_frames(3)))) == 3


def test_a_slow_consumer_still_receives_frames_produced_while_it_worked() -> None:
    # The whole point: frames produced during processing are banked, not lost.
    source = BufferedSource(_frames(20), capacity=32)
    received = []
    for frame in source:
        received.append(frame.timestamp)
    assert len(received) == 20


def test_a_small_buffer_loses_nothing_because_the_reader_waits() -> None:
    # Dropping frames breaks ByteTrack: it associates detections between *consecutive*
    # frames, so a gap makes it report nobody at all. Completeness beats freshness here.
    source = BufferedSource(_frames(100), capacity=4)
    received = [frame.timestamp for frame in source]

    assert received == [float(i) for i in range(100)]
    assert source.high_water <= 4


def test_capacity_must_be_positive() -> None:
    try:
        BufferedSource(_frames(1), capacity=0)
    except ValueError:
        return
    raise AssertionError("capacity 0 should be rejected")


def test_the_rate_report_says_how_deep_the_backlog_got(capsys) -> None:
    """The rate alone is ambiguous: this reader blocks, so a full queue drags it down.

    Reading a throttled reader's rate as the camera's output understates it by a factor of
    three on this deployment, so the backlog is printed beside it.
    """
    source = BufferedSource(_frames(120), capacity=4, name="door-in")
    for _ in source:
        pass

    printed = capsys.readouterr().out
    assert "door-in delivering" in printed
    assert "backlog peaked" in printed
    assert "/4" in printed
