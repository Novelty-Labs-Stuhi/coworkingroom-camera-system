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
    assert source.dropped == 0


def test_a_finite_source_terminates_the_iteration() -> None:
    assert len(list(BufferedSource(_frames(3)))) == 3


def test_a_slow_consumer_still_receives_frames_produced_while_it_worked() -> None:
    # The whole point: frames produced during processing are banked, not lost.
    source = BufferedSource(_frames(20), capacity=32)
    received = []
    for frame in source:
        received.append(frame.timestamp)
    assert len(received) == 20


def test_overflow_drops_the_oldest_and_keeps_going() -> None:
    # Capacity smaller than the source, consumed only after production finishes.
    source = BufferedSource(_frames(100), capacity=4)
    received = [frame.timestamp for frame in source]

    assert len(received) <= 100
    assert received == sorted(received)  # order is never scrambled
    assert source.dropped == 100 - len(received)


def test_capacity_must_be_positive() -> None:
    try:
        BufferedSource(_frames(1), capacity=0)
    except ValueError:
        return
    raise AssertionError("capacity 0 should be rejected")
