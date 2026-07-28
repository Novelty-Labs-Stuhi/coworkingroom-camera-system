"""A camera announces only the direction it sees faces in."""

from __future__ import annotations

from stuhi_vision.assembly import _directional
from stuhi_vision.domain import Direction, Outcome, Sighting


class FakePublisher:
    def __init__(self) -> None:
        self.held: list[Sighting] = []

    def hold(self, sighting: Sighting) -> None:
        self.held.append(sighting)


def _sighting(direction: Direction) -> Sighting:
    return Sighting(
        timestamp=0.0,
        direction=direction,
        name=None,
        score=0.0,
        outcome=Outcome.UNKNOWN,
    )


def test_both_passes_everything_straight_through() -> None:
    publisher = FakePublisher()
    hold = _directional("both", publisher, lambda s: None)

    hold(_sighting(Direction.IN))
    hold(_sighting(Direction.OUT))

    assert len(publisher.held) == 2


def test_only_the_reported_direction_is_filmed() -> None:
    publisher = FakePublisher()
    logged: list[Sighting] = []
    hold = _directional("in", publisher, logged.append)

    hold(_sighting(Direction.IN))
    hold(_sighting(Direction.OUT))

    # The clip is never even opened for the wrong direction: no encoding, no message.
    assert [s.direction for s in publisher.held] == [Direction.IN]
    # But it is still counted, so occupancy stays correct.
    assert [s.direction for s in logged] == [Direction.OUT]


def test_the_out_camera_is_the_mirror_image() -> None:
    publisher = FakePublisher()
    logged: list[Sighting] = []
    hold = _directional("out", publisher, logged.append)

    hold(_sighting(Direction.IN))
    hold(_sighting(Direction.OUT))

    assert [s.direction for s in publisher.held] == [Direction.OUT]
    assert [s.direction for s in logged] == [Direction.IN]
