"""Detect a person crossing the doorway threshold, and which way.

The threshold is a line segment drawn across the door in the camera image. We watch
each tracked person's foot point; when it moves from one side of the line to the
other (within the segment's span), that is a crossing. The side it ends up on tells
us whether they came *in* or went *out* -- independent of any face recognition, so
the count is always right even when identity is uncertain.
"""

from __future__ import annotations

from collections.abc import Iterable

from .config import DoorwayConfig, Point
from .domain import Crossing, Direction, Frame, TrackedPerson


def _side(a: Point, b: Point, p: Point) -> float:
    """Signed side of point ``p`` relative to the directed line ``a -> b``.

    Positive and negative denote the two half-planes; zero is on the line.
    """
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _within_segment(a: Point, b: Point, p: Point) -> bool:
    """True when ``p`` projects onto the segment ``a..b`` (not its infinite extension)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return False
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length_sq
    return 0.0 <= t <= 1.0


class DoorwayMonitor:
    """Stateful line-crossing detector, one instance per doorway."""

    def __init__(self, config: DoorwayConfig) -> None:
        self._a = config.line_a
        self._b = config.line_b
        self._inside_positive = self._resolve_inside_sign(config)
        self._last_side: dict[int, float] = {}

    def _resolve_inside_sign(self, config: DoorwayConfig) -> bool:
        """Map the human 'left'/'right' choice to the sign of the inside half-plane."""
        # A reference point one unit to the "left" of the line direction has
        # positive side by construction of the cross product.
        return config.inside_side == "left"

    def _direction(self, previous: float, current: float) -> Direction | None:
        if previous == 0 or current == 0 or (previous > 0) == (current > 0):
            return None  # no side change -> no crossing
        entered_positive_side = current > 0
        going_inside = entered_positive_side == self._inside_positive
        return Direction.IN if going_inside else Direction.OUT

    def update(self, people: Iterable[TrackedPerson], frame: Frame) -> list[Crossing]:
        """Feed one frame's tracked people; return any crossings that just occurred."""
        crossings: list[Crossing] = []
        seen: set[int] = set()
        for person in people:
            seen.add(person.track_id)
            foot = person.box.foot
            side = _side(self._a, self._b, foot)
            previous = self._last_side.get(person.track_id)
            self._last_side[person.track_id] = side
            if previous is None or not _within_segment(self._a, self._b, foot):
                continue
            direction = self._direction(previous, side)
            if direction is not None:
                crossings.append(
                    Crossing(
                        track_id=person.track_id,
                        direction=direction,
                        timestamp=frame.timestamp,
                    )
                )
        self._forget_absent(seen)
        return crossings

    def _forget_absent(self, seen: set[int]) -> None:
        """Drop tracks no longer present so the state dict cannot grow forever."""
        for track_id in [t for t in self._last_side if t not in seen]:
            del self._last_side[track_id]
