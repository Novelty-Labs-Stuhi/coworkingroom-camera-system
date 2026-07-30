"""Detect a passage through the doorway from where a track begins and ends.

The older approach drew a line across the doorway and watched a foot point cross it. That
failed repeatedly in practice: the foot point sat pinned to the bottom edge of the frame
because people pass close to the camera, the line had to be redrawn from a still image every
time a camera moved, and getting ``inside_side`` backwards silently reversed the count.

This uses a sturdier fact about the scene. **The doorframe is visible, and a person passing
through it occludes it** -- their pixels are in front of the frame, not behind it. Someone
merely moving in the background is seen *through* the opening and never overlaps it. So:

* a track that never overlaps the doorframe zone is background traffic and is ignored;
* a track that does overlap it went through the doorway, and **which way it travelled across
  the zone** is which way it went.

Crossing the box is the passage; the direction of travel is the direction. Nothing depends on
the person reaching the edge of the picture, so a track lost mid-doorway still counts, and a
repositioned camera does not invalidate the rule -- only the drawn box, which is redrawable.

Which of those two means "in" is a property of the camera's position, given as
``passing_means``. On the room-facing camera the same shape of rule applies with the edge
being the bottom of the frame -- people approach the lens and leave downwards.

Nothing here is a hairline: the zone is a broad strip and the edge has a margin, so a
knocked camera degrades gradually instead of silently counting nothing.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from .domain import Box, Crossing, Direction, Frame, TrackedPerson

Edge = Literal["left", "right", "top", "bottom"]

# How a finished track is judged.
#   "edge"     -- did it leave past a frame edge, or arrive from one? Right when the camera
#                 watches people cross its view sideways, as the doorframe camera does.
#   "travel"   -- which way did it move across the zone? Right when the doorframe is visible:
#                 crossing the box *is* the passage, and the direction of travel says which
#                 way. Needs no frame edge, so it survives a camera being repositioned.
#   "approach" -- did it grow or shrink? Right when people walk straight at the lens, where
#                 everyone is already touching the near edge and "at the edge" says nothing.
Discriminator = Literal["edge", "travel", "approach"]


@dataclass(frozen=True, slots=True)
class ThresholdConfig:
    """How one camera recognises a passage.

    ``zone`` is the part of the frame a person must overlap to count -- the visible
    doorframe -- expressed as fractions of the frame so it survives a resolution change.
    ``edge`` is the side of the frame they leave through when passing, and ``margin`` how
    close to it counts as "at" it.
    """

    zone: tuple[float, float, float, float] = (0.0, 0.0, 0.25, 1.0)
    edge: Edge = "left"
    margin: float = 0.12
    passing_means: Direction = Direction.IN
    # Which question to ask of a finished track. Measured on the room-facing camera, every
    # track began *and* ended touching the bottom edge (0.97-1.00 of the frame height),
    # because anyone that close fills the picture downwards -- so the edge test could not
    # tell an arrival from a departure and discarded nearly every real passage. Size change
    # separates them cleanly there: walking at the lens grows, walking away shrinks.
    discriminator: Discriminator = "edge"
    # How much of the frame height a track must gain or lose for "approach" to call it.
    # Measured passes changed by 0.12-0.28; people merely shifting about changed by ~0.09.
    growth_margin: float = 0.12
    # How far a track must travel across the frame, along the axis of ``edge``, for "travel"
    # to call it. Small: somebody passing through a doorway covers most of the frame, while
    # somebody standing in the doorway talking drifts by a few percent.
    travel_margin: float = 0.08
    # A person must be at least this tall in frame to be considered at the door at all.
    # Rejects distant figures that happen to line up with the zone.
    min_height: float = 0.35
    # Frames a track may be missing before it is judged finished. At a few frames per
    # second a person is easily missed for one or two frames mid-stride.
    lost_after: int = 6


@dataclass
class _Track:
    first: Box
    last: Box
    last_seen: int
    touched_zone: bool = False
    tallest: float = 0.0
    frames: int = field(default=1)


class ThresholdMonitor:
    """Turns tracked people into crossings, deciding when each track ends.

    A decision is deliberately deferred until the track is gone: whether someone *passed
    through* is only knowable once they stop being visible, and that is precisely the
    signal -- they left the frame at the door edge.
    """

    def __init__(self, config: ThresholdConfig) -> None:
        self._config = config
        self._tracks: dict[int, _Track] = {}
        self._frame_index = 0

    def update(self, people: Iterable[TrackedPerson], frame: Frame) -> list[Crossing]:
        """Feed one frame's tracked people; return crossings for tracks that just ended."""
        self._frame_index += 1
        height, width = frame.image.shape[:2]

        present = set()
        for person in people:
            present.add(person.track_id)
            self._observe(person, width, height)

        return self._resolve_finished(present, frame)

    def _observe(self, person: TrackedPerson, width: int, height: int) -> None:
        box = person.box
        relative = _relative(box, width, height)
        tall_enough = relative.height >= self._config.min_height

        track = self._tracks.get(person.track_id)
        if track is None:
            self._tracks[person.track_id] = track = _Track(
                first=box, last=box, last_seen=self._frame_index
            )
        else:
            track.last = box
            track.last_seen = self._frame_index
            track.frames += 1

        track.tallest = max(track.tallest, relative.height)
        if tall_enough and _overlaps(relative, self._config.zone):
            track.touched_zone = True

    def _resolve_finished(self, present: set[int], frame: Frame) -> list[Crossing]:
        height, width = frame.image.shape[:2]
        crossings = []
        for track_id in [
            track_id
            for track_id, track in self._tracks.items()
            if track_id not in present
            and self._frame_index - track.last_seen > self._config.lost_after
        ]:
            track = self._tracks.pop(track_id)
            direction = self._decide(track, width, height)
            if direction is not None:
                crossings.append(
                    Crossing(track_id=track_id, direction=direction, timestamp=frame.timestamp)
                )
        return crossings

    def _decide(self, track: _Track, width: int, height: int) -> Direction | None:
        """A finished track: did it pass through the doorway, and which way?"""
        if not track.touched_zone or track.tallest < self._config.min_height:
            return None  # background traffic, or never close enough to be at the door

        first = _relative(track.first, width, height)
        last = _relative(track.last, width, height)
        if self._config.discriminator == "approach":
            return self._by_size(first, last)
        if self._config.discriminator == "travel":
            return self._by_travel(first, last)
        return self._by_edge(first, last)

    def _by_edge(self, first: _Relative, last: _Relative) -> Direction | None:
        """Left past the edge, or arrived from it?"""
        left_at_edge = _at_edge(last, self._config.edge, self._config.margin)
        arrived_at_edge = _at_edge(first, self._config.edge, self._config.margin)

        if left_at_edge and not arrived_at_edge:
            return self._config.passing_means
        if arrived_at_edge and not left_at_edge:
            return _opposite(self._config.passing_means)
        # Both or neither: someone who stepped in and back out again, or who was only ever
        # at the edge. Not a passage, and guessing would put noise into the count.
        return None

    def _by_travel(self, first: _Relative, last: _Relative) -> Direction | None:
        """Which way did they move across the box?

        Crossing the doorframe *is* the passage -- the zone gate has already established
        that -- so all that remains is which way they were going. Movement is measured along
        the axis the doorway runs across, given by ``edge``: towards that side means the
        direction ``passing_means``, away from it the opposite.

        This is stronger than asking where the track ended, which was the earlier rule. A
        person does not have to reach the edge of the picture, or be visible when they get
        there; a track that is lost mid-doorway still travelled in a direction. It also has
        no dead zone: the edge rule refused to judge anyone who both arrived at the edge and
        left by it, which is what a person filling the near side of the frame looks like.
        """
        towards_lower = self._config.edge in ("left", "top")
        if self._config.edge in ("left", "right"):
            moved = _centre(last.left, last.right) - _centre(first.left, first.right)
        else:
            moved = _centre(last.top, last.bottom) - _centre(first.top, first.bottom)

        if abs(moved) < self._config.travel_margin:
            return None  # stood in the doorway rather than went through it
        towards_edge = moved < 0 if towards_lower else moved > 0
        return self._config.passing_means if towards_edge else _opposite(self._config.passing_means)

    def _by_size(self, first: _Relative, last: _Relative) -> Direction | None:
        """Grew towards the lens, or shrank away from it?

        For a camera people walk straight at, this is the only usable signal: they are
        touching the near edge the whole time, so where they start and end tells nothing.
        """
        change = last.height - first.height
        if change >= self._config.growth_margin:
            return self._config.passing_means  # came at the lens: through the door
        if change <= -self._config.growth_margin:
            return _opposite(self._config.passing_means)
        return None  # barely changed size: milling about rather than passing


@dataclass(frozen=True, slots=True)
class _Relative:
    """A box in fractions of the frame, so thresholds are resolution independent."""

    left: float
    top: float
    right: float
    bottom: float

    @property
    def height(self) -> float:
        return self.bottom - self.top


def _relative(box: Box, width: int, height: int) -> _Relative:
    return _Relative(
        left=box.x1 / width,
        top=box.y1 / height,
        right=box.x2 / width,
        bottom=box.y2 / height,
    )


def _overlaps(box: _Relative, zone: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = zone
    return box.right > x1 and box.left < x2 and box.bottom > y1 and box.top < y2


def _at_edge(box: _Relative, edge: Edge, margin: float) -> bool:
    if edge == "left":
        return box.left <= margin
    if edge == "right":
        return box.right >= 1.0 - margin
    if edge == "top":
        return box.top <= margin
    return box.bottom >= 1.0 - margin


def _centre(low: float, high: float) -> float:
    return (low + high) / 2


def _opposite(direction: Direction) -> Direction:
    return Direction.OUT if direction is Direction.IN else Direction.IN
