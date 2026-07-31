"""A passage: the box was covered, something crossed it, and it was a person.

Three signals, each used where it is strongest:

* **Pixels over the box** (:mod:`.occlusion`) say the doorframe was covered and for how long.
  This is the one signal that improves as somebody comes closer, which is exactly where a
  person detector degrades.
* **Which way the covering travelled** says in or out. Read from pixels when the episode is
  long enough to read, and from the tracked person's own travel when it is not -- on recorded
  footage the box was covered for only 2-4 frames while the track ran 33-39, so the tracker
  has an order of magnitude more evidence for a direction and is the better witness to it.
* **The tracker** establishes that the covering thing was a *person* and which one. A door
  swinging, an arm reaching through, or a bag set down in the doorway all cover the box
  identically; only a detection tells them apart, and only a track carries an identity.

A covered box with no person on it is reported and not counted. That is the whole point of
requiring both: pixels alone would count the door, detections alone would miss whoever is too
close to be recognised as a body.
"""

from __future__ import annotations

from collections.abc import Iterable

from .domain import Crossing, Direction, Frame, TrackedPerson
from .occlusion import Coverage, Occlusion
from .threshold import Report, ThresholdConfig, Touch, relative, travel_of


class PassageMonitor:
    """Turns covered-box episodes into crossings, with the tracker as corroboration."""

    def __init__(
        self,
        config: ThresholdConfig,
        occlusion: Occlusion,
        report: Report | None = None,
    ) -> None:
        self._config = config
        self._occlusion = occlusion
        self._report = report
        # Who was on the box during the episode running now: track id -> first and last box.
        self._present: dict[int, tuple] = {}

    def update(self, people: Iterable[TrackedPerson], frame: Frame) -> list[Crossing]:
        """Feed one frame's tracked people; return a crossing when an episode just ended."""
        height, width = frame.image.shape[:2]
        self._remember(people, width, height)
        episode = self._occlusion.update(frame.image)
        if episode is None:
            return []

        crossing = self._judge(episode, frame)
        self._present = {}
        return [crossing] if crossing is not None else []

    def _remember(self, people: Iterable[TrackedPerson], width: int, height: int) -> None:
        """Note every person standing on the box, and how their box moved while there."""
        for person in people:
            box = relative(person.box, width, height)
            if box.height < self._config.min_height or not _overlaps(box, self._config.zone):
                continue
            first, _ = self._present.get(person.track_id, (box, box))
            self._present[person.track_id] = (first, box)

    def _judge(self, episode: Coverage, frame: Frame) -> Crossing | None:
        """Was that a person, and which way did they go?"""
        candidate = self._who()
        direction = self._direction(episode, candidate)
        self._tell(episode, candidate, direction)
        if candidate is None or direction is None:
            return None
        return Crossing(track_id=candidate, direction=direction, timestamp=frame.timestamp)

    def _who(self) -> int | None:
        """The person on the box longest -- the one the episode is about."""
        if not self._present:
            return None
        return max(self._present, key=lambda track_id: _span(self._present[track_id]))

    def _direction(self, episode: Coverage, candidate: int | None) -> Direction | None:
        """Pixels first, the track second. Neither is guessed at."""
        towards = _towards(self._config.edge)
        if abs(episode.travelled) >= self._config.travel_margin:
            return self._read(episode.travelled, towards)
        if candidate is None:
            return None
        first, last = self._present[candidate]
        moved = travel_of(first, last, self._config.edge)
        if abs(moved) >= self._config.travel_margin:
            return self._read(moved, towards)
        return None  # covered the box without going anywhere: stood in the doorway

    def _read(self, moved: float, towards_lower: bool) -> Direction:
        went_towards_edge = moved < 0 if towards_lower else moved > 0
        passing = self._config.passing_means
        return passing if went_towards_edge else _opposite(passing)

    def _tell(self, episode: Coverage, candidate: int | None, direction: Direction | None) -> None:
        if self._report is None:
            return
        if candidate is None:
            # Worth saying out loud: something covered the doorframe and no person was on it.
            # Repeated, that is the door swinging, or the detector failing in the dark.
            self._report(
                Touch(frames=episode.frames, tallest=0.0, travelled=episode.travelled,
                      grew=0.0, direction=None, note=f"{episode.readable}, but nobody on it")
            )
            return
        first, last = self._present[candidate]
        self._report(
            Touch(
                frames=episode.frames,
                tallest=max(first.height, last.height),
                travelled=episode.travelled,
                grew=last.height - first.height,
                direction=direction,
                note=episode.readable,
            )
        )


def _span(extent: tuple) -> float:
    first, last = extent
    return abs(travel_of(first, last, "left")) + 0.001


def _overlaps(box, zone: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = zone
    return box.right > x1 and box.left < x2 and box.bottom > y1 and box.top < y2


def _towards(edge: str) -> bool:
    return edge in ("left", "top")


def _opposite(direction: Direction) -> Direction:
    return Direction.OUT if direction is Direction.IN else Direction.IN
