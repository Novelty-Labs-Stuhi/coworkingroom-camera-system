"""A passage: the box was covered, the slices lit in an order, and it was a person.

Each signal is used where it is strongest, and only there:

* **Pixel change per slice** (:mod:`.occlusion`) says something covered the doorframe, and the
  order the slices lit says which way it went. This is the only signal that gets *stronger* as
  somebody comes closer, which is where a person detector gets weaker -- a body filling the
  frame is out of distribution for one. Being an order of events, it also survives the person's
  outline swelling as they approach, which defeated every measurement of shape tried before it.
* **The tracker** says the covering thing was a *person*, and which one, so an identity can be
  attached. A swinging door, an arm reaching through and a bag set down in the doorway cover
  the box identically; only a detection tells them apart.

Three outcomes, and the middle one is the point of the slices:

===============================  =========================================================
slices lit in an order, person   a passage, counted in the direction of the order
slices lit with no order         somebody on the doorframe who did not go through: reported
covered with nobody detected     reported, never counted -- the door, or a detector failure
===============================  =========================================================

A covered box with no person on it is deliberately not counted. Pixels alone would count the
door swinging; detections alone would miss whoever stands too close to be seen as a body.
Requiring both is what makes the pair trustworthy, and the reports make each refusal visible
rather than leaving a silent gap in the count.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .domain import Crossing, Direction, Frame, TrackedPerson
from .occlusion import Coverage, Occlusion
from .threshold import ThresholdConfig, overlaps, relative


@dataclass(frozen=True, slots=True)
class Passage:
    """What was made of one episode of the box being covered."""

    coverage: Coverage
    person: int | None                 # the track id on the box, if a person was detected
    direction: Direction | None        # None: covered, but not a passage

    @property
    def readable(self) -> str:
        who = f"track {self.person}" if self.person is not None else "nobody detected"
        verdict = self.direction.value if self.direction else "no passage"
        return f"{self.coverage.readable}, {who} -> {verdict}"


Watcher = Callable[[Passage], None]


class PassageMonitor:
    """Turns covered-box episodes into crossings, with the tracker as corroboration.

    Deliberately the same shape as :class:`~.threshold.ThresholdMonitor` -- ``update(people,
    frame)`` returning crossings -- so the pipeline is unchanged by which rule a camera uses.
    """

    def __init__(
        self,
        config: ThresholdConfig,
        occlusion: Occlusion,
        watcher: Watcher | None = None,
    ) -> None:
        self._config = config
        self._occlusion = occlusion
        self._watcher = watcher
        # People seen standing on the box during the episode now running: id -> frames there.
        self._standing: dict[int, int] = {}

    def update(self, people: Iterable[TrackedPerson], frame: Frame) -> list[Crossing]:
        """Feed one frame's tracked people; return a crossing when an episode just ended."""
        height, width = frame.image.shape[:2]
        self._note(people, width, height)

        episode = self._occlusion.update(frame.image)
        if episode is None:
            return []

        passage = self._judge(episode)
        self._standing = {}
        if self._watcher is not None:
            self._watcher(passage)
        if passage.direction is None or passage.person is None:
            return []
        return [
            Crossing(
                track_id=passage.person,
                direction=passage.direction,
                timestamp=frame.timestamp,
            )
        ]

    def _note(self, people: Iterable[TrackedPerson], width: int, height: int) -> None:
        """Count frames each detected person spent on the box."""
        for person in people:
            shape = relative(person.box, width, height)
            if shape.height < self._config.min_height:
                continue  # too small to be at this door; somebody across the room
            if overlaps(shape, self._config.zone):
                self._standing[person.track_id] = self._standing.get(person.track_id, 0) + 1

    def _judge(self, episode: Coverage) -> Passage:
        person = max(self._standing, key=self._standing.get, default=None)
        return Passage(coverage=episode, person=person, direction=self._direction(episode))

    def _direction(self, episode: Coverage) -> Direction | None:
        """The order the slices lit, read against which side of the frame the doorway is on.

        A positive lag means the low-numbered slices lit first, so the covering travelled
        towards the high-numbered side -- rightwards, or downwards. No lag means no order, and
        no order means nobody went through.
        """
        if not episode.swept:
            return None
        travelled_towards_low = episode.lag > 0 if _reversed(self._config.edge) else episode.lag < 0
        passing = self._config.passing_means
        return passing if travelled_towards_low else _opposite(passing)


def _reversed(edge: str) -> bool:
    """Whether the configured edge is the high-numbered side of the slices.

    Slices are numbered left to right (or top to bottom), so an edge of "right" or "bottom"
    means travelling towards it shows up as the *later* slices lighting first being wrong way
    round -- hence the flip, in one place, named.
    """
    return edge in ("right", "bottom")


def _opposite(direction: Direction) -> Direction:
    return Direction.OUT if direction is Direction.IN else Direction.IN
