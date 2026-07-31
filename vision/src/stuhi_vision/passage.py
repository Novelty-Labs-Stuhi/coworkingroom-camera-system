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
from dataclasses import dataclass, replace

from .domain import Crossing, Direction, Frame, TrackedPerson
from .occlusion import Coverage
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
# Asked once per frame for the coverage episode that finished on it, if any. The pixels are
# run by whoever decides when the detector wakes (:class:`~.attention.Attention`), so that they
# are read exactly once a frame and the same reading drives both decisions.
Episodes = Callable[[], Coverage | None]


class PassageMonitor:
    """Turns covered-box episodes into crossings, with the tracker as corroboration.

    Deliberately the same shape as :class:`~.threshold.ThresholdMonitor` -- ``update(people,
    frame)`` returning crossings -- so the pipeline is unchanged by which rule a camera uses.
    """

    def __init__(
        self,
        config: ThresholdConfig,
        episodes: Episodes,
        watcher: Watcher | None = None,
    ) -> None:
        self._config = config
        self._episodes = episodes
        self._watcher = watcher
        # Everybody the detector saw while it was awake for the episode now running:
        # id -> (frames actually over the box, frames merely near the door).
        self._standing: dict[int, int] = {}
        self._nearby: dict[int, int] = {}

    def use_zone(self, zone: tuple[float, float, float, float]) -> None:
        """Judge against a different box from now on, without a restart.

        A zone is redrawn because the camera moved, which means the count is wrong *now* --
        waiting for a restart to apply it is most of the way to not being able to redraw it.
        """
        self._config = replace(self._config, zone=zone)

    def update(self, people: Iterable[TrackedPerson], frame: Frame) -> list[Crossing]:
        """Feed one frame's tracked people; return a crossing when an episode just ended."""
        height, width = frame.image.shape[:2]
        self._note(people, width, height)

        episode = self._episodes()
        if episode is None:
            return []

        passage = self._judge(episode)
        self._standing = {}
        self._nearby = {}
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
        """Count the frames each detected person spent over the box, and near the door.

        Both, because a drawn box can be a narrow strip -- the live one is a twelfth of the
        frame -- and a person overlaps a strip that narrow for only a frame or two. Requiring
        the overlap to be caught meant real passages went uncounted as "nobody detected" while
        the pixels showed the doorframe plainly covered.
        """
        for person in people:
            shape = relative(person.box, width, height)
            if shape.height < self._config.min_height:
                continue  # too small to be at this door; somebody across the room
            self._nearby[person.track_id] = self._nearby.get(person.track_id, 0) + 1
            if overlaps(shape, self._config.zone):
                self._standing[person.track_id] = self._standing.get(person.track_id, 0) + 1

    def _judge(self, episode: Coverage) -> Passage:
        return Passage(
            coverage=episode, person=self._who(), direction=self._direction(episode)
        )

    def _who(self) -> int | None:
        """Whose passage this was: over the box for preference, else close enough to the door.

        The fallback still requires a *detected person*, standing tall enough in frame to be at
        this door -- so a swinging door with nobody about is still never counted. What it drops
        is the demand that the one frame where they overlap a narrow strip was also a frame the
        detector managed.
        """
        for seen in (self._standing, self._nearby):
            if seen:
                return max(seen, key=seen.get)
        return None

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
