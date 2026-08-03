"""Direction from *when* the doorframe was covered relative to when the person was seen.

Every rule tried before this one asks what the doorframe was doing at some moment the track
also occupies -- the order the slices lit, or whether the box was covered on the track's first
and last frame. All of them need the detector to be looking at the same instant the pixels
are, and on this doorway it is not: measured, it acquires somebody **two to four frames after
the coverage has already ended**.

    capture_lit:  coverage 107-110  ->  track begins 114
    dual_cam1:    coverage 337-348  ->  track begins 350

So an arrival's track never starts while the frame is covered, and any rule keyed on that
sees only departures. Across every frame of recorded footage there are three
``clear->covered`` tracks and **zero** ``covered->clear`` -- a rule that would have counted in
one direction only, which is the fault it was meant to cure.

This asks a question that does not need them to coincide: **did the covering happen before
this person appeared, or after they were last seen?**

* covered *before* they appeared -> they came through the doorway and then became visible;
* covered *after* they were last seen -> they were in view, reached the doorway, and went;
* covered before *and* after -> they came in, stayed a while, and left again. That is two
  passages, and it is named as such rather than dismissed as noise -- but no direction is
  emitted for it yet. Reporting both would mean two crossings carrying one track id, and the
  Doorkeeper takes that track's session on the first and drops the second as "no session". So
  the case is identified and labelled here, and turning it into two events waits on that. It
  is the one track of six in the recorded footage still unresolved;
* no episode near them at all -> they never went near the door.

Nothing here measures a distance or reads an order, so there is no threshold on the signal
itself. The one tunable is how far from a track an episode may sit and still be its own --
and it is a *window in time*, not a margin on a measurement, so a person's outline swelling
as they approach the lens cannot corrupt it.

Deliberately reports rather than commits, for now. It resolves five of the six tracks in the
recorded footage where the live rule resolves three, and it produces both directions where the
live rule produces one -- but six tracks is six tracks, and the ground truth for them was read
by eye from the same ordering this rule uses, so the agreement is not independent evidence.
Running it beside the live rule costs nothing and settles that in a day.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

from .domain import Direction
from .threshold import ThresholdConfig, overlaps, relative

_log = logging.getLogger(__name__)

# How far either side of a track an episode may sit and still be treated as that person's.
# Measured gaps between a coverage episode ending and its person being acquired were 2 and 4
# frames; twelve is generous enough to absorb a worse dropout without reaching to a stranger.
WINDOW = 12

# Frames a track may be missing before it is judged finished. Matches the threshold rule's
# default, for the same reason: at a few frames a second somebody is easily lost mid-stride.
LOST_AFTER = 6


@dataclass(frozen=True, slots=True)
class Episode:
    """One finished stretch of the doorframe being covered, in frame numbers."""

    first: int
    last: int

    @property
    def middle(self) -> float:
        return (self.first + self.last) / 2


@dataclass(frozen=True, slots=True)
class Verdict:
    """What was made of one finished track."""

    track_id: int
    direction: Direction | None
    because: str

    @property
    def readable(self) -> str:
        way = self.direction.value if self.direction else "nothing"
        return f"track {self.track_id} -> {way} ({self.because})"


@dataclass
class _Track:
    first: int
    last: int
    tallest: float = 0.0
    reached_zone: bool = False


class OrderingRule:
    """Judges a finished track by where the doorframe's coverage sits around it."""

    def __init__(
        self,
        config: ThresholdConfig,
        window: int = WINDOW,
        lost_after: int = LOST_AFTER,
    ) -> None:
        self._config = config
        self._window = window
        self._lost_after = lost_after
        self._tracks: dict[int, _Track] = {}
        # Only the recent past matters, and an unbounded list of episodes on a camera that
        # runs for weeks is a slow leak rather than a feature.
        self._episodes: deque[Episode] = deque(maxlen=64)
        self._frame = 0

    @property
    def frame(self) -> int:
        """How many frames have been seen. Needed to place a finished episode in time.

        Coverage is only reported once it has *ended*, so its span runs back from the frame
        before the one now being handed over -- and only the rule knows which number that is.
        """
        return self._frame

    def span_of(self, covered_frames: int) -> Episode:
        """Where an episode of this length sits, given it ended on the previous frame."""
        return Episode(self._frame + 1 - covered_frames, self._frame)

    def observe(self, people, width: int, height: int, episode: Episode | None) -> list[Verdict]:
        """Feed one frame. Returns a verdict for each track that has just finished."""
        self._frame += 1
        if episode is not None:
            self._episodes.append(episode)

        present = set()
        for person in people:
            present.add(person.track_id)
            self._note(person, width, height)
        return self._finished(present)

    def _note(self, person, width: int, height: int) -> None:
        shape = relative(person.box, width, height)
        track = self._tracks.get(person.track_id)
        if track is None:
            self._tracks[person.track_id] = track = _Track(self._frame, self._frame)
        track.last = self._frame
        track.tallest = max(track.tallest, shape.height)
        # The gate that ties a coverage episode to *this* person: somebody who never came near
        # the door cannot be credited with having covered it. Without it, a door swinging shut
        # followed by anybody wandering into view reads as an arrival.
        if shape.height >= self._config.min_height and overlaps(shape, self._config.zone):
            track.reached_zone = True

    def _finished(self, present: set[int]) -> list[Verdict]:
        gone = [
            track_id
            for track_id, track in self._tracks.items()
            if track_id not in present and self._frame - track.last > self._lost_after
        ]
        verdicts = []
        for track_id in gone:
            verdicts.append(self._decide(track_id, self._tracks.pop(track_id)))
        return verdicts

    def _decide(self, track_id: int, track: _Track) -> Verdict:
        if not track.reached_zone:
            return Verdict(track_id, None, "never reached the door")

        near = [
            episode
            for episode in self._episodes
            if episode.last >= track.first - self._window
            and episode.first <= track.last + self._window
        ]
        if not near:
            return Verdict(track_id, None, "no coverage near them")

        middle = (track.first + track.last) / 2
        before = [e for e in near if e.middle < middle]
        after = [e for e in near if e.middle >= middle]

        passing = self._config.passing_means
        arriving = _opposite(passing)
        if before and after:
            # Both halves happened. Reported as such rather than refused: the count nets to
            # zero either way, but refusing loses the visit from the record entirely, and the
            # two episodes are already identified and timestamped.
            return Verdict(track_id, None, f"came {arriving.value} then went {passing.value}")
        if before:
            return Verdict(track_id, arriving, "covered before they appeared")
        return Verdict(track_id, passing, "covered after they were last seen")


def _opposite(direction: Direction) -> Direction:
    return Direction.OUT if direction is Direction.IN else Direction.IN
