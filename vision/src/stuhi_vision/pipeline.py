"""The pipeline: frames in, occupancy changes out.

Each frame: track the people, let every track's session accumulate (recognise/embed
while they are in view), ask the doorway for crossings, and commit each crossing via
the Doorkeeper. Sessions whose tracks vanish are pruned. This is the only place the
stages are joined; each stays independently testable.

Cost control lives in the detector, not here: see :class:`~.tracking.GatedTracker`, which
skips still frames and crops to the doorway without the pipeline knowing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .attention import Attention
from .domain import Crossing, Frame, Sighting, TrackedPerson
from .doorway import DoorwayMonitor
from .handlers import Doorkeeper
from .sessions import SessionManager
from .sources.base import FrameSource
from .tracking import Detector

# Announced after each committed crossing, with the evidence behind its identity.
Announcer = Callable[[Sighting], None]

# Called once per frame with everything computed for it (e.g. to draw an overlay).
FrameObserver = Callable[[Frame, list[TrackedPerson], list[Crossing]], None]


@dataclass(frozen=True, slots=True)
class Hooks:
    """What the pipeline tells the outside world. The two travel together everywhere."""

    announce: Announcer | None = None
    on_frame: FrameObserver | None = None
    # Called once per frame with nothing to say, for work that must happen on a clock rather
    # than on an event -- emptying the room at the office-day boundary. It hangs off the
    # frame loop deliberately: a timer thread that dies takes its schedule silently with it,
    # whereas if frames stop arriving the pipeline is already broken in a louder way.
    tick: Callable[[], None] | None = None


class Pipeline:
    """Wire a frame source through tracking, sessions, the doorway, and the doorkeeper."""

    def __init__(
        self,
        source: FrameSource,
        tracker: Detector,
        doorway: DoorwayMonitor,
        sessions: SessionManager,
        doorkeeper: Doorkeeper,
        hooks: Hooks | None = None,
        attention: Attention | None = None,
    ) -> None:
        self._source = source
        self._tracker = tracker
        self._doorway = doorway
        self._sessions = sessions
        self._doorkeeper = doorkeeper
        hooks = hooks or Hooks()
        self._announce = hooks.announce or (lambda sighting: None)
        self._on_frame = hooks.on_frame
        self._beat = hooks.tick or (lambda: None)
        self._attention = attention
        self._examined = 0

    def run(self) -> None:
        if self._attention is not None:
            self._run_attentively()
            return
        for frame_index, frame in enumerate(self._source):
            self._beat()
            people = self._tracker.update(frame)
            if people is None:
                # The frame was never examined (motion gate). Touch no per-track state:
                # telling the doorway "nobody is here" would make it forget which side
                # everyone was on, and a crossing spanning an idle frame would be lost.
                if self._on_frame is not None:
                    self._on_frame(frame, [], [])
                continue

            self._sessions.observe(frame, people, frame_index)
            crossings = self._doorway.update(people, frame)
            for crossing in crossings:
                sighting = self._doorkeeper.commit(crossing)
                if sighting is not None:
                    self._announce(sighting)
            self._sessions.prune({person.track_id for person in people}, frame_index)
            if self._on_frame is not None:
                self._on_frame(frame, people, crossings)

    def _run_attentively(self) -> None:
        """Detection runs only when the doorframe is busy, and over the approach to it.

        The doorway is asked on *every* frame regardless, because it is what watches the
        pixels: it is the cheap half, and it is what decides when the expensive half wakes.
        """
        for frame in self._source:
            self._beat()
            people = self._examine(frame)
            crossings = self._doorway.update(people, frame)
            for crossing in crossings:
                sighting = self._doorkeeper.commit(crossing)
                if sighting is not None:
                    self._announce(sighting)
            if self._on_frame is not None:
                self._on_frame(frame, people, crossings)

    def _examine(self, frame: Frame) -> list[TrackedPerson]:
        """Run the models over whatever attention hands back, and return the latest people.

        The approach frames go through the *whole* stage -- tracking and the session that
        recognises faces -- because a face seen on the way to the door is the only look at
        somebody who then crosses it side-on.
        """
        people: list[TrackedPerson] = []
        for examined in self._attention.examine(frame):
            seen = self._tracker.update(examined)
            if seen is None:
                continue
            people = seen
            self._sessions.observe(examined, seen, self._examined)
            self._examined += 1
        if people:
            self._sessions.prune({person.track_id for person in people}, self._examined)
        return people
