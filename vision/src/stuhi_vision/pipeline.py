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
    # Handed each frame's pixels before the detector runs, for a rule that needs to know whether
    # the doorframe was covered on this frame. Deliberately not :class:`Attention`: that one
    # *gates* detection on the box being busy, and a rule asking "was this person visible before
    # the box was covered" would then only ever see frames from after the moment in question.
    doorframe: Callable[[object], None] | None = None


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
        self._doorframe = hooks.doorframe or (lambda image: None)
        self._attention = attention
        self._examined = 0

    def run(self) -> None:
        if self._attention is not None:
            self._run_attentively()
            return
        for frame_index, frame in enumerate(self._source):
            self._beat()
            # Before the detector, so the coverage a rule reads belongs to this frame rather
            # than the one before it.
            self._doorframe(frame.image)
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

        When the newest chunk of the approach shows nobody, earlier chunks are asked for one at
        a time until somebody appears or the buffer runs out. Nothing is spent reaching back
        past the answer: most passages are settled by the first chunk, and only the awkward ones
        pay for more.
        """
        people: list[TrackedPerson] = []
        chunk = self._attention.examine(frame)
        while chunk:
            people = self._run_over(chunk, live_at=frame.timestamp) or people
            if people:
                break
            # Nobody in that chunk. The frames before it are the only place left to look, and
            # they are the difference between a counted passage and one refused for having
            # nobody on it -- which is most of the ones refused.
            chunk = self._attention.earlier()
        if people:
            self._sessions.prune({person.track_id for person in people}, self._examined)
        return people

    def _run_over(self, chunk: list, live_at: float) -> list[TrackedPerson]:
        """Track and recognise over one chunk, oldest first.

        The doorway is told about each *replayed* frame with the coverage that frame actually
        had, so a rule reading the order of things sees the approach as uncovered and the
        crossing as covered -- which is what they were. The live frame is left to the caller,
        which tells the doorway about it once per tick whether or not anything was examined.
        """
        people: list[TrackedPerson] = []
        for seen in chunk:
            found = self._tracker.update(seen.frame)
            if found is None:
                continue
            people = found
            self._sessions.observe(seen.frame, found, self._examined)
            if seen.frame.timestamp != live_at:
                self._doorway.update(found, seen.frame, covered=seen.covered)
            self._examined += 1
        return people
