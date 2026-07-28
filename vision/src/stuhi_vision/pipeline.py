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


class Pipeline:
    """Wire a frame source through tracking, sessions, the doorway, and the doorkeeper."""

    def __init__(
        self,
        source: FrameSource,
        tracker: Detector,
        doorway: DoorwayMonitor,
        sessions: SessionManager,
        doorkeeper: Doorkeeper,
        announce: Announcer | None = None,
        on_frame: FrameObserver | None = None,
    ) -> None:
        self._source = source
        self._tracker = tracker
        self._doorway = doorway
        self._sessions = sessions
        self._doorkeeper = doorkeeper
        self._announce = announce or (lambda sighting: None)
        self._on_frame = on_frame

    def run(self) -> None:
        for frame_index, frame in enumerate(self._source):
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
