"""The pipeline: frames in, occupancy changes out.

Each frame: track the people, let every track's session accumulate (recognise/embed
while they are in view), ask the doorway for crossings, and commit each crossing via
the Doorkeeper. Sessions whose tracks vanish are pruned. This is the only place the
stages are joined; each stays independently testable.
"""

from __future__ import annotations

from collections.abc import Callable

from .domain import Crossing, Direction, Frame, TrackedPerson
from .doorway import DoorwayMonitor
from .handlers import Doorkeeper
from .sessions import SessionManager
from .sources.base import FrameSource
from .tracking import PersonTracker

# Announced after each committed crossing: (direction, attributed name or None).
Announcer = Callable[[Direction, str | None], None]

# Called once per frame with everything computed for it (e.g. to draw an overlay).
FrameObserver = Callable[[Frame, list[TrackedPerson], list[Crossing]], None]


class Pipeline:
    """Wire a frame source through tracking, sessions, the doorway, and the doorkeeper."""

    def __init__(
        self,
        source: FrameSource,
        tracker: PersonTracker,
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
        self._announce = announce or (lambda direction, name: None)
        self._on_frame = on_frame

    def run(self) -> None:
        for frame_index, frame in enumerate(self._source):
            people = self._tracker.update(frame)
            self._sessions.observe(frame, people, frame_index)
            crossings = self._doorway.update(people, frame)
            for crossing in crossings:
                result = self._doorkeeper.commit(crossing)
                if result is not None:
                    self._announce(*result)
            self._sessions.prune({person.track_id for person in people}, frame_index)
            if self._on_frame is not None:
                self._on_frame(frame, people, crossings)
