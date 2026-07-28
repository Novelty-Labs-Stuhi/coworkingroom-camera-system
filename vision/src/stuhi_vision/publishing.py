"""Hold a sighting until the person has left, so its clip has an ending.

A crossing is committed the moment the foot point crosses the line -- which is the middle
of the event, not the end of it. Publishing there gives a clip that stops mid-stride and
tells you nothing about what happened next.

So a committed sighting waits. Its clip keeps filling until the frame has been clear of
people for ``clear_frames`` consecutive frames, guaranteeing the video ends on an empty
doorway, and a hard cap stops someone loitering from holding a clip open forever.

The cost is latency: a notification arrives a second or two after the crossing rather than
at it. For occupancy that is invisible, and a clip you can actually interpret is worth far
more than one that arrives marginally sooner.
"""

from __future__ import annotations

from dataclasses import dataclass

from .clips import ClipRecorder, PendingClip
from .domain import Sighting


@dataclass
class _Held:
    sighting: Sighting
    clip: PendingClip
    position: int  # 1-based order of crossing within this burst
    clear_frames: int = 0


class SightingPublisher:
    """Defers each sighting until its clip is complete, then files and announces it."""

    def __init__(
        self,
        recorder: ClipRecorder,
        publish,
        clear_frames: int = 10,
    ) -> None:
        self._recorder = recorder
        self._publish = publish
        self._clear_frames = max(1, clear_frames)
        self._held: list[_Held] = []
        self._burst_count = 0

    @property
    def pending_count(self) -> int:
        return len(self._held)

    def hold(self, sighting: Sighting) -> None:
        """Take a committed sighting and start collecting the rest of its clip.

        Crossings are numbered within a *burst* -- consecutive crossings with no clear gap
        between them, i.e. a group of people coming through together. When several people
        cross at once their clips look almost identical, so the position is the only thing
        that says which person a given clip is about.
        """
        self._burst_count += 1
        self._held.append(
            _Held(sighting=sighting, clip=self._recorder.begin(), position=self._burst_count)
        )

    def advance(self, people_present: bool) -> None:
        """Call once per frame. Publishes any sighting whose clip is now complete."""
        if not self._held:
            return
        ready = []
        for held in self._held:
            held.clear_frames = 0 if people_present else held.clear_frames + 1
            if held.clear_frames >= self._clear_frames or held.clip.full:
                ready.append(held)
        self._emit_all(ready)

    def flush(self) -> None:
        """Publish everything still waiting -- used on shutdown so nothing is lost."""
        self._emit_all(list(self._held))

    def _emit_all(self, ready: list[_Held]) -> None:
        """Publish in crossing order, so the numbering the captions show is meaningful."""
        if not ready:
            return
        total = self._burst_count
        for held in sorted(ready, key=lambda h: h.position):
            self._held.remove(held)
            self._publish(held.sighting, self._recorder.finish(held.clip), held.position, total)
        if not self._held:
            self._burst_count = 0  # burst over; the next person starts a new group
