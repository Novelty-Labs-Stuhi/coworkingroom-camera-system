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


@dataclass(frozen=True, slots=True)
class Publication:
    """One finished sighting, ready to file and announce.

    ``position`` and ``total`` place it within its burst, and ``burst`` identifies the
    burst itself -- which is what lets a whole group be labelled in one command.
    """

    sighting: Sighting
    clip: bytes | None
    position: int
    total: int
    burst: int


@dataclass
class _Held:
    sighting: Sighting
    clip: PendingClip
    position: int  # 1-based order of crossing within this burst
    burst: int
    clear_frames: int = 0


class SightingPublisher:
    """Defers each sighting until its clip is complete, then files and announces it."""

    def __init__(
        self,
        recorder: ClipRecorder,
        publish,
        clear_frames: int = 10,
        burst_gap_seconds: float = 3.0,
    ) -> None:
        self._recorder = recorder
        self._publish = publish
        self._clear_frames = max(1, clear_frames)
        self._gap = max(burst_gap_seconds, 0.0)
        self._held: list[_Held] = []
        self._burst = 0
        # How many crossed in each recent burst, so a caption can say "2 of 3" correctly even
        # though the clips finish at different moments.
        self._sizes: dict[int, int] = {}
        self._last_crossing: float | None = None

    @property
    def pending_count(self) -> int:
        return len(self._held)

    def hold(self, sighting: Sighting) -> None:
        """Take a committed sighting and start collecting the rest of its clip.

        Crossings are numbered within a *burst*: people who came through together, which is
        decided by the **gap between their crossings**. When several cross at once their clips
        look almost identical, so the position is the only thing saying which person a clip is
        about -- and that is only worth anything if the group is really a group.

        The burst used to end only when every held clip had finished, and a clip's completion
        counter resets whenever anybody is in view. So in a busy room nothing ever finished and
        every crossing joined the same group: the labelling page offered a clip as "1 of 23
        together", asking for twenty-three names in crossing order for what were twenty-three
        separate passages, minutes apart.
        """
        if self._last_crossing is None or sighting.timestamp - self._last_crossing > self._gap:
            self._burst += 1  # too long since the last one: a new group
            self._forget_old_bursts()
        self._last_crossing = sighting.timestamp
        self._sizes[self._burst] = self._sizes.get(self._burst, 0) + 1
        self._held.append(
            _Held(
                sighting=sighting,
                clip=self._recorder.begin(),
                position=self._sizes[self._burst],
                burst=self._burst,
            )
        )

    def _forget_old_bursts(self, keep: int = 32) -> None:
        """Sizes are only needed while a burst's clips are still being published."""
        for burst in sorted(self._sizes)[:-keep]:
            del self._sizes[burst]

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
        for held in sorted(ready, key=lambda h: h.position):
            self._held.remove(held)
            self._publish(
                Publication(
                    sighting=held.sighting,
                    clip=self._recorder.finish(held.clip),
                    position=held.position,
                    # This burst's own size, not however many are held right now: clips finish
                    # at different moments, and taking the running total made every caption
                    # depend on what else happened to be in flight.
                    total=self._sizes.get(held.burst, held.position),
                    burst=held.burst,
                )
            )
