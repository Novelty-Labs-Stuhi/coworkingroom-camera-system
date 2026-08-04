"""Run the expensive models only when the doorframe is busy -- including just before it was.

Detection costs about 200 ms a frame on this machine; the pixel check over the box costs a
third of a millisecond. Running detection on every frame of an empty corridor is what makes the
pipeline fall behind the cameras: measured live, both camera sockets had over a megabyte of
unread stream backed up in the kernel while the machine sat at 193% CPU.

So the pixels decide when the models run. The catch, and the reason this module exists rather
than a plain gate: **the answer is in the approach, before the box is touched.** By the time the
doorframe is covered the person is at it, often side-on or already past. Waking up then would
count passages, never name anybody, and -- for a rule that reads direction from whether somebody
was visible *before* the covering -- have nothing to read at all.

So frames are kept in a circular buffer, and when the box becomes covered they are replayed
through detection and recognition. Two things about how they are replayed earn their keep:

* **Backwards, in chunks, only as far as needed.** The newest chunk before the covering is
  handed over first. If that settled it -- somebody detected, a direction readable, a face --
  nothing more is spent. If it did not, the caller asks for the chunk before that, and so on,
  until it has enough or the buffer runs out. A close passage costs one chunk; an awkward one
  pays for what it needs. Handing over a fixed pre-roll every time paid the worst case always.
* **Each frame carries the doorframe's state as it was *then*.** Without this every replayed
  frame reads as covered, because the box *is* covered by the time the replay happens -- which
  silently collapses "seen before the covering" and "seen after it" into one answer, and a rule
  built on that distinction would answer the same way every time.

Frames within a chunk go out oldest-first, so tracking still sees time moving forwards. Track
ids may not survive from one chunk to the next; that is the acknowledged cost of not processing
frames nobody needed, and reaching one chunk further back is the remedy.

The buffer holds raw frames -- about a second of video per megabyte, and this machine has
gigabytes spare. Latency is paid only when something actually happened, and next to how much of
the day is an empty corridor there is room for it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .domain import Frame
from .occlusion import Coverage, Occlusion

# How much of the approach is kept, in *seconds*. A frame count silently changes meaning when
# the frame rate does: fifteen frames was three seconds when the pipeline could only consume
# five a second, and became one second the moment it could keep up with the camera at sixteen.
#
# Generous, because keeping a frame is nearly free now that only the chunks actually asked for
# cost anything. It bounds how far back a question may reach, not how much work is done.
_PRE_ROLL_SECONDS = 12.0
# ...and a frame ceiling, so a fast camera cannot turn those seconds into more memory than the
# machine has. At 19 fps twelve seconds is about 230 frames; this leaves headroom above that.
_MOST_FRAMES = 320
# How many frames one chunk of the replay is. Each costs a detection, so this is the unit of
# work: small enough that a close passage is cheap, large enough that a track has some frames
# to form over.
_CHUNK_FRAMES = 12
# Kept looking at, after the box clears, so a track ends on its own rather than mid-stride.
_LINGER_SECONDS = 0.7


@dataclass(frozen=True, slots=True)
class Examined:
    """One frame handed to the models, with the doorframe's state as it was on *that* frame.

    The pair is the point. Replayed frames come from before the box was covered, so a rule
    asking "was this person seen before the covering" would get "covered" for every one of them
    if it read the box as it stands at replay time.
    """

    frame: Frame
    covered: bool


class Attention:
    """Decides which frames the detector sees, and replays the approach when it matters."""

    def __init__(
        self,
        occlusion: Occlusion,
        pre_roll_seconds: float = _PRE_ROLL_SECONDS,
        linger_seconds: float = _LINGER_SECONDS,
        most_frames: int = _MOST_FRAMES,
        chunk_frames: int = _CHUNK_FRAMES,
    ) -> None:
        self._occlusion = occlusion
        self._pre_roll = max(pre_roll_seconds, 0.0)
        self._linger = max(linger_seconds, 0.0)
        self._chunk = max(chunk_frames, 1)
        self._recent: deque[Examined] = deque(maxlen=max(most_frames, 1))
        self._quiet_since: float | None = None
        self._was_busy = False
        self._episode: Coverage | None = None
        # The last frame already handed to the models, so a second person arriving right
        # behind the first is not examined twice -- and, more importantly, so the buffer can
        # keep filling while the first is still being watched. It used to be cleared on
        # waking, which left whoever came next with no approach at all.
        self._examined_until: float | None = None
        # How far back the current wake has already reached. `earlier()` walks this backwards.
        self._replayed_from: float | None = None
        # When this wake happened. The pre-roll is measured from here rather than from the edge
        # of the last chunk, so reaching back repeatedly cannot walk off into the whole buffer:
        # the total a single question may cost is bounded, however many chunks it asks for.
        self._wake_at: float | None = None

    def examine(self, frame: Frame) -> list[Examined]:
        """The frames the models should run on now -- the newest approach chunk when waking."""
        self._episode = self._occlusion.update(frame.image)
        busy = self._occlusion.busy

        here = Examined(frame=frame, covered=busy)
        self._recent.append(here)   # always: the next person's approach starts now

        if busy and not self._was_busy:
            self._was_busy = True
            self._quiet_since = None
            # The covering frame itself, behind the newest chunk of the approach to it.
            self._wake_at = frame.timestamp
            chunk = self._before(frame.timestamp)
            self._replayed_from = chunk[0].frame.timestamp if chunk else frame.timestamp
            return self._hand_over([*chunk, here])

        if busy:
            self._was_busy = True
            self._quiet_since = None
            return self._hand_over([here])

        if self._was_busy:
            # Just cleared. The linger is measured from here -- not from startup, which would
            # have the models running before anything had ever happened.
            self._quiet_since = frame.timestamp
            self._was_busy = False
        if self._quiet_since is not None and frame.timestamp - self._quiet_since <= self._linger:
            # Keep watching briefly after the box clears: the track has to end on its own
            # rather than be cut off mid-stride, and the walk away is still worth seeing.
            return self._hand_over([here])

        self._quiet_since = None
        return []

    def earlier(self) -> list[Examined]:
        """The chunk before whatever this wake has already replayed. Empty when exhausted.

        For the caller that looked at the newest chunk and still cannot tell -- nobody detected,
        no direction readable, no face. Asking again reaches further back for the price of one
        more chunk, rather than the whole buffer being spent up front on the passages that never
        needed it.
        """
        if self._replayed_from is None:
            return []
        chunk = self._before(self._replayed_from)
        if not chunk:
            return []
        self._replayed_from = chunk[0].frame.timestamp
        # Deliberately not filtered through `_examined_until`: that watermark tracks how far
        # *forward* the models have got, and every frame here is behind it by construction.
        return chunk

    def _before(self, boundary: float) -> list[Examined]:
        """The newest chunk of kept frames strictly before ``boundary``, oldest first.

        Bounded against the moment this wake happened, not against ``boundary``: measuring from
        the edge of the last chunk would let each further request buy another pre-roll, so a
        single awkward passage could walk back through the entire buffer.
        """
        reach = self._wake_at if self._wake_at is not None else boundary
        candidates = [
            seen
            for seen in self._recent
            if seen.frame.timestamp < boundary
            and reach - seen.frame.timestamp <= self._pre_roll
        ]
        return candidates[-self._chunk :]

    def _hand_over(self, seen: list[Examined]) -> list[Examined]:
        """Frames the models have not already seen, remembering how far they have got."""
        until = self._examined_until
        fresh = [s for s in seen if until is None or s.frame.timestamp > until]
        if fresh:
            self._examined_until = fresh[-1].frame.timestamp
        return fresh

    def use_zone(self, zone: tuple[float, float, float, float]) -> None:
        """Watch a redrawn box. The kept approach is still valid: it is whole frames."""
        self._occlusion.use_zone(zone)

    def episode(self) -> Coverage | None:
        """The coverage episode that finished on this frame, if one did."""
        return self._episode

    @property
    def busy(self) -> bool:
        return self._occlusion.busy
