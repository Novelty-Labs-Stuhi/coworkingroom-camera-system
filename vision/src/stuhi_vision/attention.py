"""Run the expensive models only when the doorframe is busy -- including just before it was.

Detection costs about 200 ms a frame on this machine; the pixel check over the box costs a
third of a millisecond. Running detection on every frame of an empty corridor is what makes the
pipeline fall behind the cameras: measured live, both camera sockets had over a megabyte of
unread stream backed up in the kernel while the machine sat at 193% CPU.

So the pixels decide when the models run. The catch, and the reason this module exists rather
than a plain gate: **the recognisable face is in the approach, before the box is touched.** By
the time the doorframe is covered the person is at it, often side-on or already past. Waking up
then would count passages and never name anybody.

So recent frames are kept, and when the box becomes covered they are replayed through detection
and recognition first, in order. The person is then tracked from before they arrived -- the
track id spans the approach and the crossing, so the face seen on the way in belongs to the
same session the crossing commits. That id is the whole link between "somebody crossed" and
"it was this person".

The buffer holds raw frames: a few seconds at a megabyte each, which is cheap next to what is
saved by not running a detector on an empty corridor.
"""

from __future__ import annotations

from collections import deque

from .domain import Frame
from .occlusion import Coverage, Occlusion

# How much of the approach is kept, in *seconds*. A frame count silently changes meaning when
# the frame rate does: fifteen frames was three seconds when the pipeline could only consume
# five a second, and became one second the moment it could keep up with the camera at sixteen.
_PRE_ROLL_SECONDS = 3.0
# ...but bounded by a frame count too, because the cost of waking is paid in frames, not in
# seconds: each replayed frame costs a detection, about 200 ms here. Three seconds at sixteen
# frames a second is about fifty frames, so waking spends roughly ten seconds of catching up.
# The buffer in front of the source is what absorbs that; raise them together, or the camera
# ends up waiting on a full queue. Memory is not the constraint: a frame is under a megabyte
# and this machine has gigabytes spare.
_MOST_FRAMES = 64
# Kept looking at, after the box clears, so a track ends on its own rather than mid-stride.
_LINGER_SECONDS = 0.7


class Attention:
    """Decides which frames the detector sees, and replays the approach when it matters."""

    def __init__(
        self,
        occlusion: Occlusion,
        pre_roll_seconds: float = _PRE_ROLL_SECONDS,
        linger_seconds: float = _LINGER_SECONDS,
        most_frames: int = _MOST_FRAMES,
    ) -> None:
        self._occlusion = occlusion
        self._pre_roll = max(pre_roll_seconds, 0.0)
        self._linger = max(linger_seconds, 0.0)
        self._recent: deque[Frame] = deque(maxlen=max(most_frames, 1))
        self._quiet_since: float | None = None
        self._was_busy = False
        self._episode: Coverage | None = None
        # The last frame already handed to the models, so a second person arriving right
        # behind the first is not examined twice -- and, more importantly, so the buffer can
        # keep filling while the first is still being watched. It used to be cleared on
        # waking, which left whoever came next with no approach at all.
        self._examined_until: float | None = None

    def examine(self, frame: Frame) -> list[Frame]:
        """The frames the models should run on now -- the approach too, when waking up."""
        self._episode = self._occlusion.update(frame.image)
        busy = self._occlusion.busy

        self._recent.append(frame)   # always: the next person's approach starts now

        if busy and not self._was_busy:
            self._was_busy = True
            self._quiet_since = None
            return self._hand_over(self._approach(frame.timestamp))

        if busy:
            self._was_busy = True
            self._quiet_since = None
            return self._hand_over([frame])

        if self._was_busy:
            # Just cleared. The linger is measured from here -- not from startup, which would
            # have the models running before anything had ever happened.
            self._quiet_since = frame.timestamp
            self._was_busy = False
        if self._quiet_since is not None and frame.timestamp - self._quiet_since <= self._linger:
            # Keep watching briefly after the box clears: the track has to end on its own
            # rather than be cut off mid-stride, and the walk away is still worth seeing.
            return self._hand_over([frame])

        self._quiet_since = None
        return []

    def _approach(self, now: float) -> list[Frame]:
        """The kept frames from within the pre-roll of this moment, oldest first."""
        return [frame for frame in self._recent if now - frame.timestamp <= self._pre_roll]

    def _hand_over(self, frames: list[Frame]) -> list[Frame]:
        """Frames the models have not already seen, remembering how far they have got."""
        seen = self._examined_until
        fresh = [f for f in frames if seen is None or f.timestamp > seen]
        if fresh:
            self._examined_until = fresh[-1].timestamp
        return fresh

    def episode(self) -> Coverage | None:
        """The coverage episode that finished on this frame, if one did."""
        return self._episode

    @property
    def busy(self) -> bool:
        return self._occlusion.busy
