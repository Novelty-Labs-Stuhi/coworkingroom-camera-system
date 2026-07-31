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

_PRE_ROLL = 15   # frames of approach kept: about three seconds at this camera's rate
_LINGER = 8      # frames to keep looking after the box clears, so a track ends cleanly


class Attention:
    """Decides which frames the detector sees, and replays the approach when it matters."""

    def __init__(
        self,
        occlusion: Occlusion,
        pre_roll: int = _PRE_ROLL,
        linger: int = _LINGER,
    ) -> None:
        self._occlusion = occlusion
        self._recent: deque[Frame] = deque(maxlen=max(pre_roll, 1))
        self._linger = max(linger, 0)
        self._left = 0
        self._was_busy = False
        self._episode: Coverage | None = None

    def examine(self, frame: Frame) -> list[Frame]:
        """The frames the models should run on now -- the approach too, when waking up."""
        self._episode = self._occlusion.update(frame.image)
        busy = self._occlusion.busy

        if busy and not self._was_busy:
            waking = [*self._recent, frame]     # the approach, then now
            self._recent.clear()
            self._was_busy = True
            self._left = self._linger
            return waking

        self._was_busy = busy
        if busy:
            self._left = self._linger
            return [frame]
        if self._left > 0:
            # Keep watching briefly after the box clears: the track has to end on its own
            # rather than be cut off mid-stride, and the exit walk is still worth seeing.
            self._left -= 1
            return [frame]

        self._recent.append(frame)
        return []

    def episode(self) -> Coverage | None:
        """The coverage episode that finished on this frame, if one did."""
        return self._episode

    @property
    def busy(self) -> bool:
        return self._occlusion.busy
