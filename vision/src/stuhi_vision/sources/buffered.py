"""Read a source as fast as it produces, process behind real time.

The problem this solves: the models are slower than the camera. Consuming a live source
directly means every frame that arrives while a frame is being processed is *lost* -- so a
doorway crossing lasting a second yields the one or two frames we happened to be free for.

Decoupling fixes it. A reader thread drains the camera at full rate into a bounded queue;
the pipeline consumes from the queue whenever it is ready. A one-second crossing banks a
dozen frames, and we are allowed to spend several seconds working through them. Frames
arrive in order, so tracking is unaffected -- the only cost is latency, which for
occupancy logging is nearly free.

The queue is bounded, and when it is full the reader **waits** rather than discarding.
Dropping frames looks harmless and is not: ByteTrack associates detections between
*consecutive* frames, so a gap makes it fail to confirm a track, ``boxes.id`` comes back
``None``, and the tracker reports nobody at all. An earlier version of this class dropped
its oldest frame when full, and the result was a pipeline that detected people in replay
but never once committed a crossing from the live camera.

Waiting is self-regulating: back-pressure reaches the camera's socket, so it slows down
instead of us losing continuity. Latency is bounded by ``capacity`` divided by the frame
rate -- at 20 fps, 64 frames is about three seconds behind at worst -- and since the motion
gate makes idle frames nearly free, the backlog drains during the quiet stretches.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator

from ..domain import Frame
from .base import FrameSource

_SENTINEL = object()  # posted by the reader when the upstream source ends


class BufferedSource:
    """Wrap a source so a slow consumer cannot cause frames to be missed.

    ``capacity`` is how many frames may wait. It bounds latency as well as memory: at
    ~14 KB a VGA JPEG decodes to ~900 KB, so 64 frames is tens of megabytes and a few
    seconds of backlog.
    """

    def __init__(self, source: FrameSource, capacity: int = 64) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._source = source
        self._queue: queue.Queue = queue.Queue(maxsize=capacity)
        self._stop = threading.Event()
        self._high_water = 0

    @property
    def high_water(self) -> int:
        """Deepest the backlog ever got -- at capacity, the reader was being held up."""
        return self._high_water

    def __iter__(self) -> Iterator[Frame]:
        reader = threading.Thread(target=self._read, name="frame-reader", daemon=True)
        reader.start()
        try:
            while True:
                item = self._queue.get()
                if item is _SENTINEL:
                    return
                yield item
        finally:
            self._stop.set()

    def _read(self) -> None:
        try:
            for frame in self._source:
                if self._stop.is_set():
                    return
                self._offer(frame)
        finally:
            self._queue.put(_SENTINEL)

    def _offer(self, frame: Frame) -> None:
        """Enqueue a frame, waiting if the consumer is behind. Never discards."""
        while not self._stop.is_set():
            try:
                self._queue.put(frame, timeout=0.5)
            except queue.Full:
                continue  # consumer is behind; hold the frame rather than lose continuity
            self._high_water = max(self._high_water, self._queue.qsize())
            return
