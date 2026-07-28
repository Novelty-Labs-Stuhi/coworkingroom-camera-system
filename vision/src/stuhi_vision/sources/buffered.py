"""Read a source as fast as it produces, process behind real time.

The problem this solves: the models are slower than the camera. Consuming a live source
directly means every frame that arrives while a frame is being processed is *lost* -- so a
doorway crossing lasting a second yields the one or two frames we happened to be free for.

Decoupling fixes it. A reader thread drains the camera at full rate into a bounded queue;
the pipeline consumes from the queue whenever it is ready. A one-second crossing banks a
dozen frames, and we are allowed to spend several seconds working through them. Frames
arrive in order, so tracking is unaffected -- the only cost is latency, which for
occupancy logging is nearly free.

The queue is bounded and drops its *oldest* frame when full. That matters: during a long
idle stretch the buffer must not grow without limit, and if we ever fall far behind, the
newest frames are the ones worth keeping.
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
        self._dropped = 0

    @property
    def dropped(self) -> int:
        """Frames discarded because the consumer fell too far behind."""
        return self._dropped

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
        """Enqueue a frame, discarding the oldest if the buffer is already full."""
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._dropped += 1
            except queue.Empty:  # pragma: no cover - consumer drained it in between
                pass
            try:
                self._queue.put_nowait(frame)
            except queue.Full:  # pragma: no cover - another producer cannot exist
                self._dropped += 1
