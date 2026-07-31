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

import logging
import queue
import threading
import time
from collections.abc import Iterator

from ..domain import Frame
from .base import FrameSource

_log = logging.getLogger(__name__)

_SENTINEL = object()  # posted by the reader when the upstream source ends


class BufferedSource:
    """Wrap a source so a slow consumer cannot cause frames to be missed.

    ``capacity`` is how many frames may wait. It bounds latency as well as memory: at
    ~14 KB a VGA JPEG decodes to ~900 KB, so 64 frames is tens of megabytes and a few
    seconds of backlog.
    """

    def __init__(
        self,
        source: FrameSource,
        capacity: int = 64,
        name: str = "",
        on_read=None,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._source = source
        self._name = name
        self._queue: queue.Queue = queue.Queue(maxsize=capacity)
        # Told about each frame the moment it arrives, before it joins the backlog. Anything
        # wanting the *current* view has to be fed from here: by the time a frame reaches the
        # far end of the pipeline it can be twenty seconds old.
        self._on_read = on_read
        self._stop = threading.Event()
        self._high_water = 0
        # What the camera actually delivers, as opposed to what the models keep up with. The
        # two are different numbers and the difference decides what is possible: a rule that
        # needs several frames of a passage can only run on the reader's rate, not the
        # pipeline's. Measuring it beats assuming it -- every estimate so far has been wrong.
        self._read_frames = 0
        self._read_since = 0.0
        self._rate = 0.0

    @property
    def rate(self) -> float:
        """Frames per second arriving from the camera, over the last stretch."""
        return self._rate

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
                self._count()
                if self._on_read is not None:
                    self._on_read(frame)
                self._offer(frame)
        finally:
            self._queue.put(_SENTINEL)

    def _count(self) -> None:
        """Measure the arrival rate, reporting it every hundred frames."""
        now = time.monotonic()
        if not self._read_since:
            self._read_since = now
        self._read_frames += 1
        if self._read_frames < 100:
            return
        # A coarse clock can read the interval as exactly zero over a fast hundred frames,
        # and the report used to be skipped entirely when it did -- so a source fast enough
        # to be interesting was the one that never said anything.
        elapsed = max(now - self._read_since, 1e-6)
        self._rate = self._read_frames / elapsed
        # The backlog is reported with the rate because the two together say which of
        # them is the bottleneck. This reader blocks when the queue is full, so a rate
        # measured here is the *consumer's* rate whenever the backlog is at capacity --
        # reading it as the camera's output, as has happened, understates it threefold.
        # A backlog well under capacity means the camera really is that slow.
        _log.info(
            "%s delivering %.1f fps, backlog peaked %d/%d",
            self._name or "camera",
            self._rate,
            self._high_water,
            self._queue.maxsize,
        )
        self._read_frames = 0
        self._read_since = now
        self._high_water = 0

    def _offer(self, frame: Frame) -> None:
        """Enqueue a frame, waiting if the consumer is behind. Never discards."""
        while not self._stop.is_set():
            try:
                self._queue.put(frame, timeout=0.5)
            except queue.Full:
                continue  # consumer is behind; hold the frame rather than lose continuity
            self._high_water = max(self._high_water, self._queue.qsize())
            return
