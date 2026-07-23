"""The frame-source contract, plus a small helper to read an OpenCV capture.

Both the local-camera and network-stream sources decode frames the same way; the
only differences are how the capture is opened, how timestamps are assigned, and
whether a dropped stream should reconnect. Those differences live in the subclasses;
the decode loop lives here once (DRY).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Protocol

from ..domain import Frame


class FrameSource(Protocol):
    def __iter__(self) -> Iterator[Frame]: ...


def read_capture(capture, *, wall_clock: bool, fps_hint: float = 30.0) -> Iterator[Frame]:
    """Yield frames from an opened ``cv2.VideoCapture`` until it is exhausted.

    ``wall_clock`` timestamps live sources with the real time each frame arrives;
    offline files are timestamped from their frame index and nominal FPS instead.
    """
    index = 0
    fps = capture.get(_CAP_PROP_FPS) or fps_hint
    while True:
        ok, image = capture.read()
        if not ok:
            return
        timestamp = time.time() if wall_clock else index / fps
        yield Frame(timestamp=timestamp, image=image)
        index += 1


_CAP_PROP_FPS = 5  # cv2.CAP_PROP_FPS, inlined to avoid importing cv2 in this pure helper
