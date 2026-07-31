"""The most recent frame from each camera, for anything outside the pipeline to look at.

These cameras serve **one client at a time**. So while the pipeline is streaming, nothing
else can open the camera -- not a browser, not a capture script, not a calibration page. That
has cost several wasted walks down the corridor, and it makes drawing a zone against the live
view impossible unless the pipeline hands its own frames over.

It already has them. This keeps the newest one per camera, encoded once on demand rather than
on every frame, so the cost is paid only when somebody is actually looking.

The frames come from the **reader**, not from the far end of the pipeline. That distinction is
the whole point of the file: the reader banks several hundred frames so a slow machine loses
nothing, which means the frame the pipeline is working on can be twenty seconds old. Somebody
drawing a zone, or checking what a camera can see, wants the view *now* -- so how old the frame
is, is reported with it, and a page that says "current view" can be held to it.
"""

from __future__ import annotations

import threading
import time


class LatestFrames:
    """Newest frame per camera, shared between the pipeline threads and the web UI."""

    def __init__(self, encode_jpeg) -> None:
        self._encode_jpeg = encode_jpeg
        self._frames: dict[str, object] = {}
        self._stamped: dict[str, float] = {}
        self._lock = threading.RLock()

    def put(self, camera: str, image) -> None:
        """Remember this frame. Called per frame, so it must stay trivial -- no encoding."""
        now = time.monotonic()
        with self._lock:
            self._frames[camera] = image
            self._stamped[camera] = now

    def raw(self, camera: str):
        """The stored frame itself, for a caller that wants to measure or compare it."""
        with self._lock:
            return self._frames.get(camera)

    def age(self, camera: str) -> float | None:
        """Seconds since this frame arrived, so "current view" can be checked, not trusted."""
        with self._lock:
            stamped = self._stamped.get(camera)
        return None if stamped is None else time.monotonic() - stamped

    def jpeg(self, camera: str) -> bytes | None:
        """The newest frame as JPEG, encoded now rather than kept encoded."""
        image = self.raw(camera)
        return None if image is None else self._encode_jpeg(image)

    @property
    def cameras(self) -> list[str]:
        with self._lock:
            return sorted(self._frames)
