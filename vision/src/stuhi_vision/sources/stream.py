"""Network camera source (ESP32-CAM MJPEG / RTSP) with automatic reconnect.

Cheap WiFi cameras drop their connection routinely, so a live source must survive a
stream ending and pick it back up rather than stopping the whole pipeline.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import cv2

from ..domain import Frame
from .base import read_capture

_RECONNECT_DELAY_S = 2.0


class NetworkStreamSource:
    """Frames from a network stream URL, reconnecting indefinitely on failure."""

    def __init__(self, url: str, reconnect_delay: float = _RECONNECT_DELAY_S) -> None:
        self._url = url
        self._reconnect_delay = reconnect_delay

    def __iter__(self) -> Iterator[Frame]:
        while True:
            capture = cv2.VideoCapture(self._url)
            if capture.isOpened():
                try:
                    yield from read_capture(capture, wall_clock=True)
                finally:
                    capture.release()
            time.sleep(self._reconnect_delay)  # stream ended or would not open; retry
