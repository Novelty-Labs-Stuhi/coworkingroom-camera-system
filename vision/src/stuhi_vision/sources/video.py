"""Local frame sources: a video file (offline) and a webcam (live)."""

from __future__ import annotations

from collections.abc import Iterator

import cv2

from ..domain import Frame
from .base import read_capture


class VideoFileSource:
    """Frames from a recorded video file, timestamped from their position in the file."""

    def __init__(self, path: str) -> None:
        self._path = path

    def __iter__(self) -> Iterator[Frame]:
        capture = cv2.VideoCapture(self._path)
        if not capture.isOpened():
            raise FileNotFoundError(f"cannot open video file: {self._path}")
        try:
            yield from read_capture(capture, wall_clock=False)
        finally:
            capture.release()


class WebcamSource:
    """Frames from a locally attached camera, timestamped with wall-clock time."""

    def __init__(self, index: int = 0) -> None:
        self._index = index

    def __iter__(self) -> Iterator[Frame]:
        capture = cv2.VideoCapture(self._index)
        if not capture.isOpened():
            raise RuntimeError(f"cannot open webcam index {self._index}")
        try:
            yield from read_capture(capture, wall_clock=True)
        finally:
            capture.release()
