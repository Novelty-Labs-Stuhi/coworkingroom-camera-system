"""Frame sources: video files, local webcams, and network camera streams.

A source is simply an iterable of :class:`~stuhi_vision.domain.Frame`. The rest of
the pipeline does not know or care whether frames come from a file or a live camera.
"""

from __future__ import annotations

from ..config import SourceConfig
from .base import FrameSource
from .buffered import BufferedSource
from .oriented import OrientedSource
from .stream import NetworkStreamSource
from .video import VideoFileSource, WebcamSource


def open_source(config: SourceConfig) -> FrameSource:
    """Build the frame source described by ``config``, corrected to upright."""
    return OrientedSource(_raw_source(config), config.rotate)


def _raw_source(config: SourceConfig) -> FrameSource:
    if config.kind == "file":
        return VideoFileSource(config.target)
    if config.kind == "webcam":
        return WebcamSource(int(config.target))
    if config.kind == "stream":
        return NetworkStreamSource(config.target)
    raise ValueError(f"unknown source kind: {config.kind!r}")


__all__ = [
    "BufferedSource",
    "FrameSource",
    "NetworkStreamSource",
    "OrientedSource",
    "VideoFileSource",
    "WebcamSource",
    "open_source",
]
