"""Rotate frames to upright before anything looks at them.

A camera bolted to a wall is often not level with the world -- ours ended up mounted
upside down. Neither YOLO nor RetinaFace tolerates that: both are trained on upright
images, and an inverted frame is outside what they can recognise. Measured on the same 70
frames of real footage, rotating 180 degrees took person detections from 8 to 38, made
ByteTrack confirm a track where it previously confirmed none, and took face detections
from 0 to 34.

Rotating at the source means every consumer -- detection, the doorway line, saved crops,
the review clip -- works in the same upright coordinate space, so nothing else needs to
know the camera is mounted oddly.

The proper fix is the sensor's own flip/mirror registers, which cost nothing; this exists
because reflashing a camera already on a wall is not always convenient, and because a
recorded file may need correcting after the fact.
"""

from __future__ import annotations

from collections.abc import Iterator

from ..domain import Frame
from .base import FrameSource

# cv2 rotation codes, inlined so this module stays importable without cv2 present.
_ROTATE_90_CLOCKWISE = 0
_ROTATE_180 = 1
_ROTATE_90_COUNTERCLOCKWISE = 2

_CODES = {90: _ROTATE_90_CLOCKWISE, 180: _ROTATE_180, 270: _ROTATE_90_COUNTERCLOCKWISE}


def normalise_degrees(degrees: int) -> int:
    """Validate a rotation, returning it in 0/90/180/270."""
    normalised = degrees % 360
    if normalised not in (0, 90, 180, 270):
        raise ValueError(f"rotation must be 0, 90, 180 or 270 degrees, not {degrees}")
    return normalised


class OrientedSource:
    """Wrap a source, rotating every frame by a fixed multiple of 90 degrees."""

    def __init__(self, source: FrameSource, degrees: int) -> None:
        self._source = source
        self._degrees = normalise_degrees(degrees)

    def __iter__(self) -> Iterator[Frame]:
        if self._degrees == 0:
            yield from self._source
            return

        import cv2

        code = _CODES[self._degrees]
        for frame in self._source:
            yield Frame(timestamp=frame.timestamp, image=cv2.rotate(frame.image, code))
