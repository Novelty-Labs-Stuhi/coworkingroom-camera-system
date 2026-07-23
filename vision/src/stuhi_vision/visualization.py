"""Draw the pipeline's view onto frames, for offline calibration.

With no live camera to look at, the way to place the doorway line and sanity-check
detections/tracking is to run on a recorded clip and watch an annotated copy: bounding
boxes, track ids, foot points, the threshold line, and a running event log burned in.
This module owns all the drawing; it never decides anything.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from .config import DoorwayConfig
from .domain import Crossing, Frame, TrackedPerson

_GREEN = (0, 255, 0)
_RED = (0, 0, 255)
_BLUE = (255, 0, 0)
_WHITE = (255, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_doorway(image, doorway: DoorwayConfig) -> None:
    a = tuple(int(v) for v in doorway.line_a)
    b = tuple(int(v) for v in doorway.line_b)
    cv2.line(image, a, b, _RED, 2)


def draw_person(image, person: TrackedPerson) -> None:
    box = person.box
    cv2.rectangle(image, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), _GREEN, 2)
    foot_x, foot_y = (int(v) for v in box.foot)
    cv2.circle(image, (foot_x, foot_y), 4, _BLUE, -1)
    cv2.putText(
        image,
        f"id {person.track_id}",
        (int(box.x1), max(0, int(box.y1) - 6)),
        _FONT,
        0.5,
        _GREEN,
        1,
    )


class Annotator:
    """A per-frame observer that writes an annotated video to disk."""

    def __init__(self, doorway: DoorwayConfig, output: Path, fps: float = 10.0) -> None:
        self._doorway = doorway
        self._output = output
        self._fps = fps
        self._writer = None
        self._log: list[str] = []

    def __call__(
        self, frame: Frame, people: list[TrackedPerson], crossings: list[Crossing]
    ) -> None:
        image = frame.image.copy()
        draw_doorway(image, self._doorway)
        for person in people:
            draw_person(image, person)
        for crossing in crossings:
            self._log.append(f"{crossing.direction.value}: track {crossing.track_id}")
        self._draw_log(image)
        self._write(image)

    def _draw_log(self, image) -> None:
        for i, line in enumerate(self._log[-6:]):
            cv2.putText(image, line, (8, 18 + 18 * i), _FONT, 0.5, _WHITE, 1)

    def _write(self, image) -> None:
        if self._writer is None:
            height, width = image.shape[:2]
            self._output.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(self._output), fourcc, self._fps, (width, height))
        self._writer.write(image)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
