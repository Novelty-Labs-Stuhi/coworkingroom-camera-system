"""Identify people in one uploaded motion photo (the ESP32 camera path).

The board sends a JPEG whenever it sees motion. There is no video, so no tracking and
no in/out direction -- a motion photo is a doorway *sighting*. For each photo we detect
people and, when a face is clear enough, name them against the enrolled gallery; people
with no usable face are logged as "unknown". If no person box is found (a close-up that
fills the frame), we fall back to analysing the whole image.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detection import PersonDetector
from .domain import Box
from .recognition.face import FaceRecognizer

UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Sighting:
    name: str  # recognised name, or "unknown"
    clarity: float  # face clarity 0..1 (0 when no usable face)
    box: Box


class PhotoIdentifier:
    """Detect people in a photo and name them from the gallery."""

    def __init__(self, detector: PersonDetector, faces: FaceRecognizer) -> None:
        self._detector = detector
        self._faces = faces

    def identify(self, image: np.ndarray) -> list[Sighting]:
        height, width = image.shape[:2]
        boxes = self._detector.detect(image) or [Box(0, 0, width, height)]
        return [self._identify_box(image, box) for box in boxes]

    def _identify_box(self, image: np.ndarray, box: Box) -> Sighting:
        observation = self._faces.analyze(image, box)
        if observation is None:
            return Sighting(UNKNOWN, 0.0, box)
        match = self._faces.match(observation.embedding)
        name = match.name if match is not None else UNKNOWN
        return Sighting(name, observation.clarity, box)
