"""Person detection for a single still photo (no tracking).

The stream pipeline uses PersonTracker (detection + ByteTrack). The motion-photo path
only ever has one frame at a time, so it needs detection alone -- this is a lighter
detect-only wrapper over the same YOLO model. Loaded lazily.
"""

from __future__ import annotations

from .domain import Box

_PERSON_CLASS = 0  # COCO class id for "person"


class PersonDetector:
    """Detect people in one image, returning their boxes."""

    def __init__(self, model_path: str = "yolov8n.pt", detection_conf: float = 0.4) -> None:
        self._model_path = model_path
        self._conf = detection_conf
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self._model_path)

    def detect(self, image) -> list[Box]:
        self._ensure_loaded()
        result = self._model.predict(
            image, classes=[_PERSON_CLASS], conf=self._conf, verbose=False
        )[0]
        boxes = result.boxes
        if boxes is None:
            return []
        return [Box(x1, y1, x2, y2) for x1, y1, x2, y2 in boxes.xyxy.tolist()]
