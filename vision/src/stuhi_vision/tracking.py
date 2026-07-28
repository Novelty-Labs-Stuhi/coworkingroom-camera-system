"""Person detection and tracking -- wraps Ultralytics YOLO + ByteTrack.

Gives each person in view an id that stays stable frame-to-frame, which is what the
doorway monitor needs to tell a crossing from ordinary loitering. Only this module
imports ultralytics; the model loads lazily.

``imgsz`` is the inference resolution, and it is the single biggest lever on speed:
measured on a CPU without AVX2, 640 costs ~440 ms per frame against ~200 ms at 320.
A doorway sees people close up and large in frame, so 320 is the sensible default --
raise it only if distant people are being missed.
"""

from __future__ import annotations

from .domain import Box, Frame, TrackedPerson

_PERSON_CLASS = 0  # COCO class id for "person"


class PersonTracker:
    """Detect and track people, returning stable-id boxes per frame."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        detection_conf: float = 0.4,
        imgsz: int = 320,
    ) -> None:
        self._model_path = model_path
        self._conf = detection_conf
        self._imgsz = imgsz
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self._model_path)

    def update(self, frame: Frame) -> list[TrackedPerson]:
        self._ensure_loaded()
        results = self._model.track(
            frame.image,
            persist=True,
            classes=[_PERSON_CLASS],
            conf=self._conf,
            imgsz=self._imgsz,
            verbose=False,
        )
        return list(self._to_people(results[0]))

    def _to_people(self, result) -> list[TrackedPerson]:
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []
        people = []
        ids = boxes.id.int().tolist()
        coords = boxes.xyxy.tolist()
        for track_id, (x1, y1, x2, y2) in zip(ids, coords, strict=True):
            people.append(TrackedPerson(track_id=track_id, box=Box(x1, y1, x2, y2)))
        return people
