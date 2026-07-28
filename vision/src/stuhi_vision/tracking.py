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

from collections.abc import Callable
from typing import Protocol

from .domain import Box, Frame, TrackedPerson
from .gating import MotionGate
from .region import Region

_PERSON_CLASS = 0  # COCO class id for "person"

# Builds the detection region once the frame size is known (width, height).
RegionBuilder = Callable[[int, int], Region]


class Detector(Protocol):
    """Anything that turns a frame into tracked people."""

    def update(self, frame: Frame) -> list[TrackedPerson]: ...


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


class GatedTracker:
    """A tracker wrapped in the two cost controls that make a slow machine keep up.

    Both are skips rather than approximations, and both are invisible to callers:

    * an unchanged frame cannot contain a crossing, so the motion gate returns no people
      without running the model at all -- an empty doorway costs about a millisecond;
    * detection runs only on the crop containing the doorway, and boxes are translated
      back to full-frame coordinates before they are returned.

    Omit both collaborators and this is a pass-through.
    """

    def __init__(
        self,
        tracker: Detector,
        gate: MotionGate | None = None,
        region_builder: RegionBuilder | None = None,
    ) -> None:
        self._tracker = tracker
        self._gate = gate
        self._region_builder = region_builder
        self._region: Region | None = None

    def update(self, frame: Frame) -> list[TrackedPerson]:
        if self._gate is not None and not self._gate.is_active(frame.image):
            return []
        region = self._ensure_region(frame)
        if region is None:
            return self._tracker.update(frame)
        cropped = Frame(timestamp=frame.timestamp, image=region.crop(frame.image))
        return region.people_to_frame(self._tracker.update(cropped))

    def _ensure_region(self, frame: Frame) -> Region | None:
        if self._region_builder is None:
            return None
        if self._region is None:
            height, width = frame.image.shape[:2]
            self._region = self._region_builder(width, height)
        return self._region
