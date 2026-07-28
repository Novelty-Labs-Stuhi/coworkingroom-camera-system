"""Measure how fast this machine can actually process a camera stream.

Answers the question that decides the architecture: can the server run detection and face
recognition on every frame, or must cheap motion gating come first? Reports a per-stage
breakdown, because the fix differs depending on which stage dominates.

    python tools/benchmark_stream.py --url http://192.168.8.229:81/stream --frames 40

Also reports how tall the person crops are and how often a face was found in them, which
is the evidence for whether the camera is mounted close enough to recognise anyone.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@dataclass(slots=True)
class Timings:
    """Per-stage wall-clock samples, plus what was actually seen."""

    read: list[float] = field(default_factory=list)
    detect: list[float] = field(default_factory=list)
    face: list[float] = field(default_factory=list)
    people: int = 0
    faces_found: int = 0
    crop_heights: list[float] = field(default_factory=list)


def _measure(capture, tracker, faces, frame_limit: int) -> tuple[Timings, float]:
    """Process up to ``frame_limit`` frames, returning timings and elapsed seconds."""
    from stuhi_vision.domain import Frame

    timings = Timings()
    started = time.perf_counter()
    for index in range(frame_limit):
        mark = time.perf_counter()
        ok, image = capture.read()
        if not ok:
            print("stream ended early")
            break
        timings.read.append(time.perf_counter() - mark)

        frame = Frame(timestamp=time.time(), image=image)
        mark = time.perf_counter()
        people = tracker.update(frame)
        timings.detect.append(time.perf_counter() - mark)
        timings.people += len(people)

        for person in people:
            mark = time.perf_counter()
            observation = faces.analyze(frame.image, person.box)
            timings.face.append(time.perf_counter() - mark)
            timings.crop_heights.append(person.box.y2 - person.box.y1)
            if observation is not None:
                timings.faces_found += 1

        if index == 0:
            started = time.perf_counter()  # exclude one-off model loading from the rate
    return timings, time.perf_counter() - started


def _report(label: str, samples: list[float]) -> None:
    if not samples:
        print(f"{label:<24} no samples")
        return
    print(
        f"{label:<24} mean {statistics.mean(samples) * 1000:7.1f} ms"
        f"   worst {max(samples) * 1000:7.1f} ms"
    )


def _summarise(timings: Timings, elapsed: float) -> None:
    processed = max(len(timings.read) - 1, 1)
    print()
    _report("frame read/decode", timings.read)
    _report("yolo + bytetrack", timings.detect)
    _report("insightface per person", timings.face)
    print()
    print(f"frames processed         {processed}")
    print(f"end-to-end rate          {processed / elapsed:.2f} fps")
    print(f"person detections        {timings.people}")
    print(f"faces found              {timings.faces_found} of {timings.people} person crops")
    if timings.crop_heights:
        median = statistics.median(timings.crop_heights)
        print(f"person crop height       median {median:.0f} px")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="MJPEG stream URL")
    parser.add_argument("--frames", type=int, default=40, help="frames to process")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO weights path")
    args = parser.parse_args()

    import cv2

    from stuhi_vision.recognition.face import FaceRecognizer
    from stuhi_vision.recognition.gallery import FaceGallery
    from stuhi_vision.tracking import PersonTracker

    capture = cv2.VideoCapture(args.url)
    if not capture.isOpened():
        raise SystemExit(f"could not open {args.url}")

    print("loading models (first frame includes one-off model init)...")
    tracker = PersonTracker(model_path=args.model)
    faces = FaceRecognizer(FaceGallery(), match_threshold=0.4)
    try:
        timings, elapsed = _measure(capture, tracker, faces, args.frames)
    finally:
        capture.release()
    _summarise(timings, elapsed)


if __name__ == "__main__":
    main()
