"""Does the detector lose people when they come close to the lens?

The doorframe rule depends on a track existing at all. If YOLO drops a person once they fill
the frame -- which is out of distribution for a person detector trained on whole bodies -- then
a passage is missed before any rule gets to judge it, and no threshold can recover it.

Reports, per frame: how many people were detected, the tallest box as a fraction of the frame,
and its confidence. Then the dropouts: runs of frames with nothing detected that sit *between*
frames where somebody was close. Those are the ones that would cost a count.

    python close_range.py --dir ~/capture_lit --rotate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CLOSE = 0.55  # a box this tall means the person is near the lens


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def _measure(args: argparse.Namespace) -> list[dict]:
    import cv2
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    paths = sorted(Path(args.dir).expanduser().glob("*.jpg"))
    if args.limit:
        paths = paths[: args.limit]

    readings = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        if args.rotate:
            image = cv2.rotate(image, cv2.ROTATE_180)
        height = image.shape[0]
        result = model.track(
            image, persist=True, classes=[0], conf=args.conf, imgsz=args.imgsz, verbose=False
        )[0]
        boxes = result.boxes
        tallest, best = 0.0, 0.0
        count = 0
        if boxes is not None and len(boxes):
            count = len(boxes)
            for (_, y1, _, y2), conf in zip(
                boxes.xyxy.tolist(), boxes.conf.tolist(), strict=True
            ):
                fraction = (y2 - y1) / height
                if fraction > tallest:
                    tallest, best = fraction, conf
        readings.append({"name": path.name, "count": count, "tallest": tallest, "conf": best})
    return readings


def _dropouts(readings: list[dict]) -> list[tuple[int, int, float, float]]:
    """Runs of empty frames flanked by frames where somebody was close to the lens."""
    runs = []
    start = None
    for index, reading in enumerate(readings):
        if reading["count"] == 0:
            start = index if start is None else start
            continue
        if start is not None:
            before = readings[start - 1]["tallest"] if start else 0.0
            after = reading["tallest"]
            if before >= CLOSE or after >= CLOSE:
                runs.append((start, index - start, before, after))
            start = None
    return runs


def main() -> None:
    args = _arguments()
    readings = _measure(args)
    seen = [r for r in readings if r["count"]]
    close = [r for r in seen if r["tallest"] >= CLOSE]

    print(f"frames: {len(readings)}   with a person: {len(seen)}   close (>={CLOSE}): {len(close)}")
    if close:
        confidences = sorted(r["conf"] for r in close)
        middle = confidences[len(confidences) // 2]
        print(f"close-range confidence: min {confidences[0]:.2f}  median {middle:.2f}")

    runs = _dropouts(readings)
    print(f"\ndropouts next to a close person: {len(runs)}")
    for start, length, before, after in runs[:20]:
        print(
            f"  {readings[start]['name']}: {length} empty frames "
            f"(tallest before {before:.2f}, after {after:.2f})"
        )

    print("\ntallest box per frame, where anything was seen:")
    for reading in seen[:80]:
        print(f"  {reading['name']}  n={reading['count']}  h={reading['tallest']:.2f}  "
              f"conf={reading['conf']:.2f}")


if __name__ == "__main__":
    main()
