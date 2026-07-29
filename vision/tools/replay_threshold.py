"""Replay recorded frames through the doorframe rule and report what it decides.

A rule that depends on where a track begins and ends cannot be tuned by watching the live
system: every attempt costs somebody a walk down the corridor, and a negative result says
only "nothing happened" without saying which stage lost it. Recorded frames can be replayed
as often as needed, with parameters changed between runs.

    python tools/replay_threshold.py --dir ~/capture_lit --rotate --edge left \
        --zone 0 0 0.30 1 --min-height 0.35 --passing-means in

Reports every track's first and last box, whether it touched the zone, and what the rule
concluded -- so a pass that produced nothing can be traced to the reason.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--rotate", action="store_true", help="camera hangs upside down")
    parser.add_argument("--zone", nargs=4, type=float, default=[0.0, 0.0, 0.30, 1.0])
    parser.add_argument("--edge", default="left")
    parser.add_argument("--margin", type=float, default=0.12)
    parser.add_argument("--min-height", type=float, default=0.35)
    parser.add_argument("--passing-means", default="in")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--lost-after", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="frames to read (0 = all)")
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    from stuhi_vision.domain import Box, Direction, Frame, TrackedPerson
    from stuhi_vision.threshold import ThresholdConfig, ThresholdMonitor

    config = ThresholdConfig(
        zone=tuple(args.zone),
        edge=args.edge,
        margin=args.margin,
        passing_means=Direction(args.passing_means),
        min_height=args.min_height,
        lost_after=args.lost_after,
    )
    monitor = ThresholdMonitor(config)
    model = YOLO("yolov8n.pt")

    paths = sorted(Path(args.dir).expanduser().glob("*.jpg"))
    if args.limit:
        paths = paths[: args.limit]

    seen: dict[int, dict] = {}
    crossings = []
    frame = None
    for index, path in enumerate(paths):
        image = cv2.imread(str(path))
        if image is None:
            continue
        if args.rotate:
            image = cv2.rotate(image, cv2.ROTATE_180)
        height, width = image.shape[:2]
        frame = Frame(timestamp=float(index), image=image)

        result = model.track(
            image, persist=True, classes=[0], conf=args.conf, imgsz=320, verbose=False
        )[0]
        boxes = result.boxes
        people = []
        if boxes is not None and boxes.id is not None:
            for track_id, (x1, y1, x2, y2) in zip(
                boxes.id.int().tolist(), boxes.xyxy.tolist(), strict=True
            ):
                people.append(TrackedPerson(track_id=track_id, box=Box(x1, y1, x2, y2)))
                record = seen.setdefault(
                    track_id, {"first": None, "last": None, "frames": 0, "tallest": 0.0}
                )
                relative = (x1 / width, y1 / height, x2 / width, y2 / height)
                record["first"] = record["first"] or relative
                record["last"] = relative
                record["frames"] += 1
                record["tallest"] = max(record["tallest"], (y2 - y1) / height)

        crossings += [(index, c) for c in monitor.update(people, frame)]

    # Let every open track finish, so a person still in view at the end is still judged.
    for extra in range(args.lost_after + 5):
        if frame is not None:
            crossings += [(len(paths) + extra, c) for c in monitor.update([], frame)]

    print(f"frames: {len(paths)}   tracks: {len(seen)}\n")
    print("  track  frames  tallest  first(l,t,r,b)              last(l,t,r,b)")
    for track_id, record in sorted(seen.items()):
        first = " ".join(f"{v:.2f}" for v in record["first"])
        last = " ".join(f"{v:.2f}" for v in record["last"])
        print(
            f"  {track_id:>5}  {record['frames']:>6}  {record['tallest']:>7.2f}  "
            f"({first})  ({last})"
        )

    print(f"\ncrossings: {len(crossings)}")
    for index, crossing in crossings:
        print(f"  frame {index}: {crossing.direction.value} (track {crossing.track_id})")
    if not crossings:
        print("  none -- compare the boxes above against zone/edge/min_height")


if __name__ == "__main__":
    main()
