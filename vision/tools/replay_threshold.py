"""Replay recorded frames through the doorframe rule and report what it decides.

A rule that keys on where a track begins and ends cannot be tuned against the live system:
every attempt costs somebody a walk down the corridor, and a negative result says only
"nothing happened" rather than which stage lost it. Recorded frames can be replayed as often
as needed, with the parameters changed between runs.

    python tools/replay_threshold.py --dir ~/capture_lit --rotate \
        --zone 0 0 0.30 1 --edge left --min-height 0.35 --passing-means in

Prints every track's first and last box beside the verdict, so a passage that produced
nothing can be traced to the reason rather than guessed at.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--rotate", action="store_true", help="camera hangs upside down")
    parser.add_argument("--zone", nargs=4, type=float, default=[0.0, 0.0, 0.30, 1.0])
    parser.add_argument("--edge", default="left")
    parser.add_argument("--margin", type=float, default=0.12)
    parser.add_argument("--min-height", type=float, default=0.35)
    parser.add_argument("--passing-means", default="in")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--discriminator", default="edge", choices=["edge", "approach"])
    parser.add_argument("--growth-margin", type=float, default=0.12)
    parser.add_argument("--lost-after", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="frames to read (0 = all)")
    return parser.parse_args()


def _track_people(result, width: int, height: int, seen: dict) -> list:
    """Turn one frame's detections into tracked people, recording each track's extent."""
    from stuhi_vision.domain import Box, TrackedPerson

    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return []

    people = []
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
    return people


def _replay(args: argparse.Namespace, monitor, model) -> tuple[list, dict, list]:
    """Run every frame through the rule, returning what was seen and what it decided."""
    import cv2

    from stuhi_vision.domain import Frame

    paths = sorted(Path(args.dir).expanduser().glob("*.jpg"))
    if args.limit:
        paths = paths[: args.limit]

    seen: dict[int, dict] = {}
    crossings: list = []
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
        people = _track_people(result, width, height, seen)
        crossings += [(index, crossing) for crossing in monitor.update(people, frame)]

    # Let every open track finish, so a person still in view at the end is judged too.
    for extra in range(args.lost_after + 5):
        if frame is not None:
            crossings += [(len(paths) + extra, crossing) for crossing in monitor.update([], frame)]
    return paths, seen, crossings


def _report(paths: list, seen: dict, crossings: list) -> None:
    print(f"frames: {len(paths)}   tracks: {len(seen)}")
    print()
    print("  track  frames  tallest  first(l,t,r,b)             last(l,t,r,b)")
    for track_id, record in sorted(seen.items()):
        first = " ".join(f"{value:.2f}" for value in record["first"])
        last = " ".join(f"{value:.2f}" for value in record["last"])
        print(
            f"  {track_id:>5}  {record['frames']:>6}  {record['tallest']:>7.2f}  "
            f"({first})  ({last})"
        )
    print()
    print(f"crossings: {len(crossings)}")
    for index, crossing in crossings:
        print(f"  frame {index}: {crossing.direction.value} (track {crossing.track_id})")
    if not crossings:
        print("  none -- compare the boxes above against zone / edge / min_height")


def main() -> None:
    args = _arguments()

    from ultralytics import YOLO

    from stuhi_vision.domain import Direction
    from stuhi_vision.threshold import ThresholdConfig, ThresholdMonitor

    config = ThresholdConfig(
        zone=(args.zone[0], args.zone[1], args.zone[2], args.zone[3]),
        edge=args.edge,
        margin=args.margin,
        passing_means=Direction(args.passing_means),
        min_height=args.min_height,
        discriminator=args.discriminator,
        growth_margin=args.growth_margin,
        lost_after=args.lost_after,
    )
    _report(*_replay(args, ThresholdMonitor(config), YOLO("yolov8n.pt")))


if __name__ == "__main__":
    main()
