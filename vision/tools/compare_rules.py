"""Replay recorded frames through several direction rules at once and compare them.

Every rule is asked about the *same* tracks and the *same* doorframe coverage, so a
difference in the verdicts is a difference between the rules rather than between two runs of
a detector that does not repeat itself exactly. Detection is the expensive part -- around
200 ms a frame on the deployment server -- so it runs once and its output is handed to every
monitor.

    python tools/compare_rules.py --dir ~/walkthrough --rotate \
        --zone 0 0 0.30 1 --edge left --passing-means in

Prints, per rule, how many passages it found and which way; then a per-track table with the
evidence each rule keys on, so a disagreement can be traced to the track that caused it.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# The rules that judge a finished track. "approach" is left out: it is for the room-facing
# camera, where nobody crosses the view sideways.
RULES = ("covering", "travel", "edge")


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
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--travel-margin", type=float, default=0.08)
    parser.add_argument("--lost-after", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="frames to read (0 = all)")
    return parser.parse_args()


class _Coverage:
    """The doorframe's coverage for the frame being processed, shared by every monitor."""

    def __init__(self) -> None:
        self.covered = False

    def __call__(self) -> bool:
        return self.covered


def _people(result):
    """One frame's detections as tracked people, or nothing when the tracker lost them."""
    from stuhi_vision.domain import Box, TrackedPerson

    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return []
    return [
        TrackedPerson(track_id=track_id, box=Box(x1, y1, x2, y2))
        for track_id, (x1, y1, x2, y2) in zip(
            boxes.id.int().tolist(), boxes.xyxy.tolist(), strict=True
        )
    ]


def _build(args):
    """A monitor per rule, plus the coverage they all read and the touches they report."""
    from stuhi_vision.domain import Direction
    from stuhi_vision.threshold import ThresholdConfig, ThresholdMonitor

    base = ThresholdConfig(
        zone=tuple(args.zone),
        edge=args.edge,
        margin=args.margin,
        passing_means=Direction(args.passing_means),
        min_height=args.min_height,
        travel_margin=args.travel_margin,
        lost_after=args.lost_after,
    )
    coverage = _Coverage()
    touches: dict[str, list] = {rule: [] for rule in RULES}
    monitors = {
        rule: ThresholdMonitor(
            replace(base, discriminator=rule),
            report=touches[rule].append,
            covered=coverage,
        )
        for rule in RULES
    }
    return monitors, coverage, touches


def _read(path, rotate: bool):
    """One recorded frame, turned the right way up."""
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        return None
    return cv2.rotate(image, cv2.ROTATE_180) if rotate else image


def _judge(monitors, crossings, people, frame, index: int) -> None:
    """Hand one frame's people to every rule, collecting whatever each decides."""
    for rule, monitor in monitors.items():
        crossings[rule] += [(index, c) for c in monitor.update(people, frame)]


def _replay(args, monitors, coverage):
    """Every frame through the detector once, then through each rule."""
    from ultralytics import YOLO

    from stuhi_vision.domain import Frame
    from stuhi_vision.occlusion import CoverageConfig, Occlusion

    model = YOLO("yolov8n.pt")
    occlusion = Occlusion(tuple(args.zone), CoverageConfig())

    paths = sorted(Path(args.dir).expanduser().glob("*.jpg"))
    if args.limit:
        paths = paths[: args.limit]

    crossings: dict[str, list] = {rule: [] for rule in monitors}
    frame = None
    for index, path in enumerate(paths):
        image = _read(path, args.rotate)
        if image is None:
            continue

        # The pixels first, exactly as the live pipeline does it: Attention reads the box
        # before the detector runs, so the coverage a monitor sees belongs to this frame.
        occlusion.update(image)
        coverage.covered = occlusion.busy

        frame = Frame(timestamp=float(index), image=image)
        result = model.track(
            image, persist=True, classes=[0], conf=args.conf, imgsz=args.imgsz, verbose=False
        )[0]
        _judge(monitors, crossings, _people(result), frame, index)

    # Let every open track finish, so somebody still in view at the end is judged too. The
    # coverage is held at its final reading rather than cleared: inventing a change here
    # would hand the covering rule a verdict the footage never showed it.
    for extra in range(args.lost_after + 5):
        if frame is None:
            break
        _judge(monitors, crossings, [], frame, len(paths) + extra)
    return paths, crossings


def _report(paths, crossings, touches) -> None:
    print(f"frames: {len(paths)}")
    print()
    print("rule       passages    in   out")
    for rule in RULES:
        found = [c for _, c in crossings[rule]]
        ins = sum(1 for c in found if c.direction.value == "in")
        print(f"  {rule:<9} {len(found):>8} {ins:>5} {len(found) - ins:>5}")

    print()
    print("tracks that reached the doorframe, and what each rule made of them:")
    print("  track  frames  tallest  travelled  doorframe        covering  travel    edge")
    seen = {}
    for rule in RULES:
        for order, touch in enumerate(touches[rule]):
            seen.setdefault(order, {})[rule] = touch
    for order in sorted(seen):
        row = seen[order]
        any_touch = next(iter(row.values()))
        state = f"{'covered' if any_touch.covered_first else 'clear'}->" + (
            "covered" if any_touch.covered_last else "clear"
        )
        verdicts = "  ".join(
            f"{(row[rule].direction.value if row[rule].direction else '-'):<8}" for rule in RULES
        )
        print(
            f"  {order:>5}  {any_touch.frames:>6}  {any_touch.tallest:>7.2f}  "
            f"{any_touch.travelled:>+9.2f}  {state:<15}  {verdicts}"
        )

    print()
    disagreed = [
        order
        for order, row in seen.items()
        if len({row[rule].direction for rule in RULES if rule in row}) > 1
    ]
    print(f"tracks where the rules disagree: {len(disagreed)} of {len(seen)}")


def main() -> None:
    args = _arguments()
    monitors, coverage, touches = _build(args)
    paths, crossings = _replay(args, monitors, coverage)
    _report(paths, crossings, touches)


if __name__ == "__main__":
    main()
