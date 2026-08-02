"""Run the detector and the doorframe pixels over recorded frames once, and keep the result.

Detection costs about 200 ms a frame on the deployment server, so replaying a set to try a
different rule costs five minutes of it -- which is enough friction that rules get chosen by
argument instead of by measurement. This runs the expensive part once and writes a JSON file
holding, per frame, the tracked boxes and whether the doorframe was covered, plus the frame
ranges of every coverage episode.

    python tools/cache_footage.py --dir ~/capture_lit --rotate --out ~/cache/capture_lit.json

Rule variants then score against the cache offline. The cache is tied to the zone it was
built with, because the coverage reading depends on it; the zone is recorded in the file so a
mismatch can be noticed rather than silently scored against the wrong box.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rotate", action="store_true", help="camera hangs upside down")
    parser.add_argument("--zone", nargs=4, type=float, default=[0.0, 0.0, 0.30, 1.0])
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--limit", type=int, default=0, help="frames to read (0 = all)")
    return parser.parse_args()


def _boxes(result) -> list[dict]:
    """One frame's tracked people as plain data, or nothing when the tracker lost them."""
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return []
    return [
        {"track": int(track_id), "box": [float(v) for v in xyxy]}
        for track_id, xyxy in zip(boxes.id.int().tolist(), boxes.xyxy.tolist(), strict=True)
    ]


def _read(path, rotate: bool):
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        return None
    return cv2.rotate(image, cv2.ROTATE_180) if rotate else image


def _cache(args) -> dict:
    from ultralytics import YOLO

    from stuhi_vision.occlusion import CoverageConfig, Occlusion

    model = YOLO("yolov8n.pt")
    occlusion = Occlusion(tuple(args.zone), CoverageConfig())

    paths = sorted(Path(args.dir).expanduser().glob("*.jpg"))
    if args.limit:
        paths = paths[: args.limit]

    frames: list[dict] = []
    episodes: list[dict] = []
    open_at: int | None = None
    size = None

    for index, path in enumerate(paths):
        image = _read(path, args.rotate)
        if image is None:
            continue
        size = list(image.shape[:2])

        # Pixels before the detector, as the live pipeline does it.
        finished = occlusion.update(image)
        covered = occlusion.busy
        if covered and open_at is None:
            open_at = index
        if finished is not None and open_at is not None:
            # The episode ended on the frame before this one: `update` reports it once the
            # box has cleared, so the last covered frame is the previous index.
            episodes.append(
                {
                    "first": open_at,
                    "last": index - 1,
                    "frames": finished.frames,
                    "slices": finished.slices,
                    "lag": finished.lag,
                }
            )
            open_at = None
        elif not covered:
            open_at = None

        result = model.track(
            image, persist=True, classes=[0], conf=args.conf, imgsz=args.imgsz, verbose=False
        )[0]
        frames.append({"index": index, "covered": covered, "people": _boxes(result)})

    return {
        "dir": str(Path(args.dir).expanduser()),
        "rotate": args.rotate,
        "zone": list(args.zone),
        "conf": args.conf,
        "imgsz": args.imgsz,
        "size": size,          # [height, width]
        "frames": frames,
        "episodes": episodes,
    }


def main() -> None:
    args = _arguments()
    cache = _cache(args)
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cache))
    covered = sum(1 for f in cache["frames"] if f["covered"])
    tracks = {p["track"] for f in cache["frames"] for p in f["people"]}
    print(
        f"{len(cache['frames'])} frames -> {out}\n"
        f"  covered frames: {covered}\n"
        f"  coverage episodes: {len(cache['episodes'])}\n"
        f"  distinct track ids: {len(tracks)}"
    )


if __name__ == "__main__":
    main()
