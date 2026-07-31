"""Do the slices of the doorframe box light up in order, and are there enough frames for it?

Runs the shipping :class:`~stuhi_vision.occlusion.Occlusion` over recorded frames and prints,
for every episode, the coverage of each vertical slice frame by frame. That timeline answers
two questions at once:

* **direction** -- do the slices light in sequence, left to right or right to left;
* **frame rate** -- is there more than one frame of coverage at all. A person crosses a narrow
  box in a fraction of a second, so at a few frames per second an order may simply not exist,
  and no threshold can conjure one. The number of frames per episode is the measurement.

No models involved, so it re-runs in seconds while thresholds change, and it exercises the same
code the pipeline does.

    python tools/measure_box_slices.py --dir ~/capture_lit --rotate --zone 0 0 0.30 1
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
    parser.add_argument("--covered", type=float, default=0.25)
    parser.add_argument("--difference", type=int, default=25)
    parser.add_argument("--slices", type=int, default=4)
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--travel-margin", type=float, default=0.5, help="slices, to call it")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def _reading(travelled: float, margin: float) -> str:
    if travelled <= -margin:
        return "towards the first slice"
    if travelled >= margin:
        return "towards the last slice"
    return "no clear direction"


def _rate(paths: list[Path]) -> str:
    """The capture rate, from the files' own timestamps."""
    if len(paths) < 2:
        return "unknown"
    stamps = sorted(path.stat().st_mtime for path in paths)
    span = stamps[-1] - stamps[0]
    if span <= 0:
        return "unknown"
    return f"{(len(stamps) - 1) / span:.1f} fps over {span:.0f}s"


def main() -> None:
    import cv2

    from stuhi_vision.occlusion import CoverageConfig, Occlusion

    args = _arguments()
    occlusion = Occlusion(
        zone=(args.zone[0], args.zone[1], args.zone[2], args.zone[3]),
        config=CoverageConfig(
            covered=args.covered,
            difference=args.difference,
            slices=args.slices,
            min_frames=args.min_frames,
        ),
    )

    paths = sorted(Path(args.dir).expanduser().glob("*.jpg"))
    if args.limit:
        paths = paths[: args.limit]

    episodes = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        if args.rotate:
            image = cv2.rotate(image, cv2.ROTATE_180)
        episode = occlusion.update(image)
        if episode is not None:
            episodes.append((path.name, episode))

    print(f"frames: {len(paths)}   capture rate: {_rate(paths)}   episodes: {len(episodes)}")
    for name, episode in episodes:
        print(
            f"\n  {name}  {episode.frames} frames, peak {episode.peak:.2f}, "
            f"moved {episode.travelled:+.1f} slices "
            f"-> {_reading(episode.travelled, args.travel_margin)}"
        )
        print(episode.picture)


if __name__ == "__main__":
    main()
