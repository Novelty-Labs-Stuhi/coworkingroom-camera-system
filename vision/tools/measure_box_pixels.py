"""What pixel change over the doorframe box makes of recorded frames.

Runs the shipping :class:`~stuhi_vision.occlusion.Occlusion` over a directory of frames and
prints one line per episode: how long the box was covered, how much of it, and which way the
covering travelled. No models, so it is fast and can be re-run while tuning the thresholds --
and it exercises the same code the pipeline does, so a result here means something there.

The point of measuring rather than reasoning: the covering thing's shape changes as it comes
closer, and every intuition about how that reads has so far been wrong once tested.

    python tools/measure_box_pixels.py --dir ~/capture_lit --rotate --zone 0 0 0.30 1
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
    parser.add_argument("--slices", type=int, default=5)
    parser.add_argument("--covered", type=float, default=0.25)
    parser.add_argument("--difference", type=int, default=25)
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--min-lag", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def _reading(episode) -> str:
    """Which way the slices lit up, if they lit in an order at all."""
    if not episode.swept:
        return "no order -- stood on the doorframe"
    return "leftwards" if episode.lag < 0 else "rightwards"


def main() -> None:
    import cv2

    from stuhi_vision.occlusion import CoverageConfig, Occlusion

    args = _arguments()
    occlusion = Occlusion(
        zone=(args.zone[0], args.zone[1], args.zone[2], args.zone[3]),
        config=CoverageConfig(
            slices=args.slices,
            covered=args.covered,
            difference=args.difference,
            min_frames=args.min_frames,
            min_lag=args.min_lag,
        ),
    )

    paths = sorted(Path(args.dir).expanduser().glob("*.jpg"))
    if args.limit:
        paths = paths[: args.limit]

    print("  frame                  frames  slices  peak      lag  reading")
    episodes = 0
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        if args.rotate:
            image = cv2.rotate(image, cv2.ROTATE_180)
        episode = occlusion.update(image)
        if episode is None:
            continue
        episodes += 1
        print(
            f"  {path.name:<22} {episode.frames:>5}  {episode.slices:>6}  "
            f"{episode.peak:>4.2f}  {episode.lag:>+7.2f}  {_reading(episode)}"
        )

    print(f"\nframes: {len(paths)}   episodes over the box: {episodes}")


if __name__ == "__main__":
    main()
