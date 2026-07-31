"""Rebuild the recorded groups from the gaps between crossings.

The groups on disk were assigned by a rule that ended a burst only when every clip had
finished -- and a clip's completion counter resets whenever anybody is in view, so in an
occupied room every crossing joined the same group. The labelling page ended up offering a
clip as "1 of 44 together", asking for forty-four names in crossing order for passages minutes
apart. Fixing the live rule cannot repair what is already recorded, so this recomputes it from
the one thing recorded honestly: when each crossing happened.

    python tools/regroup_sightings.py --review data/review --gallery gallery
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    from stuhi_vision.recognition.gallery import FaceGallery
    from stuhi_vision.review import ReviewQueue

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", default="data/review")
    parser.add_argument("--gallery", default="gallery")
    parser.add_argument("--gap", type=float, default=3.0, help="seconds that separate groups")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gallery_dir = Path(args.gallery)
    review = ReviewQueue(Path(args.review), FaceGallery.load(gallery_dir), gallery_dir)

    print("before:", _spread(review))
    if args.dry_run:
        print("(dry run: nothing written)")
        return
    print(f"regrouped {review.regroup(gap_seconds=args.gap)} sighting(s)")
    print("after: ", _spread(review))


def _spread(review) -> str:
    """How big the groups are, which is the thing that was wrong."""
    sizes = Counter(len(members) for members in review.groups().values())
    return ", ".join(f"{size} together x{count}" for size, count in sorted(sizes.items()))


if __name__ == "__main__":
    main()
