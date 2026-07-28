"""Undo labels that are several names in one string.

The web UI used to pass its text field straight through as a single name, so labelling a
group of people produced gallery entries like ``a, yehor`` and ``art, ilar, hubertus`` --
each a person who does not exist, holding one reference, competing with the real names.

This removes those references and returns their clips to the pending list so they can be
labelled properly. It does not guess who they were: the whole point is that the mapping
from that string to real people is unknown.

    python tools/fix_composite_labels.py --config config.toml [--apply]

Without ``--apply`` it only reports.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--apply", action="store_true", help="actually undo them")
    args = parser.parse_args()

    from stuhi_vision import config as config_module
    from stuhi_vision.recognition.gallery import FaceGallery
    from stuhi_vision.review import ReviewQueue

    cfg = config_module.load(args.config)
    gallery = FaceGallery.load(cfg.paths.gallery_dir)
    review = ReviewQueue(cfg.paths.review_dir, gallery, cfg.paths.gallery_dir)

    composites = review.composite_labels()
    print(f"gallery before: {gallery.counts()}")
    print(f"\ncomposite labels found: {len(composites)}")
    for record in composites:
        print(f"  {record.sighting_id}  labelled {record.labelled_as!r}")

    if not composites:
        return
    if not args.apply:
        print("\nnothing changed; pass --apply to undo them")
        return

    for record in composites:
        outcome = review.unlabel(record.sighting_id)
        print(f"  {record.sighting_id}: {outcome.value}")

    print(f"\ngallery after: {gallery.counts()}")
    print(f"back in the pending list: {len(review.pending(limit=100))}")


if __name__ == "__main__":
    main()
