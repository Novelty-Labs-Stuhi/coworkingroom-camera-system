"""Score the ordering rule against a cached run, beside the rules already in the codebase.

The rule is validated on recorded frames rather than by walking down a corridor, because a
negative result from a real walk says only "nothing happened" and costs somebody two minutes.
Uses the real :class:`~stuhi_vision.ordering.OrderingRule`, not a reimplementation of it -- a
test of a copy proves nothing about what runs.

    python tools/score_ordering.py --cache ~/cache/capture_lit.json --passing-means out
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--passing-means", default="out")
    parser.add_argument("--min-height", type=float, default=0.35)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--lost-after", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = _arguments()

    from stuhi_vision.domain import Box, Direction, TrackedPerson
    from stuhi_vision.ordering import Episode, OrderingRule
    from stuhi_vision.threshold import ThresholdConfig

    cache = json.loads(Path(args.cache).expanduser().read_text())
    height, width = cache["size"]
    rule = OrderingRule(
        ThresholdConfig(
            zone=tuple(cache["zone"]),
            min_height=args.min_height,
            passing_means=Direction(args.passing_means),
        ),
        window=args.window,
        lost_after=args.lost_after,
    )

    # An episode is handed over on the frame after its last covered one, which is when the
    # live pipeline learns of it: coverage is only known to have ended once it has.
    ending = {}
    for episode in cache["episodes"]:
        ending.setdefault(episode["last"] + 1, []).append(
            Episode(episode["first"], episode["last"])
        )

    print(f'{Path(cache["dir"]).name}: {len(cache["frames"])} frames, '
          f'{len(cache["episodes"])} episodes, passing_means={args.passing_means}')
    print()
    verdicts = []
    for entry in cache["frames"]:
        people = [
            TrackedPerson(track_id=p["track"], box=Box(*p["box"])) for p in entry["people"]
        ]
        for episode in ending.get(entry["index"], [None]):
            verdicts += rule.observe(people, width, height, episode)
            people = []   # the frame's people are counted once, not once per episode
    for _ in range(args.lost_after + 5):
        verdicts += rule.observe([], width, height, None)

    for verdict in verdicts:
        print(f"  {verdict.readable}")
    resolved = [v for v in verdicts if v.direction is not None]
    ins = sum(1 for v in resolved if v.direction.value == "in")
    print()
    print(f"  resolved {len(resolved)} of {len(verdicts)} finished tracks"
          f"  ({ins} in, {len(resolved) - ins} out)")


if __name__ == "__main__":
    main()
