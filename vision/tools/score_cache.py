"""Score direction rules against a cached run, and say which evidence each track carried.

Answers the question a rule comparison cannot: for a track the rule refused, *was the
doorframe covered at all while that person was present?* If it was, the refusal is the rule
missing a real passage; if it was not, the person genuinely never went through. The two need
opposite responses, and telling them apart is the whole point of keeping the cache.

    python tools/score_cache.py --cache ~/cache/capture_lit.json --edge left \
        --passing-means in

Runs entirely off the cache, so trying another rule costs a second rather than a five-minute
detection pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

RULES = ("covering", "travel", "edge")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--edge", default="left")
    parser.add_argument("--margin", type=float, default=0.12)
    parser.add_argument("--min-height", type=float, default=0.35)
    parser.add_argument("--passing-means", default="in")
    parser.add_argument("--travel-margin", type=float, default=0.08)
    parser.add_argument("--lost-after", type=int, default=6)
    return parser.parse_args()


class _Coverage:
    def __init__(self) -> None:
        self.covered = False

    def __call__(self) -> bool:
        return self.covered


def _extent(cache: dict) -> dict[int, dict]:
    """Each track's frame span, and whether the doorframe was ever covered while it ran."""
    spans: dict[int, dict] = {}
    for frame in cache["frames"]:
        for person in frame["people"]:
            span = spans.setdefault(
                person["track"], {"first": frame["index"], "last": frame["index"], "seen": 0}
            )
            span["last"] = frame["index"]
            span["seen"] += 1
    for span in spans.values():
        overlapping = [
            episode
            for episode in cache["episodes"]
            if episode["last"] >= span["first"] and episode["first"] <= span["last"]
        ]
        span["episodes"] = overlapping
        # Did an episode still have frames to run after this track was last seen? That is
        # exactly the case a rule reading the endpoint cannot see: the person vanished from
        # the detector while the doorframe was still covered.
        span["episode_outlived_track"] = any(
            episode["last"] > span["last"] for episode in overlapping
        )
    return spans


def _replay(cache: dict, args):
    """Feed the cached frames through every rule, exactly as the live monitor would."""
    import numpy as np

    from stuhi_vision.domain import Box, Direction, Frame, TrackedPerson
    from stuhi_vision.threshold import ThresholdConfig, ThresholdMonitor

    height, width = cache["size"]
    base = ThresholdConfig(
        zone=tuple(cache["zone"]),
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
            replace(base, discriminator=rule), report=touches[rule].append, covered=coverage
        )
        for rule in RULES
    }

    # The image is only measured for its shape, so a stub of the right size is enough and the
    # cache does not have to carry pixels.
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    crossings: dict[str, list] = {rule: [] for rule in RULES}
    for entry in cache["frames"]:
        coverage.covered = entry["covered"]
        people = [
            TrackedPerson(track_id=p["track"], box=Box(*p["box"])) for p in entry["people"]
        ]
        frame = Frame(timestamp=float(entry["index"]), image=blank)
        for rule, monitor in monitors.items():
            crossings[rule] += [c for c in monitor.update(people, frame)]

    last = Frame(timestamp=float(len(cache["frames"])), image=blank)
    for _ in range(args.lost_after + 5):
        for rule, monitor in monitors.items():
            crossings[rule] += [c for c in monitor.update([], last)]
    return crossings, touches


def _report(cache: dict, crossings, touches, spans) -> None:
    print(f"{Path(cache['dir']).name}: {len(cache['frames'])} frames, "
          f"{len(cache['episodes'])} coverage episodes")
    print()
    print("rule       passages    in   out")
    for rule in RULES:
        found = crossings[rule]
        ins = sum(1 for c in found if c.direction.value == "in")
        print(f"  {rule:<9} {len(found):>8} {ins:>5} {len(found) - ins:>5}")

    print()
    print("coverage episodes (frame ranges):")
    for episode in cache["episodes"]:
        print(
            f"  frames {episode['first']}-{episode['last']} "
            f"({episode['frames']} covered, {episode['slices']} slices, lag {episode['lag']:+.2f})"
        )

    print()
    print("tracks, with the evidence available to them:")
    header = "  track  frames  span            episodes overlapping  outlived?  "
    print(header + "  ".join(f"{rule:<9}" for rule in RULES))
    verdicts = {rule: {} for rule in RULES}
    for rule in RULES:
        for touch in touches[rule]:
            verdicts[rule][touch.frames] = touch.direction

    def _said(rule: str, span: dict) -> str:
        decided = verdicts[rule].get(span["seen"])
        return f"{(decided.value if decided else '-'):<9}"

    for track_id, span in sorted(spans.items()):
        overlapping = len(span["episodes"])
        outlived = "YES" if span["episode_outlived_track"] else "no"
        row = "  ".join(_said(rule, span) for rule in RULES)
        print(
            f"  {track_id:>5}  {span['seen']:>6}  "
            f"{span['first']:>5}-{span['last']:<8}  {overlapping:>19}  {outlived:>9}  {row}"
        )

    print()
    missed = [t for t, s in spans.items() if s["episode_outlived_track"]]
    print(
        "tracks whose coverage episode was still running after the detector lost them: "
        f"{len(missed)} of {len(spans)}"
    )
    if missed:
        print("  -> these are the ones an endpoint reading cannot see:", missed)


def main() -> None:
    args = _arguments()
    cache = json.loads(Path(args.cache).expanduser().read_text())
    spans = _extent(cache)
    crossings, touches = _replay(cache, args)
    _report(cache, crossings, touches, spans)


if __name__ == "__main__":
    main()
