"""Backfill the re-ID store from clips already recorded before the server embedded them.

Embeds every ``*.mp4`` in the clips directory (video id = filename stem, e.g.
``2026-07-24_20-59-21``) and adds it to the store. By default everything lands under
``unknown``; pass ``--label NAME`` when you already know a whole batch is one person.

    python -m stuhi_vision.reid_backfill --clips clips/ --store data/reid_store.json
    python -m stuhi_vision.reid_backfill --clips faces/arsenii/ --label Arsenii

Already-present video ids are skipped, so it is safe to re-run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .recognition.reid import ClipEmbedder
from .reid_store import UNKNOWN, ReidStore


def _existing_ids(store: ReidStore) -> set[str]:
    return {e["video_id"] for entries in store.people.values() for e in entries}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=Path("clips"), help="clips dir")
    parser.add_argument("--store", type=Path, default=Path("data/reid_store.json"))
    parser.add_argument("--label", default=UNKNOWN, help="name to file clips under")
    parser.add_argument("--model", default="osnet_x1_0", help="torchreid model name")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    videos = sorted(args.clips.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No .mp4 clips found in {args.clips}")

    store = ReidStore(args.store)
    embedder = ClipEmbedder(model_name=args.model, device=args.device)
    seen = _existing_ids(store)

    added = skipped = empty = 0
    for video in videos:
        video_id = video.stem
        if video_id in seen:
            print(f"  {video.name}: already in store -- skipped")
            skipped += 1
            continue
        emb = embedder.embed_video(video)
        if emb is None:
            print(f"  {video.name}: no person detected -- skipped")
            empty += 1
            continue
        store.add(video_id, emb, args.label)
        print(f"  {video.name}: added under '{args.label}'")
        added += 1

    print(f"\nDone: {added} added, {skipped} already present, {empty} without a person.")
    print(f"Store now holds: {store.summary()}")


if __name__ == "__main__":
    main()
