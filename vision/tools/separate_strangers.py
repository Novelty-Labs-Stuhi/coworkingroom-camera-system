"""Move the faces of unnamed identities out of the named gallery, where they do not belong.

Until now an unrecognised arrival's face was enrolled straight into the gallery alongside the
people who have names. That was meant to make the same person matchable next time, and it did
not work -- the matching set was rebuilt only from human-labelled faces, so those entries were
never compared against anything and a returning stranger became a new person every visit.

Fixing the matching set made them matchable, which exposed the real problem: an unverified face
competing against everybody named. More unnamed identities means closer runner-up scores, means
the margin refuses more often, means more unnamed identities. So they now live in their own
store, and this moves the ones already written.

    python tools/separate_strangers.py --gallery gallery --strangers data/strangers \
        --provisional data/provisional --dry-run

Only names listed as provisional are moved: that file records which names the *system* invented,
so somebody genuinely called something unusual is never swept up by a pattern match on the name.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery", default="gallery")
    parser.add_argument("--strangers", default="data/strangers")
    parser.add_argument("--provisional", default="data/provisional")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _provisional_names(path: Path) -> set[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return {line.strip() for line in lines if line.strip()}
    except OSError:
        return set()


def main() -> None:
    args = _arguments()
    gallery = Path(args.gallery).expanduser()
    strangers = Path(args.strangers).expanduser()
    listed = _provisional_names(Path(args.provisional).expanduser())

    if not listed:
        print(f"nothing listed in {args.provisional}: no identity is known to be provisional")
        return

    moving = [path for path in sorted(gallery.glob("*.npy")) if path.stem in listed]
    staying = sorted(p.stem for p in gallery.glob("*.npy") if p.stem not in listed)

    print(f"named people staying in {gallery}: {len(staying)}")
    print(f"provisional identities to move: {len(moving)}")
    for path in moving:
        print(f"  {path.stem}")
    if not moving:
        return

    if args.dry_run:
        print("\n--dry-run: nothing moved")
        return

    strangers.mkdir(parents=True, exist_ok=True)
    for path in moving:
        # Copied then removed rather than renamed: a rename across filesystems fails partway
        # and would leave the face in neither store.
        shutil.copy2(path, strangers / path.name)
        path.unlink()
    print(f"\nmoved {len(moving)} identity file(s) into {strangers}")


if __name__ == "__main__":
    main()
