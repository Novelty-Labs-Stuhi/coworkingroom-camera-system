"""Close visits that were never closed, so the room does not hold people from days ago.

Every entry the system could not pair with an exit left somebody inside for ever. On this
deployment that was dozens of guests, each credited with every hour of every day since -- the
figures were not slightly wrong, they were obviously wrong, which is how it was noticed.

The live faults are fixed (an unrecognised arrival now gets an enrolled identity, and an exit
always closes a visit), but nothing repairs what is already recorded. This does: any visit
still open after a cut-off gets an exit written at the end of the day it began, because nobody
sleeps in the office and midnight is a defensible guess where "still here" is not.

The exits it writes are marked ``named_by = "closed"``, so a repaired visit can always be told
from one the doorway actually saw end.

    python tools/close_stale_visits.py --database data/occupancy.db --stale-hours 6
"""

from __future__ import annotations

import argparse
import datetime
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_INSERT = """
INSERT INTO events (timestamp, name, direction, camera, named_by, natural)
VALUES (?, ?, 'out', '', 'closed', '')
"""


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="data/occupancy.db")
    parser.add_argument(
        "--stale-hours",
        type=float,
        default=6.0,
        help="leave visits younger than this alone; somebody may really still be here",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _open_visits(rows) -> dict[str, float]:
    """Who is still inside, and since when, replaying every crossing in order."""
    inside: dict[str, float] = {}
    for timestamp, name, direction in rows:
        if direction == "in":
            inside.setdefault(name, timestamp)
        else:
            inside.pop(name, None)
    return inside


def _end_of_day(timestamp: float) -> float:
    """A minute before midnight of the day the visit began."""
    started = datetime.datetime.fromtimestamp(timestamp)
    return started.replace(hour=23, minute=59, second=0, microsecond=0).timestamp()


def main() -> None:
    args = _arguments()
    connection = sqlite3.connect(args.database)
    rows = connection.execute(
        "SELECT timestamp, name, direction FROM events ORDER BY timestamp"
    ).fetchall()

    now = datetime.datetime.now().timestamp()
    cutoff = now - args.stale_hours * 3600
    open_visits = _open_visits(rows)
    stale = {name: since for name, since in open_visits.items() if since < cutoff}

    print(
        f"open visits: {len(open_visits)}   "
        f"stale (older than {args.stale_hours} h): {len(stale)}"
    )
    for name, since in sorted(stale.items(), key=lambda pair: pair[1])[:8]:
        began = datetime.datetime.fromtimestamp(since).strftime("%m-%d %H:%M")
        print(f"  {name:<24} in since {began}")
    if args.dry_run:
        print("(dry run: nothing written)")
        return

    # Repeated until it converges. Occupancy is keyed by name, so one person cannot be inside
    # twice -- but the recorded crossings contain entries that were never closed *and* later
    # entries for the same person, and each pass can only close the earliest of them. One pass
    # reported 33 closed and 34 still open, which reads like a failure and is really the next
    # layer surfacing.
    closed = 0
    for _ in range(20):
        for name, since in stale.items():
            # Never before the entry itself, and never in the future: a visit beginning late at
            # night closes a minute later rather than a minute earlier.
            connection.execute(_INSERT, (max(min(_end_of_day(since), now), since + 60), name))
            closed += 1
        connection.commit()
        rows = connection.execute(
            "SELECT timestamp, name, direction FROM events ORDER BY timestamp"
        ).fetchall()
        open_visits = _open_visits(rows)
        stale = {name: since for name, since in open_visits.items() if since < cutoff}
        if not stale:
            break

    print(f"closed {closed} visit(s)")
    print(f"still open: {len(open_visits)}   still stale: {len(stale)}")


if __name__ == "__main__":
    main()
