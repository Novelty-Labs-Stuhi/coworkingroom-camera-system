"""A record of every identity folded into another, so that folding can be undone.

Naming a provisional identity is a statement about *who that identity is*, and it is true of
every crossing it ever produced -- so the correction is applied to all of them at once. That
is what makes one label worth hundreds of unreviewed clips.

It is also destructive. Once ``guest-0803-0900`` has become ``Ilari`` throughout the record,
nothing in the events table remembers that those crossings were ever anything else, and a
mistaken merge could not be picked apart from the crossings Ilari really made. Two people's
hours would be fused permanently, on the strength of one click.

So the affected rows are written down *before* they are changed, by id. That makes a merge
exactly reversible, and it fixes the order of operations: the log is written first, so the
worst outcome of a crash is a merge that was recorded and not applied -- visible, and safe to
apply again -- rather than one applied and not recorded, which is silent and permanent.

The file is JSON lines, appended to and never rewritten. There are a handful of merges a week
at most; the simplicity of a file that can only grow is worth more here than query speed.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

_log = logging.getLogger(__name__)

# Why a merge happened. Recorded because the two carry very different confidence, and a wrong
# one should be traceable to the decision that made it rather than guessed at.
LABELLED = "labelled"      # a human named the identity
SUGGESTED = "suggested"    # a human accepted the system's proposal


@dataclass(frozen=True, slots=True)
class Merge:
    """One identity folded into another, with the rows it changed."""

    at: float
    absorbed: str          # the provisional identity that no longer exists
    into: str              # who they turned out to be
    because: str           # LABELLED or SUGGESTED
    events: list[int] = field(default_factory=list)   # the event rows rewritten, by id

    @property
    def readable(self) -> str:
        return (
            f"{self.absorbed} -> {self.into} ({self.because}, "
            f"{len(self.events)} crossing(s))"
        )


class MergeLog:
    """Append-only history of identity merges."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    def record(self, merge: Merge) -> None:
        """Write a merge down. Call *before* applying it, never after."""
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(merge)) + "\n")

    def all(self) -> list[Merge]:
        """Every merge, oldest first. Unreadable lines are skipped, not fatal."""
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        merges = []
        for line in lines:
            if not line.strip():
                continue
            try:
                merges.append(Merge(**json.loads(line)))
            except (ValueError, TypeError) as error:
                # A truncated final line after a crash must not make the whole log
                # unreadable -- losing the ability to undo every earlier merge would be a
                # far worse outcome than losing the last one.
                _log.warning("skipping unreadable merge log line: %s", error)
        return merges

    def absorbed(self) -> dict[str, str]:
        """Identity -> who it was folded into, latest wins."""
        return {merge.absorbed: merge.into for merge in self.all()}

    def latest_for(self, absorbed: str) -> Merge | None:
        """The most recent merge that folded this identity away, if any."""
        found = [merge for merge in self.all() if merge.absorbed == absorbed]
        return found[-1] if found else None


def merge_identity(store, log: MergeLog, absorbed: str, into: str, because: str, now: float):
    """Fold one identity into another across the whole record. Returns the merge, or None.

    The order is the safety property. The rows are read, then written down, then changed --
    so a crash between the log and the update leaves a merge that is recorded and not applied,
    which is visible in the log and safe to apply again. The other order leaves one applied
    and not recorded: silent, and impossible to undo.

    Merging a name into itself is refused rather than logged. It changes nothing, and a log
    full of no-ops makes the entries that matter harder to find.
    """
    if not absorbed or not into or absorbed == into:
        return None
    ids = store.events_named(absorbed)
    merge = Merge(at=now, absorbed=absorbed, into=into, because=because, events=ids)
    log.record(merge)
    store.rename_events(ids, into)
    _log.info("merged %s", merge.readable)
    return merge


def undo(store, merge: Merge) -> int:
    """Put an absorbed identity back on exactly the rows it held. Returns rows restored.

    Precise because the merge recorded ids: renaming back by name would also drag along every
    crossing the target made on their own, fusing the mistake rather than undoing it.
    """
    return store.rename_events(merge.events, merge.absorbed)
