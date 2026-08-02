"""Proof that a camera is still doing work, not merely still running.

The supervisor asked ``pgrep`` whether the process existed, which is a different question
from whether it is watching anything. A pipeline blocked on a camera socket that stopped
sending never exits, never logs, and answers that question with "yes" for as long as you
like -- there is a 39-hour hole in the record where exactly that happened.

So each camera stamps a file with the time it last handled a frame, and the supervisor reads
the stamp instead of the process table. A stamp that stops advancing is a wedged pipeline
however healthy it looks from outside.

**One file per camera, not one for the pipeline.** With two cameras a single shared stamp
keeps advancing while one of them is dead -- and the dead one may be the counting camera, in
which case nothing is counted at all and the stamp says everything is fine.

Writing is throttled and never raises. A heartbeat that threw would take down the pipeline it
exists to protect, and a full disk is a reason to keep counting, not to stop.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

_log = logging.getLogger(__name__)

# Frames arrive at 5-15 a second; stamping every one of them is thousands of pointless writes
# a minute. Once a second is far finer than any sane staleness threshold.
_INTERVAL = 1.0


class Heartbeat:
    """A file holding the epoch second at which this camera last handled a frame."""

    def __init__(
        self,
        path: Path,
        interval: float = _INTERVAL,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._path = Path(path)
        self._interval = interval
        self._now = now
        self._last = 0.0
        self._complained = False

    def beat(self) -> None:
        """Record that a frame was just handled, at most once per interval."""
        moment = self._now()
        if moment - self._last < self._interval:
            return
        self._last = moment
        self._write(moment)

    def _write(self, moment: float) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Seconds as plain text rather than relying on the file's mtime: a shell can read
            # it with `cat` and subtract, which needs no stat parsing and no assumptions
            # about how the filesystem keeps timestamps.
            self._path.write_text(f"{int(moment)}\n")
            self._complained = False
        except OSError as error:
            # Once, not once a second: a failing disk should leave one line in the log, not
            # drown the thing that would tell you what else went wrong.
            if not self._complained:
                _log.warning("could not write heartbeat %s: %s", self._path, error)
                self._complained = True


def stale(path: Path, older_than: float, now: float) -> bool:
    """Whether a stamp is missing or older than ``older_than`` seconds.

    A missing stamp counts as stale: a camera that has never written one has never handled a
    frame, which is the condition being watched for rather than an exemption from it.
    """
    try:
        return now - float(Path(path).read_text().strip()) > older_than
    except (OSError, ValueError):
        return True
