"""The part of a profile a person writes themselves.

Everything else a profile shows is derived from what the cameras recorded -- hours, streaks,
which faces recognition uses. A bio is not, and could never be: no doorway will ever work out
that somebody sits by the window and is learning Finnish. So it is the one thing here that is
*stored*, and it is stored beside the review data because it belongs to this deployment rather
than to the code.

It follows a rename, which is the whole reason this is a store rather than a field. Correcting
"ilari" to "Ilari" is one person having one name spelled properly; if their own words about
themselves stayed behind under the old spelling, the correction would quietly cost them their
profile.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

# Long enough for the few lines somebody actually wants to say, short enough that the file
# stays something a person can read. Refused rather than truncated: silently cutting off what
# somebody wrote is worse than telling them it did not fit.
LONGEST = 500


class ProfileStore:
    """Each person's bio, kept in one small JSON file.

    The path is injected rather than found: the tests use a temporary one, and a module that
    decides for itself where to write cannot be tested without touching the real thing.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._bios: dict[str, str] = {}
        self._load()

    def bio(self, name: str) -> str:
        """What this person says about themselves, or "" -- which is where everybody starts."""
        with self._lock:
            return self._bios.get(name.strip(), "")

    def write(self, name: str, bio: str) -> str:
        """Save a bio, or clear it by saving nothing. Returns what was stored.

        Raises :class:`ValueError` if it is too long, so the caller can say so rather than
        store a truncated version of somebody's own words.
        """
        name = name.strip()
        if not name:
            raise ValueError("a bio belongs to somebody, so a name is required")
        text = bio.strip()
        if len(text) > LONGEST:
            raise ValueError(f"a bio is at most {LONGEST} characters; this one is {len(text)}")
        with self._lock:
            if text:
                self._bios[name] = text
            else:
                # Clearing is a real thing to want, and an empty string kept in the file would
                # read as "written, and empty" rather than "never written".
                self._bios.pop(name, None)
            self._flush()
        return text

    def rename(self, old: str, new: str) -> bool:
        """Carry a bio to a corrected name. Returns whether anything moved.

        When both names have one -- which is what a *merge* of two real people's entries looks
        like -- the two are joined rather than one thrown away. Nothing a person wrote about
        themselves is discarded on the strength of a click somewhere else; whoever reads the
        joined result can delete the half that is wrong.
        """
        old, new = old.strip(), new.strip()
        if not old or not new or old == new:
            return False
        with self._lock:
            moving = self._bios.pop(old, "")
            if not moving:
                return False
            standing = self._bios.get(new, "")
            if standing and moving not in standing:
                self._bios[new] = f"{standing}\n{moving}"[:LONGEST]
            elif not standing:
                self._bios[new] = moving
            self._flush()
            return True

    def _load(self) -> None:
        if not self._path.exists():
            return
        stored = json.loads(self._path.read_text(encoding="utf-8"))
        self._bios = {str(name): str(bio) for name, bio in stored.get("bios", {}).items()}

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps({"bios": self._bios}, indent=2), encoding="utf-8")
