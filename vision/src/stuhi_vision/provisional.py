"""Which identities the system invented for itself, and so still need a person's name.

An unrecognised arrival is enrolled under a name the system makes up, so that the same
person coming back is matched to it rather than becoming somebody else again. That works,
but it leaves a gallery in which "arsenii" and "guest-0802-161045" sit side by side with
nothing to say that one was chosen by a human and the other was not.

The labelling page needs that distinction: an identity nobody has named is the one piece of
work worth interrupting somebody for, and everything else can wait until they look.

**Recorded rather than inferred.** The obvious shortcut is to match the name against
``guest-*``, and it is wrong in both directions: a person renamed to something beginning with
"guest" would be pestered for ever, and a name a human typed is indistinguishable from one
the system generated if the generator's format ever changes. A name is provisional because of
*how it came about*, which is knowable only at the moment it is created -- so that is when it
is written down.

The file is advisory: names it holds that are no longer in the gallery are dropped on load,
so hand-editing the gallery cannot leave this pointing at people who do not exist.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from pathlib import Path

_log = logging.getLogger(__name__)


class Provisional:
    """The set of identities the system named itself, persisted one name per line."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._names: set[str] = self._read()

    def _read(self) -> set[str]:
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return set()
        return {line.strip() for line in lines if line.strip()}

    def _write(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text("\n".join(sorted(self._names)) + "\n", encoding="utf-8")
        except OSError as error:
            # Losing this degrades the labelling page to "everyone looks named"; it must not
            # be able to stop a crossing being recorded.
            _log.warning("could not write %s: %s", self._path, error)

    def add(self, name: str) -> None:
        """Note that the system invented this name."""
        with self._lock:
            if name in self._names:
                return
            self._names.add(name)
            self._write()

    def claimed(self, name: str) -> None:
        """A human has given this identity a name, so it is no longer provisional."""
        with self._lock:
            if name not in self._names:
                return
            self._names.discard(name)
            self._write()

    def holds(self, name: str) -> bool:
        with self._lock:
            return name in self._names

    def names(self, known: Iterable[str] | None = None) -> set[str]:
        """The provisional identities, optionally narrowed to those still in the gallery.

        Passing the gallery's names is what keeps this self-healing: an identity removed or
        merged elsewhere disappears from here too, rather than lingering as work that can
        never be completed.
        """
        with self._lock:
            if known is None:
                return set(self._names)
            live = self._names & set(known)
            if live != self._names:
                self._names = live
                self._write()
            return set(live)
