"""Zones drawn by hand, stored beside the config rather than inside it.

A zone is drawn by looking at a picture, so it belongs to whoever is looking -- not in a file
they have to edit over SSH. It is kept as JSON next to the config and layered over it at
startup, which keeps the config file as the thing a person writes and this as the thing the
UI writes. Nothing here overwrites a config: an absent entry simply means "use what the
config says".

Saving a zone also saves the frame it was drawn on, which is what
:class:`~.alignment.DriftWatch` later compares against to notice the camera has moved. The two
belong together: a zone is only meaningful for as long as the view it was drawn on holds.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_FILE = "zones.json"

# Told (camera, zone) when a zone is drawn, or (camera, None) when one is removed.
Listener = Callable[[str, "DrawnZone | None"], None]


@dataclass(frozen=True, slots=True)
class DrawnZone:
    """One camera's hand-drawn zone, in fractions of its frame."""

    x1: float
    y1: float
    x2: float
    y2: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @classmethod
    def from_corners(cls, x1: float, y1: float, x2: float, y2: float) -> DrawnZone:
        """Build from two corners in any order, clamped to the frame.

        A rectangle dragged upwards or leftwards arrives with its corners reversed; storing
        that unnormalised would make every later overlap test quietly false.
        """
        left, right = sorted((_clamp(x1), _clamp(x2)))
        top, bottom = sorted((_clamp(y1), _clamp(y2)))
        if right - left < 0.01 or bottom - top < 0.01:
            raise ValueError("that zone is too small to mean anything")
        return cls(x1=left, y1=top, x2=right, y2=bottom)


class ZoneStore:
    """Reads and writes the hand-drawn zones, and the frames they were drawn on."""

    def __init__(self, directory: Path) -> None:
        self._dir = directory
        self._path = directory / _FILE
        self._lock = threading.RLock()
        self._zones: dict[str, DrawnZone] = {}
        # Told when a zone changes, so a running pipeline can pick it up. Without this a
        # redrawn zone waited for a restart, which is the same as not being able to redraw it:
        # the reason to redraw is usually that the camera has moved and the count is wrong now.
        self._listeners: list[Listener] = []
        self._load()

    def watch(self, listener: Listener) -> None:
        with self._lock:
            self._listeners.append(listener)

    def reference_path(self, camera: str) -> Path:
        """Where the frame this camera's zone was drawn on lives."""
        return self._dir / f"reference-{_safe(camera)}.jpg"

    def get(self, camera: str) -> DrawnZone | None:
        with self._lock:
            return self._zones.get(camera)

    def all(self) -> dict[str, DrawnZone]:
        with self._lock:
            return dict(self._zones)

    def save(self, camera: str, zone: DrawnZone) -> None:
        with self._lock:
            self._zones[camera] = zone
            self._write()
            listeners = list(self._listeners)
        # Outside the lock: a listener reaches into a running pipeline, and holding this lock
        # while it does would tie the UI's thread to the camera's.
        for listener in listeners:
            listener(camera, zone)

    def remove(self, camera: str) -> bool:
        """Forget this camera's drawn zone, falling back to the config. False if it had none."""
        with self._lock:
            if self._zones.pop(camera, None) is None:
                return False
            self._write()
            listeners = list(self._listeners)
        for listener in listeners:
            listener(camera, None)
        return True

    def _write(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        payload = {
            name: {"x1": z.x1, "y1": z.y1, "x2": z.x2, "y2": z.y2}
            for name, z in self._zones.items()
        }
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not self._path.exists():
            return
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        for name, entry in raw.items():
            self._zones[name] = DrawnZone(
                x1=float(entry["x1"]),
                y1=float(entry["y1"]),
                x2=float(entry["x2"]),
                y2=float(entry["y2"]),
            )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe(name: str) -> str:
    """A camera name as a filename fragment, since names come from the config."""
    return "".join(character if character.isalnum() else "-" for character in name)
