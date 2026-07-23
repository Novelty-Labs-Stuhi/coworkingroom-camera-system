"""Typed records that cross module boundaries.

Everything here is plain data -- no OpenCV, no torch, no I/O -- so the pipeline's
pure logic (doorway geometry, matching, the ledger) can be built and tested without
any heavy model dependency installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np


class Direction(Enum):
    """Which way a person crossed the doorway threshold."""

    IN = "in"
    OUT = "out"


@dataclass(frozen=True, slots=True)
class Box:
    """An axis-aligned bounding box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def centroid(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def foot(self) -> tuple[float, float]:
        """Bottom-centre point -- where the person meets the floor."""
        return (self.x1 + self.x2) / 2, self.y2

    def crop(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        x1 = max(0, int(self.x1))
        y1 = max(0, int(self.y1))
        x2 = min(w, int(self.x2))
        y2 = min(h, int(self.y2))
        return image[y1:y2, x1:x2]


@dataclass(slots=True)
class Frame:
    """One captured image plus the wall-clock time it was captured."""

    timestamp: float
    image: np.ndarray


@dataclass(slots=True)
class TrackedPerson:
    """A person detection with an identity that is stable frame-to-frame."""

    track_id: int
    box: Box


@dataclass(frozen=True, slots=True)
class Crossing:
    """A person crossing the doorway threshold. The identity data lives in the track's
    session (accumulated over the approach), so a crossing only needs to say who
    crossed, which way, and when."""

    track_id: int
    direction: Direction
    timestamp: float


@dataclass(frozen=True, slots=True)
class Event:
    """An occupancy change to be recorded: someone entered or left."""

    timestamp: float
    name: str
    direction: Direction


class EventSink(Protocol):
    """Anything that can durably record occupancy events (see store.EventStore)."""

    def record(self, event: Event) -> None: ...
