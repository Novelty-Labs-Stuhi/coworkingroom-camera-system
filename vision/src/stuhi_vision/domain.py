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


class Outcome(Enum):
    """How confidently a crossing was attributed to a person."""

    NAMED = "named"  # matched the gallery above threshold, with a margin
    UNKNOWN = "unknown"  # a face was seen but matched nobody -- enrollable
    UNIDENTIFIED = "unidentified"  # never got a usable face; no claim made


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
class Sighting:
    """A committed crossing, with the evidence behind its identity.

    Carries the face embedding and crop of the frame that produced the best match, so an
    ``UNKNOWN`` sighting can later be labelled and enrolled into the gallery.
    """

    timestamp: float
    direction: Direction
    name: str | None
    score: float
    outcome: Outcome
    face_embedding: np.ndarray | None = None
    face_crop: np.ndarray | None = None
    # True when this crossing is the one that *invented* its identity -- somebody the
    # recogniser had never seen, now enrolled under a name the system made up. It is the
    # only moment an identity is new, which makes it the natural trigger for asking a human
    # to name them: exactly once per person, with no record of what has already been asked
    # to keep in step, and unaffected by a restart.
    introduced: bool = False


@dataclass(frozen=True, slots=True)
class Event:
    """An occupancy change to be recorded: someone entered or left."""

    timestamp: float
    name: str
    direction: Direction
    # Which camera saw it. With one camera per direction, this is how a drifting count can
    # be traced to the camera responsible rather than guessed at.
    camera: str = ""
    # How an exit got its name: "face" from what the camera itself recognised, "pool" from
    # matching against the people known to be in the room, "body" from the body embedding,
    # "nobody" when it could not be told. Entries are always "face".
    named_by: str = ""
    # What the exit's *own* evidence said, before the room was consulted. When this differs
    # from the name recorded, the pool overruled the face -- which is the pair worth checking,
    # because one of the two is wrong.
    natural: str = ""


class EventSink(Protocol):
    """Anything that can durably record occupancy events (see store.EventStore)."""

    def record(self, event: Event) -> None: ...
