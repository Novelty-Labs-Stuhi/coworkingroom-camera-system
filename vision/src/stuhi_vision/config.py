"""Typed configuration, loaded once from a TOML file.

Keeping every tunable in one typed place (rather than scattered ``os.environ`` reads
or magic numbers in the code) is what lets the rest of the modules stay pure and
dependency-injected.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Point = tuple[float, float]
Side = Literal["left", "right"]
Reported = Literal["in", "out", "both"]


@dataclass(frozen=True, slots=True)
class SourceConfig:
    kind: Literal["file", "webcam", "stream"]
    target: str  # file path, webcam index, or stream URL
    # Degrees to rotate every frame so it is upright: 0, 90, 180 or 270. Detection and
    # face recognition both fail outright on an inverted image, so a camera mounted the
    # wrong way up must be corrected before anything looks at it.
    rotate: int = 0


@dataclass(frozen=True, slots=True)
class DoorwayConfig:
    line_a: Point  # one end of the threshold line, in pixels
    line_b: Point  # the other end
    inside_side: Side  # which side of the line is "inside the office"
    # Which crossings this camera films and announces. A camera only sees faces in one
    # direction; in the other it films the back of someone's head, which cannot be judged
    # or labelled -- and with a camera on each side of the door, every passage is seen
    # twice, so announcing both directions from both cameras means two messages per person.
    # Crossings in the unreported direction are still counted and written to the ledger;
    # only the clip and the chat message are suppressed.
    announce: Reported = "both"


@dataclass(frozen=True, slots=True)
class Thresholds:
    detection_conf: float = 0.4  # min YOLO confidence for a person
    face_match: float = 0.35  # min cosine to accept a face as a known person
    face_margin: float = 0.05  # best name must beat the runner-up by this, else unknown
    exit_similarity: float = 0.6  # min cosine to link an exit to an occupant
    exit_margin: float = 0.05  # best occupant must beat the runner-up by this, else ambiguous
    # Frames a track must exist before its crossing counts (anti-flicker). This is a
    # count, so it is really a *duration* divided by the frame rate: at ~2 fps a person
    # crossing in a second is seen once or twice, and a gate of 3 would reject everyone.
    # Keep it just high enough to reject single-frame noise, and raise it if the rate
    # improves.
    min_track_age: int = 2


@dataclass(frozen=True, slots=True)
class Performance:
    """Inference cost knobs. Measured, not guessed -- see docs/design.md."""

    # YOLO weights. An ONNX export avoids the slow torch fallback on CPUs without AVX2
    # and measured 160 ms/frame against 208 ms for the .pt -- see tools/export_onnx.py.
    detect_model: str = "yolov8n.pt"
    # YOLO inference resolution: 640 costs ~440 ms/frame on a CPU without AVX2, 320
    # costs ~200 ms. A doorway sees people close up, so 320 is the sensible default.
    # An ONNX export fixes this shape at export time -- keep the two in step.
    detect_imgsz: int = 320
    # Frames the reader thread may bank while the models work. This is what stops a slow
    # machine from *missing* a crossing; it trades latency for completeness.
    buffer_capacity: int = 64
    # Threads embedding faces concurrently. Per-person work is independent, unlike
    # tracking, which must stay sequential.
    face_workers: int = 4
    # Fraction of the frame that must change for a frame to reach the models at all.
    motion_min_fraction: float = 0.004
    # Fraction of the frame padded around the doorway line to form the detection crop.
    # 0 disables cropping and runs detection on the whole frame.
    crop_padding: float = 0.35
    # Frames of history kept as the clip's pre-roll -- the approach to the crossing. Held
    # as JPEG (~14 KB each), so a few seconds is cheap.
    clip_frames: int = 48
    # Consecutive people-free frames before a clip is considered finished. This is what
    # makes the video end on an empty doorway instead of stopping mid-stride.
    clip_clear_frames: int = 10
    # Hard cap on clip length, so someone loitering cannot hold a clip open forever.
    clip_max_frames: int = 200


@dataclass(frozen=True, slots=True)
class Paths:
    gallery_dir: Path = Path("gallery")
    database: Path = Path("data/occupancy.db")
    review_dir: Path = Path("data/review")  # sighting crops/embeddings awaiting a label


@dataclass(frozen=True, slots=True)
class Web:
    """The labelling UI. Runs in the pipeline process so it shares the live gallery."""

    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8800


@dataclass(frozen=True, slots=True)
class Telegram:
    """Bot credentials, read from the environment so they are never committed."""

    bot_token: str | None = None
    chat_id: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    @classmethod
    def from_env(cls) -> Telegram:
        return cls(
            bot_token=os.environ.get("TELEGRAM_BOT_TOKEN"),
            chat_id=os.environ.get("TELEGRAM_CHAT_ID"),
        )


@dataclass(frozen=True, slots=True)
class Config:
    source: SourceConfig
    doorway: DoorwayConfig
    thresholds: Thresholds = field(default_factory=Thresholds)
    performance: Performance = field(default_factory=Performance)
    paths: Paths = field(default_factory=Paths)
    web: Web = field(default_factory=Web)
    telegram: Telegram = field(default_factory=Telegram.from_env)


def _point(raw: list[float]) -> Point:
    x, y = raw
    return float(x), float(y)


def load(path: str | Path) -> Config:
    """Parse a TOML config file into a typed :class:`Config`."""
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))

    source = SourceConfig(**data["source"])
    door = data["doorway"]
    doorway = DoorwayConfig(
        line_a=_point(door["line_a"]),
        line_b=_point(door["line_b"]),
        inside_side=door["inside_side"],
        announce=door.get("announce", "both"),
    )
    thresholds = Thresholds(**data.get("thresholds", {}))
    performance = Performance(**data.get("performance", {}))
    paths_raw = data.get("paths", {})
    paths = Paths(
        gallery_dir=Path(paths_raw.get("gallery_dir", "gallery")),
        database=Path(paths_raw.get("database", "data/occupancy.db")),
        review_dir=Path(paths_raw.get("review_dir", "data/review")),
    )
    return Config(
        source=source,
        doorway=doorway,
        thresholds=thresholds,
        performance=performance,
        paths=paths,
        web=Web(**data.get("web", {})),
    )
