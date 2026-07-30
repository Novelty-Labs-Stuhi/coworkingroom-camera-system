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

from .domain import Direction
from .threshold import ThresholdConfig

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


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """One camera: its frames, how it recognises a passage, and what it reports.

    Each camera at a doorway sees faces in one direction only -- in the other it films the
    back of someone's head. With one camera per direction, both cameras see *every*
    passage, so each is given the direction it can actually judge (``announce``) and ignores
    the other. That is what stops one person being counted twice.

    How a passage is recognised belongs to the camera too, because each has its own view:
    its own pixel coordinates, its own idea of which side is inside, its own doorframe.
    ``detector`` is either a :class:`DoorwayConfig` (a line the foot point crosses) or a
    :class:`ThresholdConfig` (occluding the doorframe and leaving at an edge).
    """

    name: str
    source: SourceConfig
    detector: DoorwayConfig | ThresholdConfig
    announce: Reported = "both"

    @property
    def doorway(self) -> DoorwayConfig | None:
        """The line configuration, when this camera uses one -- for drawing overlays."""
        return self.detector if isinstance(self.detector, DoorwayConfig) else None


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
    cameras: list[CameraConfig]
    thresholds: Thresholds = field(default_factory=Thresholds)
    performance: Performance = field(default_factory=Performance)
    paths: Paths = field(default_factory=Paths)
    web: Web = field(default_factory=Web)
    telegram: Telegram = field(default_factory=Telegram.from_env)

    @property
    def camera(self) -> CameraConfig:
        """The first camera, for tools that inherently work on one view at a time."""
        return self.cameras[0]


def _point(raw: list[float]) -> Point:
    x, y = raw
    return float(x), float(y)


def _detector(raw: dict) -> DoorwayConfig | ThresholdConfig:
    """Build whichever passage detector the camera describes.

    A ``zone`` means the doorframe rule: overlap the frame and leave at an edge. Otherwise
    it is the older line-crossing form, kept because a camera that genuinely sees a
    threshold from the side is still well served by it.
    """
    if "zone" in raw or raw.get("detector") == "threshold":
        zone = raw.get("zone", [0.0, 0.0, 0.25, 1.0])
        return ThresholdConfig(
            zone=(float(zone[0]), float(zone[1]), float(zone[2]), float(zone[3])),
            edge=raw.get("edge", "left"),
            margin=float(raw.get("margin", 0.12)),
            passing_means=Direction(raw.get("passing_means", "in")),
            discriminator=raw.get("discriminator", "edge"),
            growth_margin=float(raw.get("growth_margin", 0.12)),
            min_height=float(raw.get("min_height", 0.35)),
            lost_after=int(raw.get("lost_after", 6)),
        )
    return DoorwayConfig(
        line_a=_point(raw["line_a"]),
        line_b=_point(raw["line_b"]),
        inside_side=raw["inside_side"],
    )


def _cameras(data: dict) -> list[CameraConfig]:
    """Read either a list of ``[[camera]]`` tables or the single-camera form.

    The single-camera form -- a ``[source]`` and a ``[doorway]`` table -- is still accepted
    so an existing deployment keeps working unchanged. It is exactly one camera called
    "camera", which is what it always was.
    """
    entries = data.get("camera")
    if not entries:
        if "source" not in data:
            raise ValueError("config needs either [[camera]] entries or a [source] table")
        return [
            CameraConfig(
                name=data["source"].get("name", "camera"),
                source=SourceConfig(**{k: v for k, v in data["source"].items() if k != "name"}),
                detector=_detector(data["doorway"]),
                announce=data["doorway"].get("announce", "both"),
            )
        ]

    cameras = [
        CameraConfig(
            name=entry.get("name", f"camera-{index + 1}"),
            source=SourceConfig(
                kind=entry.get("kind", "stream"),
                target=entry["target"],
                rotate=entry.get("rotate", 0),
            ),
            detector=_detector(entry),
            announce=entry.get("announce", "both"),
        )
        for index, entry in enumerate(entries)
    ]
    names = [camera.name for camera in cameras]
    if len(set(names)) != len(names):
        # Names identify a camera in events and in the UI, so duplicates would make the
        # record of which camera saw what meaningless.
        raise ValueError(f"camera names must be unique, got {names}")
    return cameras


def load(path: str | Path) -> Config:
    """Parse a TOML config file into a typed :class:`Config`."""
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))

    cameras = _cameras(data)
    thresholds = Thresholds(**data.get("thresholds", {}))
    performance = Performance(**data.get("performance", {}))
    paths_raw = data.get("paths", {})
    paths = Paths(
        gallery_dir=Path(paths_raw.get("gallery_dir", "gallery")),
        database=Path(paths_raw.get("database", "data/occupancy.db")),
        review_dir=Path(paths_raw.get("review_dir", "data/review")),
    )
    return Config(
        cameras=cameras,
        thresholds=thresholds,
        performance=performance,
        paths=paths,
        web=Web(**data.get("web", {})),
    )
