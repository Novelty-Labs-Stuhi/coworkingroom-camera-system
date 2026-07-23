"""Typed configuration, loaded once from a TOML file.

Keeping every tunable in one typed place (rather than scattered ``os.environ`` reads
or magic numbers in the code) is what lets the rest of the modules stay pure and
dependency-injected.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Point = tuple[float, float]
Side = Literal["left", "right"]


@dataclass(frozen=True, slots=True)
class SourceConfig:
    kind: Literal["file", "webcam", "stream"]
    target: str  # file path, webcam index, or stream URL


@dataclass(frozen=True, slots=True)
class DoorwayConfig:
    line_a: Point  # one end of the threshold line, in pixels
    line_b: Point  # the other end
    inside_side: Side  # which side of the line is "inside the office"


@dataclass(frozen=True, slots=True)
class Thresholds:
    detection_conf: float = 0.4  # min YOLO confidence for a person
    face_match: float = 0.35  # min cosine to accept a face as a known person
    face_clarity_min: float = 0.5  # face clarity that is "good enough" to lock identity early
    exit_similarity: float = 0.6  # min cosine to link an exit to an occupant
    exit_margin: float = 0.05  # best occupant must beat the runner-up by this, else ambiguous
    min_track_age: int = 3  # frames a track must exist before its crossing counts (anti-flicker)


@dataclass(frozen=True, slots=True)
class Paths:
    gallery_dir: Path = Path("gallery")
    database: Path = Path("data/occupancy.db")


@dataclass(frozen=True, slots=True)
class Config:
    source: SourceConfig
    doorway: DoorwayConfig
    thresholds: Thresholds = field(default_factory=Thresholds)
    paths: Paths = field(default_factory=Paths)


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
    )
    thresholds = Thresholds(**data.get("thresholds", {}))
    paths_raw = data.get("paths", {})
    paths = Paths(
        gallery_dir=Path(paths_raw.get("gallery_dir", "gallery")),
        database=Path(paths_raw.get("database", "data/occupancy.db")),
    )
    return Config(source=source, doorway=doorway, thresholds=thresholds, paths=paths)
