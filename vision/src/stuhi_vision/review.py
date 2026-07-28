"""Sightings awaiting a human verdict, and the labelling that teaches the gallery.

Every committed crossing is filed here with the face embedding and crop of its best
frame. That turns "we do not know this person" from a dead end into the mechanism by
which the system improves: label the sighting and its embedding is enrolled into the
gallery, so the next time that person walks in they are recognised.

Labelling an already-labelled sighting *corrects* it -- the embedding is removed from the
wrong name before being added to the right one, so one mistake does not poison the
gallery forever.

Layout under the review directory, one set per sighting id::

    2026-07-27_18-04-11.npy    the face embedding
    2026-07-27_18-04-11.jpg    the face crop, for sending to a human
    index.json                 the sighting records

Note the SQLite occupancy event keeps the name it was recorded with; relabelling teaches
the gallery and fixes the review record, but does not rewrite history.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np

from .domain import Outcome, Sighting
from .recognition.gallery import FaceGallery

_INDEX = "index.json"


class LabelOutcome(Enum):
    """What a labelling request actually did."""

    ENROLLED = "enrolled"  # first label for this sighting; one reference added
    CORRECTED = "corrected"  # moved from a previous name; no duplicate left behind
    UNCHANGED = "unchanged"  # already labelled that way, so nothing was added
    NO_FACE = "no_face"  # nothing to enrol -- no embedding was kept
    UNKNOWN_ID = "unknown_id"

    @property
    def succeeded(self) -> bool:
        return self in (LabelOutcome.ENROLLED, LabelOutcome.CORRECTED, LabelOutcome.UNCHANGED)


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    """One sighting as filed for review."""

    sighting_id: str
    timestamp: float
    direction: str
    outcome: str
    name: str | None
    score: float
    labelled_as: str | None = None
    # Place within a burst -- a group of people who crossed together with no clear gap.
    # Their clips are cut from the same window and look alike, so the order they crossed in
    # is the only thing distinguishing them, and it is how a whole group gets labelled at
    # once. Defaults keep records written before this existed loadable.
    position: int = 1
    burst: int = 0

    @property
    def display_name(self) -> str:
        return self.labelled_as or self.name or "unknown"


class ReviewQueue:
    """Disk-backed sightings plus the label/correct operations over the gallery."""

    def __init__(self, directory: Path, gallery: FaceGallery, gallery_dir: Path) -> None:
        self._dir = directory
        self._gallery = gallery
        self._gallery_dir = gallery_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, ReviewRecord] = {}
        # Sightings are filed by the pipeline thread and labelled by the chat poller and
        # the web UI, so the record index needs guarding as much as the gallery does.
        self._lock = threading.RLock()
        self._load()

    # --- recording ----------------------------------------------------------
    def labelled(self, limit: int = 50) -> list[ReviewRecord]:
        """Sightings that already carry a label, newest first, so one can be corrected."""
        with self._lock:
            done = [r for r in self._records.values() if r.labelled_as is not None]
        done.sort(key=lambda record: record.timestamp, reverse=True)
        return done[:limit]

    def record(
        self,
        sighting: Sighting,
        encode_jpeg=None,
        clip: bytes | None = None,
        position: int = 1,
        burst: int = 0,
    ) -> str:
        """File a sighting and return its id.

        ``encode_jpeg`` turns the crop into JPEG bytes; injected so this module needs no
        image library (and so tests need no OpenCV). ``clip`` is an optional MP4 of the
        moment, kept alongside so a human can see what happened rather than only a crop.
        """
        sighting_id = self._next_id(sighting.timestamp)
        if sighting.face_embedding is not None:
            np.save(self._dir / f"{sighting_id}.npy", sighting.face_embedding)
        if sighting.face_crop is not None and encode_jpeg is not None:
            jpeg = encode_jpeg(sighting.face_crop)
            if jpeg:
                (self._dir / f"{sighting_id}.jpg").write_bytes(jpeg)
        if clip:
            (self._dir / f"{sighting_id}.mp4").write_bytes(clip)
        self._records[sighting_id] = ReviewRecord(
            sighting_id=sighting_id,
            timestamp=sighting.timestamp,
            direction=sighting.direction.value,
            outcome=sighting.outcome.value,
            name=sighting.name,
            score=round(sighting.score, 4),
            position=position,
            burst=burst,
        )
        self._flush()
        return sighting_id

    # --- labelling ----------------------------------------------------------
    def label(self, sighting_id: str, name: str) -> LabelOutcome:
        """Enrol a sighting's face under ``name``, correcting any previous label.

        Labelling the same sighting the same way twice is deliberately a no-op: one
        sighting contributes exactly one reference vector, however many times it is
        labelled. Otherwise a repeated command would quietly weight that one face more
        heavily than everybody else's.

        The outcome is reported rather than a bare success flag, so the person labelling
        can tell "added" from "already like that" instead of guessing.
        """
        with self._lock:
            record = self._records.get(sighting_id)
            if record is None:
                return LabelOutcome.UNKNOWN_ID
            embedding_path = self._dir / f"{sighting_id}.npy"
            if not embedding_path.exists():
                return LabelOutcome.NO_FACE

            previous = record.labelled_as
            embedding = np.load(embedding_path)

            # Two things can make this a no-op: the record already says this name, or the
            # vector is already enrolled under it (the same sighting labelled from the chat
            # and from the web UI). Either way one sighting must count exactly once.
            if previous == name or self._gallery.contains(name, embedding):
                if previous != name:
                    self._remember(record, name)
                return LabelOutcome.UNCHANGED

            if previous is not None:
                self._gallery.discard(previous, embedding)
            self._gallery.add(name, embedding)
            self._gallery.save(self._gallery_dir)
            self._remember(record, name)
            return LabelOutcome.CORRECTED if previous is not None else LabelOutcome.ENROLLED

    def label_burst(self, sighting_id: str, names: list[str]) -> list[tuple[str, LabelOutcome]]:
        """Label everyone who crossed alongside ``sighting_id``, in crossing order.

        ``names[0]`` goes to whoever crossed first, and so on. Extra names are ignored and
        missing ones leave that person unlabelled, so a mistaken count cannot silently
        attach the wrong name to somebody.
        """
        members = self.burst_members(sighting_id)
        return [
            (member.sighting_id, self.label(member.sighting_id, name))
            for member, name in zip(members, names, strict=False)
        ]

    def burst_members(self, sighting_id: str) -> list[ReviewRecord]:
        """Everyone who crossed in the same burst as this sighting, in crossing order."""
        with self._lock:
            record = self._records.get(sighting_id)
            if record is None:
                return []
            if not record.burst:  # written before bursts existed: it stands alone
                return [record]
            members = [r for r in self._records.values() if r.burst == record.burst]
        return sorted(members, key=lambda r: r.position)

    def _remember(self, record: ReviewRecord, name: str) -> None:
        self._records[record.sighting_id] = ReviewRecord(**{**asdict(record), "labelled_as": name})
        self._flush()

    # --- queries ------------------------------------------------------------
    def crop_path(self, sighting_id: str) -> Path | None:
        path = self._dir / f"{sighting_id}.jpg"
        return path if path.exists() else None

    def clip_path(self, sighting_id: str) -> Path | None:
        path = self._dir / f"{sighting_id}.mp4"
        return path if path.exists() else None

    def pending(self, limit: int = 20) -> list[ReviewRecord]:
        """Unlabelled sightings that have a face to enrol, newest first."""
        candidates = [
            record
            for record in self._records.values()
            if record.labelled_as is None
            and record.outcome != Outcome.UNIDENTIFIED.value
            and (self._dir / f"{record.sighting_id}.npy").exists()
        ]
        candidates.sort(key=lambda record: record.timestamp, reverse=True)
        return candidates[:limit]

    def counts(self) -> dict[str, int]:
        """Enrolled reference vectors per name."""
        return self._gallery.counts()

    # --- persistence --------------------------------------------------------
    def _next_id(self, timestamp: float) -> str:
        base = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d_%H-%M-%S")
        if base not in self._records:
            return base
        suffix = 2
        while f"{base}-{suffix}" in self._records:
            suffix += 1
        return f"{base}-{suffix}"

    def _load(self) -> None:
        path = self._dir / _INDEX
        if not path.exists():
            return
        raw = json.loads(path.read_text(encoding="utf-8"))
        for entry in raw:
            record = ReviewRecord(**entry)
            self._records[record.sighting_id] = record

    def _flush(self) -> None:
        payload = [asdict(record) for record in self._records.values()]
        (self._dir / _INDEX).write_text(json.dumps(payload, indent=2), encoding="utf-8")
