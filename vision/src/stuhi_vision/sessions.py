"""Per-track accumulation while a person is in view.

Recognition and embedding run every frame for everyone visible -- no entry/exit role
is assumed. Each track carries a :class:`TrackSession` that keeps:

* the best-face-so-far -> once a frame's face clarity is "good enough" the identity is
  recognized and locked (early accept, low latency); the body embedding from that same
  clear frame becomes the stored entry embedding;
* the sharpest body crop seen -> used as the exit query (and as an entry fallback when
  no clear face was ever seen).

Nothing is committed here. The doorway crossing decides whether a session becomes an
entry or an exit; sessions that never cross are pruned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .domain import Frame, TrackedPerson
from .quality import laplacian_sharpness
from .recognition.body import BodyEmbedder
from .recognition.face import FaceRecognizer

_MIN_CROP_SIDE = 40  # ignore crops smaller than this (too far / too little detail)
_EXPIRY_GRACE = 30  # frames a track may be unseen before its session is dropped


@dataclass(slots=True)
class TrackSession:
    track_id: int
    first_frame: int
    last_frame: int
    name: str | None = None
    locked: bool = False  # a good-enough face was seen; stop re-recognising
    best_face_clarity: float = -1.0
    entry_embedding: np.ndarray | None = None  # body embed from the clearest-face frame
    best_body_sharpness: float = -1.0
    body_embedding: np.ndarray | None = None  # sharpest body crop (exit query / fallback)

    @property
    def age(self) -> int:
        return self.last_frame - self.first_frame + 1


class SessionManager:
    """Owns the live sessions and updates them each frame."""

    def __init__(
        self, faces: FaceRecognizer, bodies: BodyEmbedder, face_clarity_min: float
    ) -> None:
        self._faces = faces
        self._bodies = bodies
        self._clarity_min = face_clarity_min
        self._sessions: dict[int, TrackSession] = {}

    def observe(self, frame: Frame, people: list[TrackedPerson], frame_index: int) -> None:
        for person in people:
            session = self._sessions.get(person.track_id)
            if session is None:
                session = TrackSession(person.track_id, frame_index, frame_index)
                self._sessions[person.track_id] = session
            session.last_frame = frame_index
            if not _too_small(frame, person):
                self._update_body(session, frame, person)
                if not session.locked:
                    self._update_face(session, frame, person)

    def pop(self, track_id: int) -> TrackSession | None:
        return self._sessions.pop(track_id, None)

    def prune(self, active_ids: set[int], frame_index: int) -> None:
        stale = [
            track_id
            for track_id, session in self._sessions.items()
            if track_id not in active_ids and frame_index - session.last_frame > _EXPIRY_GRACE
        ]
        for track_id in stale:
            del self._sessions[track_id]

    def _update_body(self, session: TrackSession, frame: Frame, person: TrackedPerson) -> None:
        sharpness = laplacian_sharpness(person.box.crop(frame.image))
        if sharpness > session.best_body_sharpness:
            embedding = self._bodies.embed(frame.image, person.box)
            if embedding is not None:
                session.best_body_sharpness = sharpness
                session.body_embedding = embedding

    def _update_face(self, session: TrackSession, frame: Frame, person: TrackedPerson) -> None:
        observation = self._faces.analyze(frame.image, person.box)
        if observation is None:
            return
        if observation.clarity > session.best_face_clarity:
            session.best_face_clarity = observation.clarity
            match = self._faces.match(observation.embedding)
            session.name = match.name if match is not None else None
            session.entry_embedding = self._bodies.embed(frame.image, person.box)
        if observation.clarity >= self._clarity_min:
            session.locked = True  # good enough -- accept early and stop re-recognising


def _too_small(frame: Frame, person: TrackedPerson) -> bool:
    crop = person.box.crop(frame.image)
    return crop.size == 0 or min(crop.shape[:2]) < _MIN_CROP_SIDE
