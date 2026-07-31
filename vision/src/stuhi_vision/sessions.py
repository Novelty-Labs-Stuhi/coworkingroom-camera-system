"""Per-track accumulation while a person is in view.

Recognition and embedding run every frame for everyone visible -- no entry/exit role is
assumed. Each track carries a :class:`TrackSession` that keeps:

* a :class:`~.identity.RunningIdentity` -> the best face match seen so far, plus the
  embedding and crop of the frame that produced it (kept so an unrecognised face can be
  labelled and enrolled afterwards);
* the sharpest body crop seen -> used as the exit query, since a person walking away
  shows no face.

Face clarity is still measured, but only recorded as diagnostic metadata -- it is not a
gate. See :mod:`.identity` for why the running maximum makes a separate gate unnecessary.

Nothing is committed here. The doorway crossing decides whether a session becomes an
entry or an exit; sessions that never cross are pruned.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .domain import Box, Frame, TrackedPerson
from .identity import RunningIdentity
from .quality import laplacian_sharpness
from .recognition.body import BodyEmbedder
from .recognition.face import FaceObservation, FaceRecognizer

_MIN_CROP_SIDE = 40  # ignore crops smaller than this (too far / too little detail)
_EXPIRY_GRACE = 30  # frames a track may be unseen before its session is dropped


@dataclass(slots=True)
class TrackSession:
    track_id: int
    first_frame: int
    last_frame: int
    identity: RunningIdentity = field(default_factory=lambda: RunningIdentity(0.0, 0.0))
    best_face_clarity: float = -1.0  # metadata only, for tuning the thresholds later
    face_embedding: np.ndarray | None = None  # face from the best-scoring frame
    face_crop: np.ndarray | None = None  # that same frame's crop, for labelling
    entry_embedding: np.ndarray | None = None  # body embed from the best-scoring frame
    best_body_sharpness: float = -1.0
    body_embedding: np.ndarray | None = None  # sharpest body crop (exit query / fallback)

    @property
    def age(self) -> int:
        return self.last_frame - self.first_frame + 1


class SessionManager:
    """Owns the live sessions and updates them each frame."""

    def __init__(
        self,
        faces: FaceRecognizer,
        bodies: BodyEmbedder,
        face_match: float,
        face_margin: float,
        face_workers: int = 4,
    ) -> None:
        self._faces = faces
        self._bodies = bodies
        self._face_match = face_match
        self._face_margin = face_margin
        self._face_workers = max(1, face_workers)
        self._pool: ThreadPoolExecutor | None = None
        self._sessions: dict[int, TrackSession] = {}

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    def observe(self, frame: Frame, people: list[TrackedPerson], frame_index: int) -> None:
        for person in people:
            session = self._sessions.get(person.track_id)
            if session is None:
                session = TrackSession(
                    person.track_id,
                    frame_index,
                    frame_index,
                    identity=RunningIdentity(self._face_match, self._face_margin),
                )
                self._sessions[person.track_id] = session
            session.last_frame = frame_index

        usable = [person for person in people if not _too_small(frame, person)]
        # Face analysis is the expensive part and is independent per person, so run it
        # concurrently; onnxruntime releases the GIL, so threads genuinely overlap.
        # Session state is then folded in sequentially, keeping mutation single-threaded.
        observations = self._analyze_faces(frame, usable)
        for person, observation in zip(usable, observations, strict=True):
            self._update_body(self._sessions[person.track_id], frame, person)
            if observation is not None:
                self._apply_face(self._sessions[person.track_id], frame, person, observation)

    def _analyze_faces(
        self, frame: Frame, people: list[TrackedPerson]
    ) -> list[FaceObservation | None]:
        if not people:
            return []
        if len(people) == 1 or self._face_workers == 1:
            return [self._faces.analyze(frame.image, person.box) for person in people]
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=self._face_workers, thread_name_prefix="face"
            )
        return list(
            self._pool.map(lambda person: self._faces.analyze(frame.image, person.box), people)
        )

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

    def _apply_face(
        self,
        session: TrackSession,
        frame: Frame,
        person: TrackedPerson,
        observation: FaceObservation,
    ) -> None:
        if not session.identity.observe(self._faces.rank(observation.embedding)):
            return
        # This frame is the best look at the face so far -- keep everything from it.
        session.best_face_clarity = observation.clarity
        session.face_embedding = observation.embedding
        session.face_crop = _portrait(frame.image, observation.box or person.box)
        session.entry_embedding = self._bodies.embed(frame.image, person.box)


def _portrait(image, box: Box):
    """The face with room around it: what somebody labels from.

    The whole person box was kept before, and at this doorway that is mostly torso -- a face a
    few dozen pixels across inside a body-sized picture is not something you can put a name to.
    Padding is generous rather than tight: a face cropped to its own edges loses the hair, ears
    and jaw, which is much of what a person is recognised by, and a mis-detected box would cut
    the face in half.
    """
    padding = 0.6
    width, height = box.x2 - box.x1, box.y2 - box.y1
    grown = Box(
        box.x1 - width * padding,
        box.y1 - height * padding,
        box.x2 + width * padding,
        box.y2 + height * padding,
    )
    crop = grown.crop(image)
    return crop.copy() if crop.size else box.crop(image).copy()


def _too_small(frame: Frame, person: TrackedPerson) -> bool:
    crop = person.box.crop(frame.image)
    return crop.size == 0 or min(crop.shape[:2]) < _MIN_CROP_SIDE
