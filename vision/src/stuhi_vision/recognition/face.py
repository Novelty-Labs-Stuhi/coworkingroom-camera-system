"""Face detection, clarity scoring, and recognition at the doorway.

Wraps InsightFace, which bundles two models run in sequence: a *detection* model
(gives the face box, landmarks and a score -- the raw material for a clarity score)
and a *recognition* model (the ArcFace embedding used to tell who it is). The face
signal is clothes- and cap-invariant, so it is the durable identity used for naming.

InsightFace is imported lazily so the pure logic and tests never need it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..domain import Box
from .embeddings import Match, normalize
from .gallery import FaceGallery


@dataclass(frozen=True, slots=True)
class FaceObservation:
    """One detected face in a crop: its embedding and how clear/usable it is."""

    embedding: np.ndarray
    clarity: float  # 0..1, combines detection score, face size, and frontality


class FaceRecognizer:
    """Detect and score faces, and match them against an enrolled gallery."""

    def __init__(self, gallery: FaceGallery, match_threshold: float) -> None:
        self._gallery = gallery
        self._threshold = match_threshold
        self._app = None  # lazily initialised InsightFace FaceAnalysis

    def _model(self):
        if self._app is None:
            import onnxruntime
            from insightface.app import FaceAnalysis

            providers = onnxruntime.get_available_providers()
            use_gpu = "CUDAExecutionProvider" in providers
            app = FaceAnalysis(name="buffalo_l", providers=providers)
            app.prepare(ctx_id=0 if use_gpu else -1, det_size=(640, 640))
            self._app = app
        return self._app

    def analyze(self, image: np.ndarray, box: Box) -> FaceObservation | None:
        """Best face inside ``box`` with a clarity score, or ``None`` if none found."""
        crop = box.crop(image)
        if crop.size == 0:
            return None
        faces = self._model().get(crop)
        if not faces:
            return None
        best = max(faces, key=lambda f: f.det_score)
        embedding = normalize(np.asarray(best.embedding, dtype=np.float32))
        clarity = _clarity(best, crop.shape[0])
        return FaceObservation(embedding=embedding, clarity=clarity)

    def match(self, embedding: np.ndarray) -> Match | None:
        """Match a face embedding against the gallery (or ``None`` if unknown)."""
        return self._gallery.match(embedding, self._threshold)

    def embed(self, image: np.ndarray, box: Box) -> np.ndarray | None:
        """Face embedding only -- used for enrolling gallery photos."""
        observation = self.analyze(image, box)
        return observation.embedding if observation is not None else None


def _clarity(face, crop_height: int) -> float:
    """Combine detection confidence, face size and frontality into a 0..1 score."""
    score = float(np.clip(face.det_score, 0.0, 1.0))
    face_height = float(face.bbox[3] - face.bbox[1])
    size_factor = min(1.0, face_height / (0.4 * crop_height + 1e-6))
    return score * size_factor * _frontality(face.kps)


def _frontality(landmarks: np.ndarray) -> float:
    """1.0 for a head-on face, falling off as it turns (nose off the eye midpoint)."""
    left_eye, right_eye, nose = landmarks[0], landmarks[1], landmarks[2]
    eye_distance = float(np.linalg.norm(right_eye - left_eye)) + 1e-6
    midpoint_x = (left_eye[0] + right_eye[0]) / 2
    offset = abs(nose[0] - midpoint_x) / eye_distance
    return float(np.clip(1.0 - 2.0 * offset, 0.0, 1.0))
