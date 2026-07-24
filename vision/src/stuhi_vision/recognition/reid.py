"""Clip-level person re-ID: detect the person and embed their appearance (OSNet).

Composes the existing YOLO :class:`PersonDetector` with a torchreid OSNet feature
extractor. One clip -> one L2-normalised appearance vector (the average over every
frame in which a person was found). Everything heavy (torch, ultralytics, torchreid)
is imported lazily so this module stays cheap to import.
"""

from __future__ import annotations

import numpy as np

from ..detection import PersonDetector
from .embeddings import normalize


def _largest_person_crop(frame_bgr: np.ndarray, detector: PersonDetector) -> np.ndarray | None:
    """Return the largest person crop in a BGR frame as RGB, or None."""
    boxes = detector.detect(frame_bgr)
    if not boxes:
        return None
    box = max(boxes, key=lambda b: (b.x2 - b.x1) * (b.y2 - b.y1))
    crop = box.crop(frame_bgr)
    if crop.size == 0:
        return None
    return crop[:, :, ::-1].copy()  # BGR -> RGB for torchreid


class ClipEmbedder:
    """Turn frames (or a clip) into one averaged OSNet appearance vector."""

    def __init__(self, model_name: str = "osnet_x1_0", device: str = "cpu") -> None:
        self._model_name = model_name
        self._device = device
        self._detector: PersonDetector | None = None
        self._extractor = None

    def _ensure_loaded(self) -> None:
        if self._extractor is not None:
            return
        try:  # package layout moved between torchreid releases
            from torchreid.reid.utils import FeatureExtractor
        except ModuleNotFoundError:
            from torchreid.utils import FeatureExtractor
        self._detector = PersonDetector()
        self._extractor = FeatureExtractor(
            model_name=self._model_name, model_path="", device=self._device
        )

    def embed_frame(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """Detect the person in one BGR frame and embed them, or None if absent."""
        self._ensure_loaded()
        crop = _largest_person_crop(frame_bgr, self._detector)
        if crop is None:
            return None
        features = self._extractor([crop]).cpu().numpy()  # (1, dim)
        return normalize(features[0].astype(np.float32))

    def embed_frames(self, frames_bgr: list[np.ndarray]) -> np.ndarray | None:
        """Average the per-frame embeddings over a sequence of BGR frames."""
        vectors = [v for v in (self.embed_frame(f) for f in frames_bgr) if v is not None]
        if not vectors:
            return None
        return normalize(np.mean(vectors, axis=0).astype(np.float32))

    def embed_jpegs(self, jpegs: list[bytes]) -> np.ndarray | None:
        """Decode a list of JPEG byte-strings and embed them (the clip-upload path)."""
        import cv2

        frames = []
        for jpg in jpegs:
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(img)
        return self.embed_frames(frames)

    def embed_video(self, path) -> np.ndarray | None:
        """Read every frame of a video file and embed the person across them."""
        import cv2

        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        return self.embed_frames(frames)
