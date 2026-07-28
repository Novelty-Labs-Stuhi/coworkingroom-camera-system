"""Decide which frames deserve the expensive models.

Detection and recognition cost hundreds of milliseconds each; comparing a frame to the
previous one costs about a millisecond. So an empty doorway should cost almost nothing,
leaving the whole machine free for the second when someone actually walks through.

The measure is the fraction of pixels that changed materially since the last frame, on a
downscaled greyscale copy (downscaling is both faster and a cheap noise filter). Above a
threshold the frame is *active*.

Two details that matter more than the threshold:

* **Hysteresis.** After activity, a number of following frames pass regardless. A person
  pausing mid-stride would otherwise flicker the gate shut and break the track.
* **The reference is the previous frame, not a long-term background.** A background model
  would be better at ignoring a swaying blind, but it also learns a stationary person into
  the background, which is exactly who we care about.
"""

from __future__ import annotations

import numpy as np

_WIDTH = 160  # width of the downscaled comparison image
_PIXEL_DELTA = 18  # per-pixel intensity change counted as "different" (0..255)


class MotionGate:
    """Stateful frame-to-frame change detector with hysteresis."""

    def __init__(self, min_fraction: float = 0.004, linger_frames: int = 12) -> None:
        self._min_fraction = min_fraction
        self._linger = linger_frames
        self._previous: np.ndarray | None = None
        self._remaining = 0
        self._last_fraction = 0.0

    @property
    def last_fraction(self) -> float:
        """Fraction of pixels that changed on the most recent frame."""
        return self._last_fraction

    def is_active(self, image: np.ndarray) -> bool:
        """True when this frame should go to the expensive models."""
        small = self._prepare(image)
        previous, self._previous = self._previous, small
        if previous is None:
            self._remaining = self._linger  # no reference yet: assume active
            return True

        difference = np.abs(small.astype(np.int16) - previous.astype(np.int16))
        self._last_fraction = float(np.count_nonzero(difference > _PIXEL_DELTA) / small.size)

        if self._last_fraction >= self._min_fraction:
            self._remaining = self._linger
            return True
        if self._remaining > 0:
            self._remaining -= 1
            return True
        return False

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        import cv2

        height, width = image.shape[:2]
        scale = _WIDTH / max(width, 1)
        small = cv2.resize(image, (_WIDTH, max(int(height * scale), 1)))
        if small.ndim == 3:
            small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(small, (5, 5), 0)
