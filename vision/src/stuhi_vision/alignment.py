"""Notice when a camera has been moved, so a hand-drawn zone can be redrawn.

Every zone, edge and margin is expressed in the camera's own frame. That is fine until
somebody knocks the camera: the numbers stay valid and stop meaning anything, and the
failure is silent -- passages simply stop being counted, or start being counted wrongly.
Nothing in the pictures looks broken.

So a reference frame is kept from the moment the zone was drawn, and the live view is
compared against it. Phase correlation gives the translation between two images directly, in
pixels, which is exactly the quantity worth alerting on: "the camera has shifted 20 px" is
actionable, where a similarity score is not.

Two things keep it from crying wolf:

* **Only look at empty frames.** A person walking through is a large moving object and drags
  the correlation with them. The pipeline already knows whether it detected anybody, so
  checks happen only when it saw nobody.
* **Require several consecutive readings.** One frame can disagree because of noise,
  auto-exposure, or a door swinging in the background.

Rotation and zoom are not measured. A camera twisted in place can therefore go unnoticed,
which is worth knowing rather than pretending otherwise -- though in practice a knock that
rotates a camera usually shifts it too.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_WIDTH = 240  # frames are compared at this width: enough detail, cheap, and noise-tolerant


@dataclass(frozen=True, slots=True)
class Drift:
    """How far the view has moved from its reference, in pixels of the original frame."""

    shift_x: float
    shift_y: float
    magnitude: float
    confidence: float

    @property
    def readable(self) -> str:
        return f"{self.magnitude:.0f} px ({self.shift_x:+.0f}, {self.shift_y:+.0f})"


class DriftWatch:
    """Compares the live view against the frame the zone was drawn on."""

    def __init__(
        self,
        reference_path: Path,
        tolerance_px: float = 15.0,
        confirmations: int = 4,
        min_confidence: float = 0.15,
    ) -> None:
        self._path = reference_path
        self._tolerance = tolerance_px
        self._confirmations = max(1, confirmations)
        self._min_confidence = min_confidence
        self._reference = None  # the downscaled, windowed reference
        self._scale = 1.0
        # The reference's *original* size. Comparing the downscaled shapes is not enough:
        # every 4:3 frame reduces to the same 240x180, so 320x240 would silently be measured
        # against a 640x480 reference and report a large invented shift.
        self._size: tuple[int, int] | None = None
        self._over = 0
        self._latest: Drift | None = None
        self._load()

    @property
    def latest(self) -> Drift | None:
        return self._latest

    @property
    def has_reference(self) -> bool:
        return self._reference is not None

    @property
    def tolerance_px(self) -> float:
        """How far the view may shift before it stops matching the zone drawn on it."""
        return self._tolerance

    def remember(self, image) -> None:
        """Adopt this frame as the reference -- called when a zone is drawn."""
        import cv2

        self._path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self._path), image)
        self._reference, self._scale = _prepare(image)
        self._size = (image.shape[1], image.shape[0])
        self._over = 0
        self._latest = None

    def forget(self) -> None:
        """Stop watching this view -- called when the zone it was drawn for is removed.

        The reference image is left on disk. It costs nothing, and deleting the only record of
        what the camera used to see, to serve a button press, is a poor trade.
        """
        self._reference = None
        self._size = None
        self._over = 0
        self._latest = None

    def check(self, image) -> Drift | None:
        """Measure the shift of an *empty* frame. ``None`` until there is a reference."""
        if self._reference is None:
            return None
        import cv2

        if self._size != (image.shape[1], image.shape[0]):
            return None  # a resolution change is not drift; it is a different camera
        current, _ = _prepare(image)
        (shift_x, shift_y), confidence = cv2.phaseCorrelate(self._reference, current)
        drift = Drift(
            shift_x=shift_x / self._scale,
            shift_y=shift_y / self._scale,
            magnitude=float((shift_x**2 + shift_y**2) ** 0.5) / self._scale,
            confidence=float(confidence),
        )
        self._latest = drift

        if drift.confidence < self._min_confidence:
            return drift  # a featureless or very dark frame says nothing either way
        self._over = self._over + 1 if drift.magnitude > self._tolerance else 0
        return drift

    @property
    def has_moved(self) -> bool:
        """True once the shift has exceeded tolerance on several consecutive empty frames."""
        return self._over >= self._confirmations

    def acknowledge(self) -> None:
        """Stop reporting the current movement, without adopting the new view.

        Used after warning somebody: the camera has still moved, and the zone is still wrong,
        but there is no point repeating it every frame until they redraw it.
        """
        self._over = 0

    def _load(self) -> None:
        if not self._path.exists():
            return
        import cv2

        image = cv2.imread(str(self._path))
        if image is not None:
            self._reference, self._scale = _prepare(image)
            self._size = (image.shape[1], image.shape[0])


def _prepare(image):
    """Greyscale, downscale and window a frame for phase correlation.

    The Hanning window matters: without it the frame edges dominate the transform and a
    shift is measured against the border rather than the scene.
    """
    import cv2
    import numpy as np

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    height, width = grey.shape[:2]
    scale = _WIDTH / width
    small = cv2.resize(grey, (_WIDTH, max(1, int(height * scale))))
    windowed = small.astype(np.float32)
    window = cv2.createHanningWindow((windowed.shape[1], windowed.shape[0]), cv2.CV_32F)
    return windowed * window, scale
