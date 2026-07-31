"""Something is covering the doorframe: pixels, not detections.

The doorframe box is a piece of the picture whose appearance is known when nobody is in it. A
person standing in the doorway covers it, and the fraction of it that no longer matches is a
direct measure of that -- with a property the detector cannot match: **it gets stronger the
closer somebody is**, where a person detector gets weaker, because a body filling the frame is
out of distribution for one.

What it cannot do is say what the covering thing *is*. A swinging door, an arm reaching
through, a bag put down in the doorway and a person all read the same. So this module answers
only "is the box covered, and which way did the covering travel", and the tracker is what
establishes a person was there -- see :mod:`.passage`.

Two things keep it honest:

* **The background is learned only from uncovered frames.** A person standing in the doorway
  can therefore never be absorbed into it, while a door left open or a light switched on is.
* **The direction comes from where the covered region reaches**, not from correlating the box
  between frames. Measured on real passages, correlation reads backwards when the covering
  thing is also growing: somebody walking close past the lens swells across the box, and the
  correlation follows the swelling rather than the travel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_BACKGROUND_FRAMES = 60  # a minute or so of uncovered frames at this frame rate


@dataclass(frozen=True, slots=True)
class Coverage:
    """One finished episode of the box being covered."""

    frames: int
    peak: float
    travelled: float  # how far the near edge of the covering moved, as a fraction of the box

    @property
    def readable(self) -> str:
        return (
            f"box covered for {self.frames} frames, peak {self.peak:.2f}, "
            f"near edge moved {self.travelled:+.2f}"
        )


@dataclass(frozen=True, slots=True)
class CoverageConfig:
    """How much change counts as covered, and for how long."""

    # Fraction of the box that must differ from the background. Measured passages peaked at
    # 0.38-0.56 of the box, so this leaves room while staying clear of noise.
    covered: float = 0.25
    # Grey levels of difference that count as changed, per pixel.
    difference: int = 25
    # Frames of coverage before it is treated as an episode rather than a flicker.
    min_frames: int = 2
    background_frames: int = _BACKGROUND_FRAMES


@dataclass
class _Episode:
    frames: int = 0
    peak: float = 0.0
    # Where the covering sat in each frame: its near edge and its middle, in box columns.
    edges: list[int] = field(default_factory=list)
    middles: list[float] = field(default_factory=list)


class Occlusion:
    """Watches one box for something covering it."""

    def __init__(self, zone: tuple[float, float, float, float], config: CoverageConfig) -> None:
        self._zone = zone
        self._config = config
        self._recent: list = []
        self._background = None
        self._episode: _Episode | None = None
        self._covered = 0.0
        self._width = 1

    @property
    def covered(self) -> float:
        """The most recent reading, for reporting."""
        return self._covered

    @property
    def busy(self) -> bool:
        """Whether the box is covered right now."""
        return self._episode is not None

    def update(self, image) -> Coverage | None:
        """Feed one frame. Returns an episode only on the frame its coverage ends."""
        box = self._prepare(image)
        self._width = box.shape[1]
        changed = self._compare(box)
        self._covered = float(changed.mean()) if changed is not None else 0.0

        if self._covered >= self._config.covered and changed is not None:
            self._extend(changed)
            return None
        return self._finish(box)

    def _extend(self, changed) -> None:
        import numpy as np

        if self._episode is None:
            self._episode = _Episode()
        self._episode.frames += 1
        self._episode.peak = max(self._episode.peak, self._covered)
        columns = np.flatnonzero(changed.any(axis=0))
        if columns.size:
            self._episode.edges.append(int(columns[0]))
            # The middle of the covering, weighted by how much of each column is covered, so a
            # body sweeping across moves it steadily. Taken every frame rather than only at the
            # ends: an episode is often two to four frames, and a line through every point
            # reads a direction where the difference between two points is inside the noise.
            weights = changed.sum(axis=0)
            self._episode.middles.append(
                float((np.arange(changed.shape[1]) * weights).sum() / max(weights.sum(), 1))
            )

    def _finish(self, box) -> Coverage | None:
        """Coverage has ended: learn from this frame, and report the episode if it was one."""
        episode, self._episode = self._episode, None
        self._learn(box)
        if episode is None or episode.frames < self._config.min_frames:
            return None
        return Coverage(
            frames=episode.frames,
            peak=episode.peak,
            travelled=_travelled(episode.middles, self._width),
        )

    def _compare(self, box):
        if self._background is None or self._background.shape != box.shape:
            return None
        import cv2

        return cv2.absdiff(box, self._background) > self._config.difference

    def _learn(self, box) -> None:
        import numpy as np

        self._recent.append(box)
        if len(self._recent) > self._config.background_frames:
            self._recent.pop(0)
        self._background = np.median(np.stack(self._recent), axis=0).astype(box.dtype)

    def _prepare(self, image):
        import cv2

        height, width = image.shape[:2]
        x1, y1, x2, y2 = self._zone
        crop = image[
            int(y1 * height) : max(int(y2 * height), int(y1 * height) + 1),
            int(x1 * width) : max(int(x2 * width), int(x1 * width) + 1),
        ]
        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        return cv2.GaussianBlur(grey, (5, 5), 0)


def _travelled(middles: list[float], width: int) -> float:
    """How far the covering moved across the box, as a fraction of its width.

    A straight line fitted through every frame's middle, extended over the episode, rather
    than the difference between the first and last. With two to four frames -- which is what a
    passage at this frame rate gives -- the endpoints alone are dominated by however the
    covering happened to be shaped in those two frames, and read as no movement.
    """
    if len(middles) < 2:
        return 0.0
    import numpy as np

    frames = np.arange(len(middles), dtype=float)
    slope = float(np.polyfit(frames, np.array(middles, dtype=float), 1)[0])
    return slope * (len(middles) - 1) / max(width, 1)
