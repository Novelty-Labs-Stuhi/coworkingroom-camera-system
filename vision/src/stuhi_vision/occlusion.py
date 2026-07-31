"""Something is covering the doorframe: pixels, in vertical slices.

The doorframe box is a piece of the picture whose appearance is known when nobody is in it. A
person standing in the doorway covers it, and the fraction of it that no longer matches
measures that directly -- with a property no detector has: **it gets stronger the closer
somebody is**, where a person detector gets weaker, a body filling the frame being out of
distribution for one.

The box is split into vertical slices, and each is measured separately. That gives the
direction for free: somebody crossing covers the slices **in order**, and the order is which
way they went. It is also coarse on purpose -- a slice is many pixels wide, so noise in a few
of them cannot move the reading, where a per-column measure can.

What pixels cannot say is what the covering thing *is*. A swinging door, an arm reaching
through, a bag set down in the doorway and a person all read alike. So this module answers only
"is the box covered, in which slices, and in what order", and the tracker establishes that a
person was there -- see :mod:`.passage`.

Two things keep it honest:

* **The background is learned only from uncovered frames**, so a person standing in the
  doorway can never be absorbed into it, while a door left open or a light switched on is.
* **Direction comes from the order the slices light**, not from correlating the box between
  frames. Measured on real passages, correlation reads backwards when the covering is also
  growing: somebody passing close to the lens swells across the box, and correlation follows
  the swelling rather than the travel.

Whether there are enough frames for an order to exist at all is a property of the camera, not
of this code: use ``tools/measure_box_slices.py`` on recorded frames to see the timeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_SLICES = 4  # across the box; enough for an order, wide enough to ignore pixel noise
_BACKGROUND_FRAMES = 60


@dataclass(frozen=True, slots=True)
class CoverageConfig:
    """How much change counts as covered, and for how long."""

    # Fraction of a slice that must differ from the background for that slice to count as
    # covered. Measured passages covered 0.38-0.56 of the whole box at their peak.
    covered: float = 0.25
    difference: int = 25  # grey levels of difference, per pixel
    slices: int = _SLICES
    min_frames: int = 2  # frames of coverage before it is an episode rather than a flicker
    background_frames: int = _BACKGROUND_FRAMES


@dataclass(frozen=True, slots=True)
class Coverage:
    """One finished episode of the box being covered, and how it moved across the slices."""

    frames: int
    peak: float
    travelled: float          # slices crossed, signed; negative is towards slice zero
    timeline: tuple[tuple[float, ...], ...]   # per frame, coverage of each slice

    @property
    def readable(self) -> str:
        return (
            f"box covered for {self.frames} frames, peak {self.peak:.2f}, "
            f"moved {self.travelled:+.1f} slices"
        )

    @property
    def picture(self) -> str:
        """The timeline as one line per frame, for reading in a log or a terminal."""
        return "\n".join(
            "    " + " ".join(f"{value:.2f}" for value in row) for row in self.timeline
        )


@dataclass
class _Episode:
    frames: int = 0
    peak: float = 0.0
    rows: list[tuple[float, ...]] = field(default_factory=list)
    middles: list[float] = field(default_factory=list)


class Occlusion:
    """Watches one box, in vertical slices, for something covering it."""

    def __init__(self, zone: tuple[float, float, float, float], config: CoverageConfig) -> None:
        self._zone = zone
        self._config = config
        self._recent: list = []
        self._background = None
        self._episode: _Episode | None = None
        self._slices: tuple[float, ...] = ()

    @property
    def slices(self) -> tuple[float, ...]:
        """The most recent per-slice coverage, for reporting."""
        return self._slices

    @property
    def busy(self) -> bool:
        return self._episode is not None

    def update(self, image) -> Coverage | None:
        """Feed one frame. Returns an episode only on the frame its coverage ends."""
        box = self._prepare(image)
        self._slices = self._measure(box)
        covered = max(self._slices) if self._slices else 0.0

        if covered >= self._config.covered:
            self._extend()
            return None
        return self._finish(box)

    def _measure(self, box) -> tuple[float, ...]:
        """The changed fraction of each vertical slice of the box."""
        if self._background is None or self._background.shape != box.shape:
            return ()
        import cv2
        import numpy as np

        changed = cv2.absdiff(box, self._background) > self._config.difference
        columns = np.array_split(changed, self._config.slices, axis=1)
        return tuple(float(part.mean()) for part in columns)

    def _extend(self) -> None:
        if self._episode is None:
            self._episode = _Episode()
        self._episode.frames += 1
        self._episode.peak = max(self._episode.peak, *self._slices)
        self._episode.rows.append(self._slices)
        middle = _middle(self._slices)
        if middle is not None:
            self._episode.middles.append(middle)

    def _finish(self, box) -> Coverage | None:
        """Coverage has ended: learn from this frame, and report the episode if it was one."""
        episode, self._episode = self._episode, None
        self._learn(box)
        if episode is None or episode.frames < self._config.min_frames:
            return None
        return Coverage(
            frames=episode.frames,
            peak=episode.peak,
            travelled=_travelled(episode.middles),
            timeline=tuple(episode.rows),
        )

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


def _middle(slices: tuple[float, ...]) -> float | None:
    """Where the covering sits across the slices, weighted by how covered each one is."""
    total = sum(slices)
    if total <= 0:
        return None
    return sum(index * value for index, value in enumerate(slices)) / total


def _travelled(middles: list[float]) -> float:
    """How many slices the covering crossed, from a line fitted through every frame.

    Every frame rather than first-versus-last: an episode can be two or three frames, and the
    endpoints alone are dominated by whatever shape the covering happened to have in those two.
    """
    if len(middles) < 2:
        return 0.0
    import numpy as np

    frames = np.arange(len(middles), dtype=float)
    slope = float(np.polyfit(frames, np.array(middles, dtype=float), 1)[0])
    return slope * (len(middles) - 1)
