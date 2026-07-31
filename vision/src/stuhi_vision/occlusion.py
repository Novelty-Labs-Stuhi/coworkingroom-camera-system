"""Something is covering the doorframe, and in what order: pixels, not detections.

The box is split into vertical slices, and each slice is watched for its pixels changing. Two
things fall out of that, which is why it is worth doing rather than watching the box as a
whole:

* **Movement**: any slice changing means something is on the doorframe. This is the one signal
  that gets *stronger* as somebody comes closer, where a person detector gets weaker -- a body
  filling the frame is out of distribution for one.
* **Direction**: the *order* the slices light up. Somebody walking through covers the near
  slice first and the far slice last, or the reverse, and which it is says in or out. An order
  is a sequence of events rather than a measurement of shape, so it survives the thing that
  defeated every earlier attempt: a person's outline swelling as they approach the lens, which
  made both a correlation and a centre-of-mass read backwards.
* **Neither**: slices that light without an order -- the same ones covered throughout -- is
  somebody standing in the doorway rather than passing through it. That case is now
  distinguishable instead of being a guess.

What pixels cannot say is what the covering thing *is*. A swinging door, an arm reaching
through and a person read alike, so the tracker is what establishes a person was there --
see :mod:`.passage`.

The background each slice is compared against is learned only from frames where that slice is
uncovered, so somebody standing in the doorway is never absorbed into it, while a door left
open or a light switched on is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_SLICES = 5  # enough to give an order across a doorframe; few enough that each stays reliable


@dataclass(frozen=True, slots=True)
class CoverageConfig:
    """How much change counts as covered, and what counts as a sweep across the box."""

    slices: int = _SLICES
    # Fraction of a slice that must differ from its background for that slice to be covered.
    covered: float = 0.25
    # Grey levels of difference that count as changed, per pixel.
    difference: int = 25
    # Frames of coverage before it is an episode rather than a flicker.
    min_frames: int = 2
    # Slices that must be covered at some point for a passage to be possible. Two is the
    # minimum that can carry an order; more would refuse somebody who clipped the doorframe.
    min_slices: int = 2
    # Frames of lag per slice before an order counts as a direction rather than noise. At a
    # few frames a second a person crosses a slice in well under a frame, so this is small.
    min_lag: float = 0.25
    # How slowly the background follows the view: the reciprocal is the weight each new
    # uncovered frame gets. A running average, not a median over stored frames -- the median
    # cost a sort of millions of elements *per frame* and pinned the machine.
    background_frames: int = 60
    # Width the box is compared at. Downscaling costs nothing and removes sensor noise; the
    # slices are read as fractions of it, so it changes no threshold.
    compare_width: int = 160
    # Coverage lasting longer than this is the view having changed, not somebody passing: a
    # light switched on, furniture moved, the door left open in a new position. Without it the
    # background can never be relearned -- it only learns from uncovered frames -- so one
    # lighting change would jam the detector permanently, reporting coverage for ever and
    # never counting another passage. Twenty seconds or so at this frame rate; nobody walks
    # through a doorway that slowly.
    stuck_after: int = 90


@dataclass(frozen=True, slots=True)
class Coverage:
    """One finished episode of the box being covered."""

    frames: int
    peak: float
    slices: int          # how many slices were covered at some point
    lag: float           # frames of delay per slice: negative means the far side lit first
    order: tuple[float, ...]   # when each slice first lit, in frames from the episode's start

    @property
    def swept(self) -> bool:
        """Whether the covering crossed the box rather than sitting in it."""
        return self.lag != 0.0

    @property
    def readable(self) -> str:
        way = "no order" if not self.swept else f"lag {self.lag:+.2f} frames per slice"
        return (
            f"box covered for {self.frames} frames, {self.slices} slices, "
            f"peak {self.peak:.2f}, {way}"
        )


@dataclass
class _Episode:
    frames: int = 0
    peak: float = 0.0
    # The frame within the episode each slice was first covered on.
    first_seen: dict[int, int] = field(default_factory=dict)


class Occlusion:
    """Watches one box, sliced vertically, for something covering it."""

    def __init__(self, zone: tuple[float, float, float, float], config: CoverageConfig) -> None:
        self._zone = zone
        self._config = config
        self._background = None   # float32, so the average does not quantise away
        self._episode: _Episode | None = None
        self._covered = 0.0

    @property
    def covered(self) -> float:
        """The most recent reading: the fraction of the whole box that changed."""
        return self._covered

    @property
    def busy(self) -> bool:
        return self._episode is not None

    def update(self, image) -> Coverage | None:
        """Feed one frame. Returns an episode only on the frame its coverage ends."""
        box = self._prepare(image)
        changed = self._compare(box)
        if changed is None:
            self._covered = 0.0
            self._learn(box)
            return None

        per_slice = _by_slice(changed, self._config.slices)
        self._covered = float(changed.mean())
        threshold = self._config.covered
        lit = [index for index, fraction in enumerate(per_slice) if fraction >= threshold]

        if lit:
            return self._extend(lit, max(per_slice), box)
        return self._finish(box)

    def _extend(self, lit: list[int], peak: float, box) -> Coverage | None:
        if self._episode is None:
            self._episode = _Episode()
        episode = self._episode
        for index in lit:
            episode.first_seen.setdefault(index, episode.frames)
        episode.frames += 1
        episode.peak = max(episode.peak, peak)

        if episode.frames >= self._config.stuck_after:
            # Whatever this is, it is the view now. Adopt it and forget the episode: reporting
            # a passage from it would be false, and waiting for it to clear could be for ever.
            self._episode = None
            self._adopt(box)
        return None

    def _finish(self, box) -> Coverage | None:
        episode, self._episode = self._episode, None
        self._learn(box)
        if episode is None or episode.frames < self._config.min_frames:
            return None
        return Coverage(
            frames=episode.frames,
            peak=episode.peak,
            slices=len(episode.first_seen),
            lag=self._lag(episode.first_seen),
            order=tuple(
                float(episode.first_seen.get(index, -1)) for index in range(self._config.slices)
            ),
        )

    def _lag(self, first_seen: dict[int, int]) -> float:
        """Frames of delay per slice across the box: the slope of when each slice lit.

        Positive means the low-numbered (left) slices lit first, so the covering travelled
        rightwards. Zero means no order worth reading -- somebody standing in the doorway, or a
        single slice covered, or every slice lighting in the same frame.
        """
        if len(first_seen) < self._config.min_slices:
            return 0.0
        import numpy as np

        indices = np.array(sorted(first_seen), dtype=float)
        times = np.array([first_seen[int(index)] for index in indices], dtype=float)
        if times.max() == times.min():
            return 0.0
        slope = float(np.polyfit(indices, times, 1)[0])
        return slope if abs(slope) >= self._config.min_lag else 0.0

    def _compare(self, box):
        if self._background is None or self._background.shape != box.shape:
            return None

        import numpy as np

        return np.abs(box - self._background) > self._config.difference

    def _adopt(self, box) -> None:
        """Start the background again from this frame: the view itself has changed."""
        self._background = box.copy()
        self._covered = 0.0

    def _learn(self, box) -> None:
        """Let the background drift towards this frame. Only uncovered frames reach here."""
        if self._background is None:
            self._background = box.copy()
            return
        weight = 1.0 / max(self._config.background_frames, 1)
        self._background += weight * (box - self._background)

    def _prepare(self, image):
        import cv2

        height, width = image.shape[:2]
        x1, y1, x2, y2 = self._zone
        crop = image[
            int(y1 * height) : max(int(y2 * height), int(y1 * height) + 1),
            int(x1 * width) : max(int(x2 * width), int(x1 * width) + 1),
        ]
        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        wide = self._config.compare_width
        if grey.shape[1] > wide:
            scale = wide / grey.shape[1]
            grey = cv2.resize(grey, (wide, max(1, int(grey.shape[0] * scale))))
        return cv2.GaussianBlur(grey, (5, 5), 0).astype("float32")


def _by_slice(changed, slices: int) -> list[float]:
    """The fraction of each vertical slice that changed."""
    from itertools import pairwise

    import numpy as np

    columns = changed.shape[1]
    edges = np.linspace(0, columns, slices + 1).astype(int)
    return [
        float(changed[:, start:stop].mean()) if stop > start else 0.0
        for start, stop in pairwise(edges)
    ]
