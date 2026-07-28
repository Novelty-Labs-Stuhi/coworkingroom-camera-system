"""Frames must reach the models upright, whatever way the camera is bolted to the wall."""

from __future__ import annotations

import numpy as np
import pytest

from stuhi_vision.domain import Frame
from stuhi_vision.sources.oriented import OrientedSource, normalise_degrees


def _marked_frame() -> Frame:
    # A single bright pixel near the top-left corner, so rotation is unambiguous.
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    image[1, 2] = 255
    return Frame(timestamp=7.0, image=image)


def _corner_of(frame: Frame) -> tuple[int, int]:
    ys, xs = np.nonzero(frame.image[:, :, 0])
    return int(ys[0]), int(xs[0])


def test_zero_degrees_passes_frames_through_untouched() -> None:
    original = _marked_frame()
    (result,) = list(OrientedSource([original], 0))
    assert result is original


def test_180_moves_the_mark_to_the_opposite_corner() -> None:
    (result,) = list(OrientedSource([_marked_frame()], 180))
    # An 8x12 image: (1, 2) inverted becomes (6, 9).
    assert _corner_of(result) == (6, 9)
    assert result.image.shape == (8, 12, 3)


def test_90_swaps_the_axes() -> None:
    (result,) = list(OrientedSource([_marked_frame()], 90))
    assert result.image.shape == (12, 8, 3)


def test_timestamps_survive_rotation() -> None:
    (result,) = list(OrientedSource([_marked_frame()], 180))
    assert result.timestamp == 7.0


def test_negative_and_wrapping_angles_are_accepted() -> None:
    assert normalise_degrees(-90) == 270
    assert normalise_degrees(360) == 0
    assert normalise_degrees(540) == 180


def test_an_arbitrary_angle_is_refused() -> None:
    # Silently rounding would leave frames subtly wrong, which is worse than failing.
    with pytest.raises(ValueError, match="0, 90, 180 or 270"):
        OrientedSource([], 45)
