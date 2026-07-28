"""The detection crop, and the coordinate translation that keeps it invisible."""

from __future__ import annotations

import numpy as np

from stuhi_vision.domain import Box, TrackedPerson
from stuhi_vision.region import Region


def test_region_covers_the_line_with_padding() -> None:
    region = Region.around((320, 100), (320, 380), width=640, height=480, padding=0.25)

    assert region.x1 < 320 < region.x2  # the line is inside
    assert region.y1 < 100 and region.y2 > 380


def test_region_is_clamped_to_the_frame() -> None:
    region = Region.around((10, 10), (630, 470), width=640, height=480, padding=0.5)

    assert (region.x1, region.y1) == (0, 0)
    assert (region.x2, region.y2) == (640, 480)


def test_translation_undoes_the_crop_offset() -> None:
    region = Region(x1=100, y1=50, x2=500, y2=400)
    # A box at the crop's own origin really sits at the crop's offset in the frame.
    assert region.to_frame(Box(0, 0, 10, 20)) == Box(100, 50, 110, 70)


def test_cropping_then_translating_recovers_the_original_position() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    region = Region(x1=100, y1=50, x2=500, y2=400)
    image[120:200, 180:240] = 255  # a marker at a known full-frame position

    crop = region.crop(image)
    ys, xs = np.nonzero(crop[:, :, 0])
    found = Box(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))

    assert region.to_frame(found) == Box(180, 120, 240, 200)


def test_people_are_translated_and_keep_their_ids() -> None:
    region = Region(x1=100, y1=50, x2=500, y2=400)
    people = [TrackedPerson(track_id=7, box=Box(0, 0, 10, 10))]

    translated = region.people_to_frame(people)

    assert translated[0].track_id == 7
    assert translated[0].box == Box(100, 50, 110, 60)
