"""Lighting normalization applied to a crop before embedding.

The same person under a bright doorway and a dim room should embed similarly, so we
even out exposure and local contrast first. CLAHE (Contrast-Limited Adaptive
Histogram Equalization) on the luminance channel does this: it equalizes contrast in
small tiles (local, so a dark corner and a bright patch both normalize) while clipping
the histogram to avoid amplifying noise. The same transform is applied on entry and
exit so both sit on the same footing when compared.
"""

from __future__ import annotations

import cv2
import numpy as np

_CLIP_LIMIT = 2.0
_TILE_GRID = (8, 8)


def normalize_lighting(bgr: np.ndarray) -> np.ndarray:
    """Return a lighting-normalized copy of a BGR image (CLAHE on the L channel)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=_CLIP_LIMIT, tileGridSize=_TILE_GRID)
    lightness = clahe.apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2BGR)
