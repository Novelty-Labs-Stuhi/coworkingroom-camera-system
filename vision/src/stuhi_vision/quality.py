"""Image-quality measures used to pick the best crop of a person.

These are identity-agnostic: they say how *clear* a crop is, not who it is. Sharpness
(variance of the Laplacian) is the main signal -- a blurry crop has little
high-frequency detail, so its Laplacian variance is low.
"""

from __future__ import annotations

import cv2
import numpy as np


def laplacian_sharpness(bgr: np.ndarray) -> float:
    """Higher means sharper. Near zero means blurry/flat."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
