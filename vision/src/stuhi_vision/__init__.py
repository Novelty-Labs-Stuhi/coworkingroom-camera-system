"""stuhi_vision -- single-camera, face-based occupancy tracking for the stuhi office.

Import-light on purpose: heavy dependencies (OpenCV, torch, InsightFace, Ultralytics)
are pulled in lazily by the modules that need them, so the pure logic imports fast.
"""

__version__ = "0.1.0"
