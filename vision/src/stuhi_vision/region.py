"""Restrict detection to the part of the frame that contains the doorway.

Detection cost scales with pixel count, and most of a frame is wall. Cropping to the
region around the threshold line buys speed *and* accuracy: fewer distant people wandering
past to spawn tracks that never cross anything.

The subtlety is coordinates. Everything downstream -- the doorway line, the annotated
review video, the stored crops -- works in full-frame pixels. So detections found inside a
crop must be translated back before anyone else sees them. Getting that wrong shifts every
box by the crop offset, which looks exactly like a mis-drawn doorway line. Hence this lives
in one small module with tests, rather than inline where it would be easy to forget.
"""

from __future__ import annotations

from dataclasses import dataclass

from .domain import Box, TrackedPerson


@dataclass(frozen=True, slots=True)
class Region:
    """A sub-rectangle of the frame, in full-frame pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int

    @classmethod
    def of(
        cls,
        zone: tuple[float, float, float, float],
        width: int,
        height: int,
    ) -> Region:
        """The part of the frame a drawn zone covers, in pixels.

        Used where a zone means "only look here": on the room camera, the doorway is one
        corner of a wide view full of people at desks, and everything outside it is a
        distraction the detector pays for on every frame and can mistake for somebody at the
        door. Boxes are translated back to full-frame coordinates, so nothing downstream can
        tell the difference.
        """
        x1, y1, x2, y2 = zone
        return cls(
            x1=max(0, int(x1 * width)),
            y1=max(0, int(y1 * height)),
            x2=min(width, int(x2 * width)),
            y2=min(height, int(y2 * height)),
        )

    @classmethod
    def around(
        cls,
        line_a: tuple[float, float],
        line_b: tuple[float, float],
        width: int,
        height: int,
        padding: float = 0.35,
    ) -> Region:
        """A region covering the doorway line plus ``padding`` of the frame around it.

        People are taller than the line and approach it from either side, so the box needs
        generous margins -- a crop that clips someone's head defeats the purpose.
        """
        pad_x = padding * width
        pad_y = padding * height
        x1 = min(line_a[0], line_b[0]) - pad_x
        x2 = max(line_a[0], line_b[0]) + pad_x
        y1 = min(line_a[1], line_b[1]) - pad_y
        y2 = max(line_a[1], line_b[1]) + pad_y
        return cls(
            x1=max(0, int(x1)),
            y1=max(0, int(y1)),
            x2=min(width, round(x2)),
            y2=min(height, round(y2)),
        )

    def crop(self, image):
        return image[self.y1 : self.y2, self.x1 : self.x2]

    def to_frame(self, box: Box) -> Box:
        """Translate a box found inside the crop back into full-frame coordinates."""
        return Box(
            x1=box.x1 + self.x1,
            y1=box.y1 + self.y1,
            x2=box.x2 + self.x1,
            y2=box.y2 + self.y1,
        )

    def people_to_frame(self, people: list[TrackedPerson]) -> list[TrackedPerson]:
        return [
            TrackedPerson(track_id=person.track_id, box=self.to_frame(person.box))
            for person in people
        ]
