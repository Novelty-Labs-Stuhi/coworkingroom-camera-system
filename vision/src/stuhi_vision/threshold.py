"""Detect a passage through the doorway from where a track begins and ends.

The older approach drew a line across the doorway and watched a foot point cross it. That
failed repeatedly in practice: the foot point sat pinned to the bottom edge of the frame
because people pass close to the camera, the line had to be redrawn from a still image every
time a camera moved, and getting ``inside_side`` backwards silently reversed the count.

This uses a sturdier fact about the scene. **The doorframe is visible, and a person passing
through it occludes it** -- their pixels are in front of the frame, not behind it. Someone
merely moving in the background is seen *through* the opening and never overlaps it. So:

* a track that never overlaps the doorframe zone is background traffic and is ignored;
* a track that does overlap it went through the doorway, and **which way it travelled across
  the zone** is which way it went.

Crossing the box is the passage; the direction of travel is the direction. Nothing depends on
the person reaching the edge of the picture, so a track lost mid-doorway still counts, and a
repositioned camera does not invalidate the rule -- only the drawn box, which is redrawable.

Which of those two means "in" is a property of the camera's position, given as
``passing_means``. On the room-facing camera the same shape of rule applies with the edge
being the bottom of the frame -- people approach the lens and leave downwards.

Nothing here is a hairline: the zone is a broad strip and the edge has a margin, so a
knocked camera degrades gradually instead of silently counting nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Literal

from .domain import Box, Crossing, Direction, Frame, TrackedPerson

Edge = Literal["left", "right", "top", "bottom"]

# How a finished track is judged.
#   "edge"     -- did it leave past a frame edge, or arrive from one? Right when the camera
#                 watches people cross its view sideways, as the doorframe camera does.
#   "travel"   -- which way did it move across the zone? Right when the doorframe is visible:
#                 crossing the box *is* the passage, and the direction of travel says which
#                 way. Needs no frame edge, so it survives a camera being repositioned.
#   "approach" -- did it grow or shrink? Right when people walk straight at the lens, where
#                 everyone is already touching the near edge and "at the edge" says nothing.
#   "covering" -- was the doorframe *covered* when the track ended, or when it began? Asks
#                 the pixels, not the box: the doorframe's pixels change only when a body is
#                 in front of it, so this needs no displacement and no threshold at all.
Discriminator = Literal["edge", "travel", "approach", "covering", "preceded"]

# Whether the doorframe is covered *right now*. Read once per frame and shared by every
# track, because coverage is a fact about the doorway rather than about one person -- which
# is also why a track must overlap the zone before the coverage is credited to it.
Covered = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class ThresholdConfig:
    """How one camera recognises a passage.

    ``zone`` is the part of the frame a person must overlap to count -- the visible
    doorframe -- expressed as fractions of the frame so it survives a resolution change.
    ``edge`` is the side of the frame they leave through when passing, and ``margin`` how
    close to it counts as "at" it.
    """

    zone: tuple[float, float, float, float] = (0.0, 0.0, 0.25, 1.0)
    edge: Edge = "left"
    margin: float = 0.12
    passing_means: Direction = Direction.IN
    # Which question to ask of a finished track. Measured on the room-facing camera, every
    # track began *and* ended touching the bottom edge (0.97-1.00 of the frame height),
    # because anyone that close fills the picture downwards -- so the edge test could not
    # tell an arrival from a departure and discarded nearly every real passage. Size change
    # separates them cleanly there: walking at the lens grows, walking away shrinks.
    discriminator: Discriminator = "edge"
    # How much of the frame height a track must gain or lose for "approach" to call it.
    # Measured passes changed by 0.12-0.28; people merely shifting about changed by ~0.09.
    growth_margin: float = 0.12
    # How far a track must travel across the frame, along the axis of ``edge``, for "travel"
    # to call it. Small: somebody passing through a doorway covers most of the frame, while
    # somebody standing in the doorway talking drifts by a few percent.
    travel_margin: float = 0.08
    # A person must be at least this tall in frame to be considered at the door at all.
    # Rejects distant figures that happen to line up with the zone.
    min_height: float = 0.35
    # Frames a track may be missing before it is judged finished. At a few frames per
    # second a person is easily missed for one or two frames mid-stride.
    lost_after: int = 6


@dataclass(frozen=True, slots=True)
class Touch:
    """A track that reached the doorframe box, and what was made of it."""

    frames: int
    tallest: float
    travelled: float   # along the axis of ``edge``; negative is towards left or top
    grew: float        # change in height as a fraction of the frame
    direction: Direction | None   # None: it reached the box but was not judged a passage
    covered_first: bool = False   # the doorframe was covered on the track's first frame
    covered_last: bool = False    # ...and on its last
    # Which came first for the "preceded" rule: being seen off the doorframe, or the doorframe
    # activating. Reported because a verdict from an ordering is unarguable-with unless the
    # ordering is on the record beside it.
    outside_at: int | None = None
    covered_at: int | None = None

    @property
    def order(self) -> str:
        """The two moments the "preceded" rule compares, in words."""
        if self.covered_at is None and self.outside_at is None:
            return "never off the box, never covered"
        if self.covered_at is None:
            return f"seen off the box at {self.outside_at}, never covered"
        if self.outside_at is None:
            return f"covered at {self.covered_at}, never seen off the box"
        first = "seen off the box" if self.outside_at < self.covered_at else "covered"
        if self.outside_at == self.covered_at:
            first = "both in the same frame"
        return f"off the box at {self.outside_at}, covered at {self.covered_at} ({first} first)"

    @property
    def readable(self) -> str:
        verdict = self.direction.value if self.direction else "no passage"
        covering = f"{'covered' if self.covered_first else 'clear'}"
        covering += f"->{'covered' if self.covered_last else 'clear'}"
        return (
            f"touched the box over {self.frames} frames, "
            f"travelled {self.travelled:+.2f}, grew {self.grew:+.2f}, "
            f"doorframe {covering}, {self.order} -> {verdict}"
        )


Report = Callable[[Touch], None]


@dataclass
class _Track:
    first: Box
    last: Box
    last_seen: int
    touched_zone: bool = False
    tallest: float = 0.0
    frames: int = field(default=1)
    # Whether the doorframe was covered on this track's first and most recent frame. The
    # "last" one is overwritten every frame, so when the track is finally judged it holds
    # the state as the person was last *seen* -- not as it is several frames later, by
    # which time whatever they were covering has cleared.
    covered_first: bool = False
    covered_last: bool = False
    # For the "preceded" rule: the frame this person was first seen *outside* the zone, and the
    # frame the zone first activated while they were on screen. Which came first is the whole
    # verdict. Kept as frame indices rather than booleans because "before" is the question, and
    # a pair of flags cannot answer it.
    outside_at: int | None = None
    covered_at: int | None = None


class ThresholdMonitor:
    """Turns tracked people into crossings, deciding when each track ends.

    A decision is deliberately deferred until the track is gone: whether someone *passed
    through* is only knowable once they stop being visible, and that is precisely the
    signal -- they left the frame at the door edge.
    """

    def __init__(
        self,
        config: ThresholdConfig,
        report: Report | None = None,
        covered: Covered | None = None,
    ) -> None:
        self._config = config
        self._tracks: dict[int, _Track] = {}
        self._frame_index = 0
        # Every track that touched the box is reported, counted or not. Without this a
        # refused passage is indistinguishable from one the tracker never saw, and the two
        # need entirely different fixes.
        self._report = report
        # Only the "covering" discriminator asks this. Left unset it reads as never covered,
        # which makes that rule judge nothing rather than judge wrongly.
        self._covered = covered

    def use_zone(self, zone: tuple[float, float, float, float]) -> None:
        """Judge against a different box from now on, without a restart.

        A zone is redrawn because the camera moved, which means the count is wrong *now* --
        waiting for a restart to apply it is most of the way to not being able to redraw it.
        """
        self._config = replace(self._config, zone=zone)

    def update(self, people: Iterable[TrackedPerson], frame: Frame) -> list[Crossing]:
        """Feed one frame's tracked people; return crossings for tracks that just ended."""
        self._frame_index += 1
        height, width = frame.image.shape[:2]

        # Read once, not once per person: it is one fact about the doorway, and asking it
        # again mid-frame could give two tracks different answers about the same moment.
        covered = self._covered() if self._covered is not None else False

        present = set()
        for person in people:
            present.add(person.track_id)
            self._observe(person, width, height, covered)

        return self._resolve_finished(present, frame)

    def _observe(
        self, person: TrackedPerson, width: int, height: int, covered: bool = False
    ) -> None:
        box = person.box
        shape = relative(box, width, height)
        tall_enough = shape.height >= self._config.min_height

        track = self._tracks.get(person.track_id)
        if track is None:
            self._tracks[person.track_id] = track = _Track(
                first=box, last=box, last_seen=self._frame_index, covered_first=covered
            )
        else:
            track.last = box
            track.last_seen = self._frame_index
            track.frames += 1

        track.covered_last = covered
        track.tallest = max(track.tallest, shape.height)
        inside_zone = overlaps(shape, self._config.zone)
        if tall_enough and inside_zone:
            track.touched_zone = True

        # For the "preceded" rule. Both are first-wins: the question is which happened first, so
        # a later sighting outside the zone says nothing that the first one did not already say.
        if not inside_zone and track.outside_at is None:
            track.outside_at = self._frame_index
        if covered and track.covered_at is None:
            track.covered_at = self._frame_index

    def _resolve_finished(self, present: set[int], frame: Frame) -> list[Crossing]:
        height, width = frame.image.shape[:2]
        crossings = []
        for track_id in [
            track_id
            for track_id, track in self._tracks.items()
            if track_id not in present
            and self._frame_index - track.last_seen > self._config.lost_after
        ]:
            track = self._tracks.pop(track_id)
            direction = self._decide(track, width, height)
            if direction is not None:
                crossings.append(
                    Crossing(track_id=track_id, direction=direction, timestamp=frame.timestamp)
                )
        return crossings

    def _worth_judging(self, track: _Track) -> bool:
        """Whether this track is a candidate for a passage at all.

        Two different questions, and the difference is the point of the doorframe:

        * for **preceded**, the doorframe's *pixels* changing is the passage. Somebody going
          through occludes the frame, and that is the event -- so a covering is required and a
          person's own box is never asked to overlap anything. It is consulted only for the
          order of things. Requiring the box to reach the zone would throw away exactly the
          crossings the doorframe exists to catch: the person half behind the door, or beside
          the frame, while the pixels plainly change.
        * every other rule asks whether the *person's box* reached the zone, which is a
          weaker question -- it is what lets somebody crossing the room behind the door look
          like a passage.
        """
        if self._config.discriminator == "preceded":
            return track.covered_at is not None
        return track.touched_zone and track.tallest >= self._config.min_height

    def _decide(self, track: _Track, width: int, height: int) -> Direction | None:
        """A finished track: did it pass through the doorway, and which way?"""
        if not self._worth_judging(track):
            return None  # background traffic, or never close enough to be at the door

        first = relative(track.first, width, height)
        last = relative(track.last, width, height)
        if self._config.discriminator == "approach":
            direction = self._by_size(first, last)
        elif self._config.discriminator == "travel":
            direction = self._by_travel(first, last)
        elif self._config.discriminator == "covering":
            direction = self._by_covering(track)
        elif self._config.discriminator == "preceded":
            direction = self._by_preceded(track)
        else:
            direction = self._by_edge(first, last)
        self._tell(track, first, last, direction)
        return direction

    def _tell(self, track, first: _Relative, last: _Relative, direction) -> None:
        """Say what a track that reached the box did, whether or not it counted."""
        if self._report is None:
            return
        self._report(
            Touch(
                frames=track.frames,
                tallest=track.tallest,
                travelled=_leading(last, self._config.edge)
                - _leading(first, self._config.edge),
                grew=last.height - first.height,
                direction=direction,
                covered_first=track.covered_first,
                covered_last=track.covered_last,
                outside_at=track.outside_at,
                covered_at=track.covered_at,
            )
        )

    def _by_edge(self, first: _Relative, last: _Relative) -> Direction | None:
        """Left past the edge, or arrived from it?"""
        left_at_edge = _at_edge(last, self._config.edge, self._config.margin)
        arrived_at_edge = _at_edge(first, self._config.edge, self._config.margin)

        if left_at_edge and not arrived_at_edge:
            return self._config.passing_means
        if arrived_at_edge and not left_at_edge:
            return _opposite(self._config.passing_means)
        # Both or neither: someone who stepped in and back out again, or who was only ever
        # at the edge. Not a passage, and guessing would put noise into the count.
        return None

    def _by_travel(self, first: _Relative, last: _Relative) -> Direction | None:
        """Which way did they move across the box?

        Crossing the doorframe *is* the passage -- the zone gate has already established
        that -- so all that remains is which way they were going. Movement is measured along
        the axis the doorway runs across, given by ``edge``: towards that side means the
        direction ``passing_means``, away from it the opposite.

        Movement is read from the *leading* side of the box -- the side facing that edge --
        not its centre. Replaying real passages showed why: someone walking close past the
        lens has their box swell in both directions at once, so the centre barely moves while
        they cross the whole picture. One measured passage went from (0.30..0.42) to
        (0.01..0.68): the centre shifted 0.015, which reads as standing still, while the
        leading side swept 0.29 across the frame.

        This is stronger than asking where the track ended, which was the earlier rule. A
        person does not have to reach the edge of the picture, or be visible when they get
        there; a track that is lost mid-doorway still travelled in a direction. It also has
        no dead zone: the edge rule refused to judge anyone who both arrived at the edge and
        left by it, which is what a person filling the near side of the frame looks like.
        """
        towards_lower = self._config.edge in ("left", "top")
        moved = _leading(last, self._config.edge) - _leading(first, self._config.edge)

        if abs(moved) < self._config.travel_margin:
            return None  # stood in the doorway rather than went through it
        towards_edge = moved < 0 if towards_lower else moved > 0
        return self._config.passing_means if towards_edge else _opposite(self._config.passing_means)

    def _by_covering(self, track: _Track) -> Direction | None:
        """Was the doorframe covered as the track ended, or as it began?

        The doorframe's pixels change only when a body is *in front of* it: somebody in the
        corridor beyond is seen through the opening and never covers it. So the coverage
        state at the two moments a track is bounded by says which way the person went:

        * covered as they vanished -- they were still in the doorway when they left the
          picture, so they went through it;
        * covered as they appeared, clear afterwards -- they came through it and walked on
          into view.

        Neither means they were never at the door. **Both** means they went to the doorway
        and came back from it, which is not a passage and is refused rather than guessed.

        Unlike every other discriminator this measures no distance and needs no margin: it
        asks a question the pixels answer outright, so there is nothing here to tune. It is
        also the one signal that gets *stronger* as the person comes closer, where the
        detector that produced the track gets weaker.
        """
        if track.covered_last and not track.covered_first:
            return self._config.passing_means
        if track.covered_first and not track.covered_last:
            return _opposite(self._config.passing_means)
        return None

    def _by_preceded(self, track: _Track) -> Direction | None:
        """Was the person already visible outside the doorframe before it was covered?

        The only question this rule asks. Somebody walking **in** is seen approaching first and
        covers the frame afterwards; somebody walking **out** covers the frame on their way to
        being seen, so the covering comes first. Which of the two happened earlier is the
        direction, and nothing else is consulted.

        It deliberately does not read the order the doorframe's slices lit. That order is what
        the coverage rule depends on, and it is unavailable exactly when it is needed: somebody
        close to the lens covers every slice within one frame, which left about half of all
        episodes with no readable direction on the live camera. Two events with a clear before
        and after are available whenever the person is detected at all.

        Refuses rather than guesses in all three ways it can be short of an answer: the frame
        never covered, the person never seen off it, or both in the same frame -- one frame is
        not an order, and inventing one here would put a crossing on the record that the footage
        never showed.
        """
        if track.covered_at is None or track.outside_at is None:
            return None
        if track.outside_at < track.covered_at:
            return self._config.passing_means
        # Covered *before* the person was seen, or covered in the same frame they appeared in:
        # both mean the frame was already going when they showed up, which is what coming out
        # looks like. The same-frame case is deliberately not a refusal -- somebody stepping out
        # is detected and covers the frame within one frame at this frame rate, and refusing it
        # would throw away the commonest exit there is.
        return _opposite(self._config.passing_means)

    def _by_size(self, first: _Relative, last: _Relative) -> Direction | None:
        """Grew towards the lens, or shrank away from it?

        For a camera people walk straight at, this is the only usable signal: they are
        touching the near edge the whole time, so where they start and end tells nothing.
        """
        change = last.height - first.height
        if change >= self._config.growth_margin:
            return self._config.passing_means  # came at the lens: through the door
        if change <= -self._config.growth_margin:
            return _opposite(self._config.passing_means)
        return None  # barely changed size: milling about rather than passing


@dataclass(frozen=True, slots=True)
class _Relative:
    """A box in fractions of the frame, so thresholds are resolution independent."""

    left: float
    top: float
    right: float
    bottom: float

    @property
    def height(self) -> float:
        return self.bottom - self.top


def relative(box: Box, width: int, height: int) -> _Relative:
    """A detected box as fractions of the frame. Public: the coverage rule needs it too."""
    return _Relative(
        left=box.x1 / width,
        top=box.y1 / height,
        right=box.x2 / width,
        bottom=box.y2 / height,
    )


def overlaps(box: _Relative, zone: tuple[float, float, float, float]) -> bool:
    """Whether a box is over the drawn zone at all. Public: the coverage rule needs it too."""
    x1, y1, x2, y2 = zone
    return box.right > x1 and box.left < x2 and box.bottom > y1 and box.top < y2


def _at_edge(box: _Relative, edge: Edge, margin: float) -> bool:
    if edge == "left":
        return box.left <= margin
    if edge == "right":
        return box.right >= 1.0 - margin
    if edge == "top":
        return box.top <= margin
    return box.bottom >= 1.0 - margin


def _leading(box: _Relative, edge: Edge) -> float:
    """The side of the box facing ``edge`` -- the part of a person that arrives first."""
    return {"left": box.left, "right": box.right, "top": box.top, "bottom": box.bottom}[edge]


def _opposite(direction: Direction) -> Direction:
    return Direction.OUT if direction is Direction.IN else Direction.IN
