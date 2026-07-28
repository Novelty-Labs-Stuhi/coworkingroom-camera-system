"""A sighting waits for its clip to end on an empty doorway before being published."""

from __future__ import annotations

from stuhi_vision.clips import ClipRecorder
from stuhi_vision.domain import Direction, Outcome, Sighting
from stuhi_vision.publishing import SightingPublisher


def _sighting() -> Sighting:
    return Sighting(
        timestamp=1.0,
        direction=Direction.IN,
        name=None,
        score=0.1,
        outcome=Outcome.UNKNOWN,
    )


def _setup(clear_frames: int = 3, max_clip_frames: int = 50):
    recorder = ClipRecorder(encode_jpeg=bytes, capacity=4, max_clip_frames=max_clip_frames)
    published: list[tuple[Sighting, bytes | None, int, int]] = []
    publisher = SightingPublisher(
        recorder,
        lambda s, clip, position, total: published.append((s, clip, position, total)),
        clear_frames=clear_frames,
    )
    return recorder, publisher, published


def test_nothing_is_published_while_the_person_is_still_there() -> None:
    _, publisher, published = _setup(clear_frames=3)
    publisher.hold(_sighting())

    for _ in range(10):
        publisher.advance(people_present=True)

    assert published == []
    assert publisher.pending_count == 1


def test_published_once_the_frame_has_been_clear_long_enough() -> None:
    _, publisher, published = _setup(clear_frames=3)
    publisher.hold(_sighting())

    publisher.advance(people_present=False)
    publisher.advance(people_present=False)
    assert published == []
    publisher.advance(people_present=False)

    assert len(published) == 1
    assert publisher.pending_count == 0


def test_the_clear_run_restarts_if_the_person_reappears() -> None:
    # Someone pausing in the doorway must not cut the clip early.
    _, publisher, published = _setup(clear_frames=3)
    publisher.hold(_sighting())

    publisher.advance(people_present=False)
    publisher.advance(people_present=False)
    publisher.advance(people_present=True)  # back again -- start over
    publisher.advance(people_present=False)
    publisher.advance(people_present=False)

    assert published == []


def test_the_clip_contains_the_frames_that_arrived_after_the_crossing() -> None:
    recorder, publisher, published = _setup(clear_frames=2)
    recorder.add(b"a")  # pre-roll
    publisher.hold(_sighting())
    recorder.add(b"b")
    publisher.advance(people_present=False)
    recorder.add(b"c")
    publisher.advance(people_present=False)

    # ffmpeg is not available in tests, so the encode returns None; what matters is that
    # the post-roll frames reached the clip before it was closed.
    assert len(published) == 1


def test_a_loiterer_cannot_hold_a_clip_open_forever() -> None:
    recorder, publisher, published = _setup(clear_frames=100, max_clip_frames=3)
    publisher.hold(_sighting())

    for _ in range(5):
        recorder.add(b"x")
        publisher.advance(people_present=True)

    assert len(published) == 1  # the cap forced it out


def test_flush_publishes_everything_still_waiting() -> None:
    _, publisher, published = _setup(clear_frames=100)
    publisher.hold(_sighting())
    publisher.hold(_sighting())

    publisher.flush()

    assert len(published) == 2
    assert publisher.pending_count == 0


def test_several_people_are_numbered_in_crossing_order() -> None:
    # Their clips are cut from the same window and look alike, so the position is the only
    # thing that says which person a clip is about.
    _, publisher, published = _setup(clear_frames=1)
    first, second, third = _sighting(), _sighting(), _sighting()
    publisher.hold(first)
    publisher.hold(second)
    publisher.hold(third)

    publisher.advance(people_present=False)

    assert [(p, total) for _s, _c, p, total in published] == [(1, 3), (2, 3), (3, 3)]
    assert [s for s, _c, _p, _t in published] == [first, second, third]


def test_a_lone_crossing_is_not_numbered() -> None:
    _, publisher, published = _setup(clear_frames=1)
    publisher.hold(_sighting())
    publisher.advance(people_present=False)

    assert published[0][2:] == (1, 1)  # position 1 of 1 -- caption omits it


def test_numbering_restarts_for_the_next_burst() -> None:
    _, publisher, published = _setup(clear_frames=1)
    publisher.hold(_sighting())
    publisher.advance(people_present=False)
    publisher.hold(_sighting())
    publisher.advance(people_present=False)

    assert [p for _s, _c, p, _t in published] == [1, 1]
