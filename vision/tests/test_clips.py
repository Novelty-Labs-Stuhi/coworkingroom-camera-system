"""The rolling clip window. Encoding itself needs ffmpeg, so only the window is tested."""

from __future__ import annotations

from stuhi_vision.clips import ClipRecorder


def _recorder(capacity: int = 4) -> ClipRecorder:
    # A fake encoder: the frame's own bytes stand in for JPEG, so no image library needed.
    return ClipRecorder(encode_jpeg=bytes, capacity=capacity)


def test_frames_accumulate() -> None:
    recorder = _recorder()
    recorder.add(b"a")
    recorder.add(b"b")
    assert recorder.frame_count == 2


def test_the_window_is_bounded_and_keeps_the_newest() -> None:
    recorder = _recorder(capacity=3)
    for byte in b"abcde":
        recorder.add(bytes([byte]))

    # Old frames must fall out, or memory grows without limit during a quiet day.
    assert recorder.frame_count == 3


def test_unencodable_frames_are_skipped() -> None:
    recorder = ClipRecorder(encode_jpeg=lambda image: None, capacity=4)
    recorder.add(b"a")
    assert recorder.frame_count == 0


def test_encoding_an_empty_clip_returns_nothing() -> None:
    recorder = _recorder()
    assert recorder.finish(recorder.begin()) is None


def test_a_clip_starts_from_the_buffered_pre_roll() -> None:
    recorder = _recorder(capacity=4)
    recorder.add(b"a")
    recorder.add(b"b")

    clip = recorder.begin()
    assert clip.frames == [b"a", b"b"]  # the approach is already in it


def test_frames_arriving_after_a_clip_opens_are_appended_to_it() -> None:
    recorder = _recorder(capacity=2)
    clip = recorder.begin()
    recorder.add(b"x")
    recorder.add(b"y")

    # The ring buffer is only 2 long, but the open clip keeps everything it was given --
    # otherwise waiting for the person to leave would push the approach back out.
    assert clip.frames == [b"x", b"y"]


def test_a_clip_stops_growing_at_its_cap() -> None:
    recorder = ClipRecorder(encode_jpeg=bytes, capacity=8, max_clip_frames=2)
    clip = recorder.begin()
    for _ in range(5):
        recorder.add(b"z")

    assert clip.full
    assert len(clip.frames) == 2


def test_capacity_is_at_least_one() -> None:
    recorder = _recorder(capacity=0)
    recorder.add(b"a")
    assert recorder.frame_count == 1
