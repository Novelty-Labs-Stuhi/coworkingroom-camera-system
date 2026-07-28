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


def test_encoding_an_empty_window_returns_nothing() -> None:
    assert _recorder().encode() is None


def test_capacity_is_at_least_one() -> None:
    recorder = _recorder(capacity=0)
    recorder.add(b"a")
    assert recorder.frame_count == 1
