"""A rolling window of recent frames, encodable to a short video on demand.

A single face crop is the best thing to *label*, but it is a poor thing to *judge*: you
cannot see which way someone went, whether two people came through together, or why the
name was wrong. So every frame is kept briefly, and when a crossing commits, the window
around it becomes an MP4.

Frames are held as JPEG rather than decoded arrays. A VGA frame is ~14 KB compressed
against ~900 KB raw, so a few seconds of history costs kilobytes instead of tens of
megabytes -- and JPEG is exactly what ffmpeg's image2pipe wants, so encoding is a pipe
rather than a conversion.

H.264 with yuv420p and +faststart is what Telegram plays inline; other codecs arrive as a
file or a still thumbnail.
"""

from __future__ import annotations

import shutil
import subprocess
from collections import deque
from collections.abc import Callable

JpegEncoder = Callable[[object], bytes | None]


class ClipRecorder:
    """Keeps the last ``capacity`` frames as JPEG and encodes them into an MP4."""

    def __init__(
        self,
        encode_jpeg: JpegEncoder,
        capacity: int = 48,
        fps: int = 8,
    ) -> None:
        self._encode_jpeg = encode_jpeg
        self._frames: deque[bytes] = deque(maxlen=max(1, capacity))
        self._fps = max(1, fps)

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def add(self, image) -> None:
        """Remember one frame. Cheap enough to call on every frame, including idle ones."""
        jpeg = self._encode_jpeg(image)
        if jpeg:
            self._frames.append(jpeg)

    def encode(self) -> bytes | None:
        """Encode the current window to MP4 bytes, or ``None`` if that is not possible."""
        if not self._frames:
            return None
        if not shutil.which("ffmpeg"):
            print("  -> ffmpeg not found; install it: sudo apt-get install -y ffmpeg")
            return None
        return _encode_h264(list(self._frames), self._fps)


def _encode_h264(frames: list[bytes], fps: int) -> bytes | None:
    """Pipe concatenated JPEGs through ffmpeg and read the MP4 back from stdout."""
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart+frag_keyframe+empty_moov",
        "-f",
        "mp4",
        "pipe:1",
    ]
    try:
        finished = subprocess.run(
            command, input=b"".join(frames), capture_output=True, timeout=60, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"  -> clip encode failed to start: {exc}")
        return None
    if finished.returncode != 0 or not finished.stdout:
        detail = finished.stderr.decode(errors="replace")[-300:]
        print(f"  -> clip encode failed: {detail}")
        return None
    return finished.stdout
