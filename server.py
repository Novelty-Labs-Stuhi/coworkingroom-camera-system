"""Doorway clip receiver: stitch a motion clip and forward it to Telegram.

The ESP32 records a ~1.5 s burst of frames whenever it detects motion and POSTs them
to ``/clip`` as one length-prefixed binary stream (per frame: a little-endian uint32
length, then that many JPEG bytes). We encode the frames into an H.264 MP4 with ffmpeg
and forward the video to Telegram.

Run it (on the machine the camera uploads to). Port 3400 matches the camera's upload
target in uploader.cpp -- it is in the firewall's allowed range:
    pip install fastapi uvicorn httpx
    sudo apt-get install -y ffmpeg          # the encoder
    uvicorn server:app --host 0.0.0.0 --port 3400

Telegram is optional: copy telegram_config.example.py -> telegram_config.py. If it is
missing, clips are still saved to ./clips/; they just are not sent.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, Request

_ROOT = Path(__file__).parent
_CLIP_DIR = _ROOT / "clips"
_DEFAULT_FPS = 10  # fallback when the camera doesn't send an X-Fps header

_CLIP_DIR.mkdir(exist_ok=True)

# Telegram config (optional): copy telegram_config.example.py -> telegram_config.py
try:
    from telegram_config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

    TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
except ImportError:
    TELEGRAM_ENABLED = False
    print("telegram_config.py not found -- clips will be saved but NOT sent")

app = FastAPI(title="stuhi doorway camera")


def _split_frames(body: bytes) -> list[bytes]:
    """Split a length-prefixed frame stream into individual JPEG payloads."""
    frames: list[bytes] = []
    off, n = 0, len(body)
    while off + 4 <= n:
        (length,) = struct.unpack_from("<I", body, off)
        off += 4
        if length == 0 or off + length > n:
            break  # truncated / malformed tail -- keep what we have
        frames.append(body[off : off + length])
        off += length
    return frames


def _encode_mp4(frames: list[bytes], fps: int, path: Path) -> bool:
    """Encode concatenated JPEG frames into an H.264 MP4 via ffmpeg.

    H.264 + yuv420p + faststart is what Telegram actually plays inline as a video;
    the older mp4v/MPEG-4 output only shows up as a static thumbnail.
    """
    if not shutil.which("ffmpeg"):
        print("  -> ffmpeg not found; install it: sudo apt-get install -y ffmpeg")
        return False
    cmd = [
        "ffmpeg", "-y",
        "-f", "image2pipe", "-vcodec", "mjpeg", "-framerate", str(fps), "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(path),
    ]
    proc = subprocess.run(cmd, input=b"".join(frames), capture_output=True)
    if proc.returncode != 0:
        print(f"  -> ffmpeg failed: {proc.stderr.decode(errors='replace')[-500:]}")
        return False
    return True


async def send_video_to_telegram(video_bytes: bytes, caption: str) -> None:
    """POST the clip to the Telegram Bot API's sendVideo method."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo"
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                url,
                data={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "caption": caption,
                    "supports_streaming": "true",
                },
                files={"video": ("motion.mp4", video_bytes, "video/mp4")},
            )
        if resp.status_code != 200:
            print(f"  -> Telegram error {resp.status_code}: {resp.text}")
    except Exception as exc:  # a network hiccup must not crash the upload
        print(f"  -> Telegram send failed: {exc}")


@app.post("/clip")
async def clip(request: Request) -> dict:
    body = await request.body()
    now = datetime.now()

    frames = _split_frames(body)
    if not frames:
        print(f"Received clip with no frames ({len(body)} bytes)")
        return {"status": "error", "reason": "no frames"}

    try:
        fps = int(request.headers.get("x-fps", _DEFAULT_FPS))
    except ValueError:
        fps = _DEFAULT_FPS
    fps = max(1, min(fps, 30))

    path = _CLIP_DIR / f"{now:%Y-%m-%d_%H-%M-%S}.mp4"
    if not _encode_mp4(frames, fps, path):
        return {"status": "error", "reason": "video encode failed"}

    print(f"Saved {path.name} ({len(frames)} frames @ {fps} fps)")

    if TELEGRAM_ENABLED:
        await send_video_to_telegram(path.read_bytes(), f"motion - {path.name}")

    return {"status": "ok", "frames": len(frames), "fps": fps, "file": path.name}
