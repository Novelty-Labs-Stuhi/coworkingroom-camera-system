"""Doorway clip receiver: stitch a motion clip, identify who is there, forward to Telegram.

The ESP32 records a ~1.5 s burst of frames whenever it detects motion and POSTs them
to ``/clip`` as one length-prefixed binary stream (per frame: a little-endian uint32
length, then that many JPEG bytes). We stitch the frames into an MP4, detect the person
and recognise their face against the enrolled gallery on the sharpest frame, log the
sighting, and forward the video to Telegram with a caption naming who was seen.
``GET /sightings`` returns the recent log.

Run it (on the machine the camera uploads to). Port 3400 matches the camera's upload
target in uploader.cpp -- it is in the firewall's allowed range:
    pip install -e vision[server]                 # vision engine + FastAPI/uvicorn/httpx
    uvicorn server:app --host 0.0.0.0 --port 3400

Enrol the people you want named first:
    python -m stuhi_vision enroll "Ilari" faces/ilari_*.jpg   # writes ./gallery/

Telegram is optional: copy telegram_config.example.py -> telegram_config.py. If it is
missing, clips are still saved and identified; they just are not sent.

The camera sends single-direction, motion-triggered grayscale clips (not a continuous
stream), so this is a doorway identification log: who was seen, and when.
"""

from __future__ import annotations

import os
import struct
import sys
from datetime import datetime
from pathlib import Path

# Make the vision engine importable whether or not it was pip-installed.
sys.path.insert(0, str(Path(__file__).parent / "vision" / "src"))

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, Request

from stuhi_vision.detection import PersonDetector
from stuhi_vision.photo_ingest import PhotoIdentifier
from stuhi_vision.recognition.face import FaceRecognizer
from stuhi_vision.recognition.gallery import FaceGallery
from stuhi_vision.store import SightingStore

_ROOT = Path(__file__).parent
_CLIP_DIR = _ROOT / "clips"
_GALLERY_DIR = _ROOT / "gallery"
_DATABASE = _ROOT / "data" / "sightings.db"
_FACE_MATCH = float(os.environ.get("STUHI_FACE_MATCH", "0.35"))
_DEFAULT_FPS = 10  # fallback when the camera doesn't send an X-Fps header

_CLIP_DIR.mkdir(exist_ok=True)

# Telegram config (optional): copy telegram_config.example.py -> telegram_config.py
try:
    from telegram_config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

    TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
except ImportError:
    TELEGRAM_ENABLED = False
    print("telegram_config.py not found -- clips will be saved/identified but NOT sent")

app = FastAPI(title="stuhi doorway camera")

_identifier = PhotoIdentifier(
    PersonDetector(),
    FaceRecognizer(FaceGallery.load(_GALLERY_DIR), _FACE_MATCH),
)
_store = SightingStore(_DATABASE)


def _describe(seen: list[str]) -> str:
    return "seen: " + ", ".join(seen) if seen else "motion (no person recognised)"


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


def _decode(frames: list[bytes]) -> list[np.ndarray]:
    """Decode JPEG frames to BGR images, dropping any that fail to decode."""
    images = []
    for jpg in frames:
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
    return images


def _sharpest(images: list[np.ndarray]) -> np.ndarray:
    """Pick the least-blurry frame (highest Laplacian variance) for recognition."""
    return max(images, key=lambda im: cv2.Laplacian(im, cv2.CV_64F).var())


def _write_mp4(images: list[np.ndarray], path: Path, fps: int) -> bool:
    """Stitch frames into an MP4. Returns False if the writer couldn't open."""
    height, width = images[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        return False
    for img in images:
        # VideoWriter needs every frame the same size as the first.
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        writer.write(img)
    writer.release()
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
    images = _decode(frames)
    if not images:
        print(f"Received clip with no decodable frames ({len(body)} bytes)")
        return {"status": "error", "reason": "no decodable frames"}

    try:
        fps = int(request.headers.get("x-fps", _DEFAULT_FPS))
    except ValueError:
        fps = _DEFAULT_FPS
    fps = max(1, min(fps, 30))

    path = _CLIP_DIR / f"{now:%Y-%m-%d_%H-%M-%S}.mp4"
    if not _write_mp4(images, path, fps):
        print("VideoWriter failed to open (no mp4v codec?)")
        return {"status": "error", "reason": "video encode failed"}

    # Identify on the sharpest frame -- one recognition pass, best chance of a face.
    seen: list[str] = []
    for sighting in _identifier.identify(_sharpest(images)):
        _store.record(now.timestamp(), sighting.name, sighting.clarity)
        seen.append(sighting.name)

    print(f"Saved {path.name} ({len(images)} frames @ {fps} fps) -> {_describe(seen)}")

    if TELEGRAM_ENABLED:
        await send_video_to_telegram(path.read_bytes(), f"{path.name} - {_describe(seen)}")

    return {
        "status": "ok",
        "frames": len(images),
        "fps": fps,
        "file": path.name,
        "seen": seen,
    }


@app.get("/sightings")
def sightings(limit: int = 50) -> dict:
    recent = [
        {
            "time": datetime.fromtimestamp(ts).isoformat(timespec="seconds"),
            "name": name,
            "clarity": round(clarity, 3),
        }
        for ts, name, clarity in _store.recent(limit)
    ]
    return {"count": len(recent), "recent": recent}
