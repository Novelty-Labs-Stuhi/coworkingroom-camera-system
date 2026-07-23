"""Doorway photo receiver: save the photo, identify who is there, forward to Telegram.

The ESP32 POSTs a JPEG to ``/upload`` whenever it detects motion. We save it, detect
the person and recognise their face against the enrolled gallery, log the sighting, and
forward the photo to Telegram with a caption naming who was seen. ``GET /sightings``
returns the recent log.

Run it (on the machine the camera uploads to). Port 3400 matches the camera's upload
target in uploader.cpp -- it is in the firewall's allowed range:
    pip install -e vision[server]                 # vision engine + FastAPI/uvicorn/httpx
    uvicorn server:app --host 0.0.0.0 --port 3400

Enrol the people you want named first:
    python -m stuhi_vision enroll "Ilari" faces/ilari_*.jpg   # writes ./gallery/

Telegram is optional: copy telegram_config.example.py -> telegram_config.py. If it is
missing, photos are still saved and identified; they just are not sent.

There is no in/out direction here -- the camera sends single motion-triggered grayscale
frames, not video, so this is a doorway identification log (who was seen, and when).
"""

from __future__ import annotations

import os
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
_PHOTO_DIR = _ROOT / "photos"
_GALLERY_DIR = _ROOT / "gallery"
_DATABASE = _ROOT / "data" / "sightings.db"
_FACE_MATCH = float(os.environ.get("STUHI_FACE_MATCH", "0.35"))

_PHOTO_DIR.mkdir(exist_ok=True)

# Telegram config (optional): copy telegram_config.example.py -> telegram_config.py
try:
    from telegram_config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

    TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
except ImportError:
    TELEGRAM_ENABLED = False
    print(
        "telegram_config.py not found -- photos will be saved/identified but NOT sent"
    )

app = FastAPI(title="stuhi doorway camera")

_identifier = PhotoIdentifier(
    PersonDetector(),
    FaceRecognizer(FaceGallery.load(_GALLERY_DIR), _FACE_MATCH),
)
_store = SightingStore(_DATABASE)


def _describe(seen: list[str]) -> str:
    return "seen: " + ", ".join(seen) if seen else "motion (no person recognised)"


async def send_to_telegram(image_bytes: bytes, caption: str) -> None:
    """POST the photo to the Telegram Bot API's sendPhoto method."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                url,
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},
                files={"photo": ("motion.jpg", image_bytes, "image/jpeg")},
            )
        if resp.status_code != 200:
            print(f"  -> Telegram error {resp.status_code}: {resp.text}")
    except Exception as exc:  # a network hiccup must not crash the upload
        print(f"  -> Telegram send failed: {exc}")


@app.post("/upload")
async def upload(request: Request) -> dict:
    data = await request.body()
    now = datetime.now()
    path = _PHOTO_DIR / f"{now:%Y-%m-%d_%H-%M-%S}.jpg"
    path.write_bytes(data)

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    seen: list[str] = []
    if image is not None:
        for sighting in _identifier.identify(image):
            _store.record(now.timestamp(), sighting.name, sighting.clarity)
            seen.append(sighting.name)
    print(f"Saved {path.name} ({len(data)} bytes) -> {_describe(seen)}")

    if TELEGRAM_ENABLED:
        await send_to_telegram(data, f"{path.name} - {_describe(seen)}")

    return {"status": "ok", "bytes": len(data), "file": path.name, "seen": seen}


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
