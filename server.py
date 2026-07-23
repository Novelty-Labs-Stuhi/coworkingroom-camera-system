"""Receives JPEGs POSTed by the ESP32 camera, saves them, and forwards
them to Telegram.

Run it with:
    pip install fastapi uvicorn httpx
    uvicorn server:app --host 0.0.0.0 --port 3400

The --host 0.0.0.0 is what lets the ESP32 reach it over the LAN.

Telegram credentials live in telegram_config.py (gitignored). Copy
telegram_config.example.py to telegram_config.py and fill it in. If that
file is missing, photos are still saved to disk; they just aren't sent.
"""

from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, Request

# --- Telegram config (optional) --------------------------------------------
try:
    from telegram_config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
except ImportError:
    TELEGRAM_ENABLED = False
    print("telegram_config.py not found — photos will be saved but NOT sent to Telegram")
# ---------------------------------------------------------------------------

app = FastAPI()

PHOTO_DIR = Path(__file__).parent / "photos"
PHOTO_DIR.mkdir(exist_ok=True)


async def send_to_telegram(image_bytes: bytes, filename: str) -> None:
    """POST the photo to the Telegram Bot API's sendPhoto method."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                url,
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": f"Motion: {filename}"},
                files={"photo": (filename, image_bytes, "image/jpeg")},
            )
        if resp.status_code == 200:
            print(f"  -> sent to Telegram")
        else:
            print(f"  -> Telegram error {resp.status_code}: {resp.text}")
    except Exception as exc:  # network hiccup shouldn't crash the upload
        print(f"  -> Telegram send failed: {exc}")


@app.post("/upload")
async def upload(request: Request):
    data = await request.body()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = PHOTO_DIR / f"{timestamp}.jpg"
    path.write_bytes(data)
    print(f"Saved {path.name} ({len(data)} bytes)")

    if TELEGRAM_ENABLED:
        await send_to_telegram(data, path.name)

    return {"status": "ok", "bytes": len(data), "file": path.name}
