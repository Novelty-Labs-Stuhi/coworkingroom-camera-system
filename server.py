"""Receives JPEGs POSTed by the ESP32 camera and saves them to ./photos/.

Run it with:
    pip install fastapi uvicorn
    uvicorn server:app --host 0.0.0.0 --port 8000

The --host 0.0.0.0 is what lets the ESP32 reach it over the LAN.
"""

from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request

app = FastAPI()

PHOTO_DIR = Path(__file__).parent / "photos"
PHOTO_DIR.mkdir(exist_ok=True)


@app.post("/upload")
async def upload(request: Request):
    data = await request.body()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = PHOTO_DIR / f"{timestamp}.jpg"
    path.write_bytes(data)
    print(f"Saved {path.name} ({len(data)} bytes)")
    return {"status": "ok", "bytes": len(data), "file": path.name}
