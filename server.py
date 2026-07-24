"""Doorway clip receiver with person re-ID and a Telegram label loop.

Flow (see the design sketch):

    video -> server -> [yolo + torchreid] embedding
                          |                     ^
                indexed by video          compare vs labeled sets
                          v               (min cosine similarity)
                   embeddings store  ------------------------------
                          |
                    named or unknown -> Telegram --(you reply /label)--> store updated

The ESP32 POSTs a ~1.5 s burst of JPEG frames to ``/clip`` (per frame: a little-endian
uint32 length + JPEG bytes). We stitch them into an H.264 MP4, embed the person with
OSNet, match against the labeled store, forward the video to Telegram naming who it is
(or "unknown"), and store the embedding under that name keyed by the clip's video id.

Reply in Telegram to teach it:
    /label <video_id> <name>     tag a clip's person as <name>
    /label <name>                (as a reply to a clip message) same, id read from it
    /cosine_score                show the current match threshold
    /cosine_score <value>        set it (the one tuning knob)
    /people                      how many clips are stored per name

Run it (on the machine the camera uploads to):
    pip install fastapi uvicorn httpx torchreid   # + torch/ultralytics/opencv
    sudo apt-get install -y ffmpeg
    uvicorn server:app --host 0.0.0.0 --port 3400
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import struct
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, Request

# Make the vision engine importable whether or not it was pip-installed.
sys.path.insert(0, str(Path(__file__).parent / "vision" / "src"))

from stuhi_vision.recognition.reid import ClipEmbedder
from stuhi_vision.reid_store import UNKNOWN, ReidStore

_ROOT = Path(__file__).parent
_CLIP_DIR = _ROOT / "clips"
_STORE_PATH = _ROOT / "data" / "reid_store.json"
_DEFAULT_FPS = 10  # fallback when the camera doesn't send an X-Fps header

_CLIP_DIR.mkdir(exist_ok=True)

# Telegram config (optional): copy telegram_config.example.py -> telegram_config.py
try:
    from telegram_config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

    TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
except ImportError:
    TELEGRAM_ENABLED = False
    print("telegram_config.py not found -- clips saved & embedded but NOT sent")

_embedder = ClipEmbedder()
_store = ReidStore(_STORE_PATH)


# --- clip stream + video encoding -------------------------------------------
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

    H.264 + yuv420p + faststart is what Telegram plays inline as a video; the older
    mp4v output only shows up as a static thumbnail.
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


# --- Telegram: send clips out, poll commands in -----------------------------
def _api(method: str) -> str:
    return f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"


async def send_video(video_bytes: bytes, caption: str) -> None:
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                _api("sendVideo"),
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


async def send_message(text: str) -> None:
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            await client.post(_api("sendMessage"), data={"chat_id": TELEGRAM_CHAT_ID, "text": text})
    except Exception as exc:
        print(f"  -> Telegram reply failed: {exc}")


def _video_id_from_reply(message: dict) -> str | None:
    """Pull 'video: <id>' out of the caption of the message being replied to."""
    replied = message.get("reply_to_message", {})
    caption = replied.get("caption", "")
    for line in caption.splitlines():
        if line.startswith("video:"):
            return line.split(":", 1)[1].strip()
    return None


async def _handle_command(message: dict) -> None:
    text = (message.get("text") or "").strip()
    if not text.startswith("/"):
        return
    parts = text.split()
    cmd, args = parts[0].lstrip("/").lower(), parts[1:]

    if cmd == "cosine_score":
        if not args:
            await send_message(f"cosine_score threshold = {_store.threshold:.2f}")
            return
        try:
            value = float(args[0])
        except ValueError:
            await send_message("usage: /cosine_score 0.80")
            return
        _store.set_threshold(value)
        await send_message(f"cosine_score threshold set to {value:.2f}")

    elif cmd == "label":
        # /label <video_id> <name>   OR   (reply to a clip) /label <name>
        if len(args) >= 2:
            video_id, name = args[0], " ".join(args[1:])
        elif len(args) == 1 and (video_id := _video_id_from_reply(message)):
            name = args[0]
        else:
            await send_message("usage: /label <video_id> <name>  (or reply to a clip with /label <name>)")
            return
        ok = _store.relabel(video_id, name)
        await send_message(f"labeled {video_id} -> {name}" if ok else f"unknown video id: {video_id}")

    elif cmd == "people":
        summary = _store.summary()
        if not summary:
            await send_message("store is empty")
        else:
            await send_message("\n".join(f"{name}: {count}" for name, count in summary.items()))


async def _poll_telegram() -> None:
    """Long-poll getUpdates for label/threshold commands until the app shuts down."""
    offset = 0
    async with httpx.AsyncClient(timeout=40) as client:
        while True:
            try:
                resp = await client.get(
                    _api("getUpdates"), params={"offset": offset, "timeout": 30}
                )
                for update in resp.json().get("result", []):
                    offset = update["update_id"] + 1
                    message = update.get("message")
                    if message:
                        await _handle_command(message)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"  -> Telegram poll error: {exc}")
                await asyncio.sleep(5)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    task = None
    if TELEGRAM_ENABLED:
        task = asyncio.create_task(_poll_telegram())
    yield
    if task:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


app = FastAPI(title="stuhi doorway camera", lifespan=lifespan)


@app.post("/clip")
async def clip(request: Request) -> dict:
    body = await request.body()
    now = datetime.now()
    video_id = f"{now:%Y-%m-%d_%H-%M-%S}"

    frames = _split_frames(body)
    if not frames:
        print(f"Received clip with no frames ({len(body)} bytes)")
        return {"status": "error", "reason": "no frames"}

    try:
        fps = int(request.headers.get("x-fps", _DEFAULT_FPS))
    except ValueError:
        fps = _DEFAULT_FPS
    fps = max(1, min(fps, 30))

    path = _CLIP_DIR / f"{video_id}.mp4"
    if not await asyncio.to_thread(_encode_mp4, frames, fps, path):
        return {"status": "error", "reason": "video encode failed"}

    # Embed the person (torch is CPU-heavy -> off the event loop) and match the store.
    emb = await asyncio.to_thread(_embedder.embed_jpegs, frames)
    if emb is None:
        name, score, caption = None, 0.0, "motion (no person detected)"
    else:
        name, score = _store.match(emb)
        _store.add(video_id, emb, name or UNKNOWN)
        caption = f"{name} ({score:.2f})" if name else f"unknown (best {score:.2f})"
    caption += f"\nvideo: {video_id}"
    if name is None and emb is not None:
        caption += "\nreply: /label <name>"

    print(f"Saved {video_id}.mp4 ({len(frames)} frames @ {fps} fps) -> {caption.splitlines()[0]}")

    if TELEGRAM_ENABLED:
        await send_video(path.read_bytes(), caption)

    return {"status": "ok", "frames": len(frames), "fps": fps, "video_id": video_id, "name": name}


@app.get("/people")
def people() -> dict:
    return {"threshold": _store.threshold, "people": _store.summary()}
