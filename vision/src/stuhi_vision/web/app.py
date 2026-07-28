"""A small web UI for labelling sightings, alongside the Telegram route.

Both routes drive the same :class:`~..review.ReviewQueue`, so a sighting labelled in the
chat shows as labelled here and vice versa, and labelling the same sighting through both
still adds exactly one reference vector.

It runs *inside* the pipeline process. That is deliberate: the gallery is held in memory by
the recogniser, so a separate process could only write files and the running pipeline would
not see the new face until it restarted. Sharing the object means a label takes effect on
the very next frame.

Markup and styling live in ``templates/`` and ``static/`` rather than in Python strings.
"""

from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.requests import Request

from ..review import ReviewQueue

_HERE = Path(__file__).parent


class LabelRequest(BaseModel):
    sighting_id: str
    name: str


def _as_dict(record) -> dict:
    return {
        "id": record.sighting_id,
        "direction": record.direction,
        "outcome": record.outcome,
        "score": record.score,
        "name": record.display_name,
        "labelled_as": record.labelled_as,
    }


def _add_api_routes(app: FastAPI, review: ReviewQueue) -> None:
    @app.get("/api/sightings")
    def sightings() -> JSONResponse:
        return JSONResponse(
            {
                "pending": [_as_dict(r) for r in review.pending(limit=50)],
                "labelled": [_as_dict(r) for r in review.labelled(limit=50)],
                "people": review.counts(),
            }
        )

    @app.post("/api/label")
    def label(body: LabelRequest) -> JSONResponse:
        name = body.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="a name is required")
        outcome = review.label(body.sighting_id, name)
        if not outcome.succeeded:
            raise HTTPException(status_code=404, detail=outcome.value)
        return JSONResponse({"outcome": outcome.value, "people": review.counts()})


def _add_media_routes(app: FastAPI, review: ReviewQueue) -> None:
    @app.get("/media/{sighting_id}.mp4")
    def clip(sighting_id: str) -> FileResponse:
        path = review.clip_path(sighting_id)
        if path is None:
            raise HTTPException(status_code=404, detail="no clip for that sighting")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/media/{sighting_id}.jpg")
    def crop(sighting_id: str) -> FileResponse:
        path = review.crop_path(sighting_id)
        if path is None:
            raise HTTPException(status_code=404, detail="no face crop for that sighting")
        return FileResponse(path, media_type="image/jpeg")


def create_app(review: ReviewQueue) -> FastAPI:
    """Build the labelling UI over an existing review queue."""
    app = FastAPI(title="stuhi labelling")
    templates = Jinja2Templates(directory=str(_HERE / "templates"))
    app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")

    @app.get("/")
    def index(request: Request):
        return templates.TemplateResponse(request, "index.html")

    _add_api_routes(app, review)
    _add_media_routes(app, review)
    return app


class WebUI:
    """Runs the labelling UI in a background thread for the lifetime of the pipeline."""

    def __init__(self, review: ReviewQueue, host: str = "0.0.0.0", port: int = 8800) -> None:
        self._review = review
        self._host = host
        self._port = port
        self._server = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        import uvicorn

        config = uvicorn.Config(
            create_app(self._review),
            host=self._host,
            port=self._port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, name="web-ui", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
