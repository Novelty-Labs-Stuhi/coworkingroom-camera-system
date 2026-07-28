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

from ..names import parse_names
from ..review import ReviewQueue

_HERE = Path(__file__).parent


class LabelRequest(BaseModel):
    sighting_id: str
    name: str


class DismissRequest(BaseModel):
    sighting_id: str


def _as_dict(record, groups: dict[int, int] | None = None) -> dict:
    size = (groups or {}).get(record.burst, 1)
    return {
        "id": record.sighting_id,
        "direction": record.direction,
        "outcome": record.outcome,
        "score": record.score,
        "name": record.display_name,
        "labelled_as": record.labelled_as,
        # A group label is assigned by crossing order, so it is the case most likely to
        # have the right names on the wrong people. Surfacing the position lets a human
        # check rather than trust it.
        "position": record.position,
        "group_size": size,
    }


def _suspect_dicts(review: ReviewQueue, report) -> list[dict]:
    """Pair each suspect reference with its sighting, so the UI can show the clip."""
    groups = review.group_sizes()
    entries = []
    for suspect in report.suspects:
        record = review.get(suspect.sighting_id)
        base = _as_dict(record, groups) if record is not None else {"id": suspect.sighting_id}
        entries.append(
            {
                **base,
                "similarity": suspect.similarity,
                "average": suspect.average,
                "shortfall": suspect.shortfall,
                "reason": suspect.reason,
            }
        )
    return entries


def _apply_label(review: ReviewQueue, sighting_id: str, text: str) -> JSONResponse:
    """One name labels one sighting; several label a whole group in crossing order."""
    names = parse_names(text)
    if not names:
        raise HTTPException(status_code=400, detail="a name is required")

    if len(names) == 1:
        outcome = review.label(sighting_id, names[0])
        if not outcome.succeeded:
            raise HTTPException(status_code=404, detail=outcome.value)
        return JSONResponse({"outcome": outcome.value, "people": review.counts()})

    if len(review.burst_members(sighting_id)) < 2:
        # Refusing beats inventing: passing the text through as one name is what created
        # gallery entries like "a, yehor".
        raise HTTPException(
            status_code=400,
            detail=(
                f"{len(names)} names given but only one person crossed here. "
                "Label them one clip at a time."
            ),
        )

    results = review.label_burst(sighting_id, names)
    return JSONResponse(
        {
            "outcome": "enrolled",
            "people": review.counts(),
            "group": [{"id": id_, "outcome": result.value} for id_, result in results],
        }
    )


def _add_api_routes(app: FastAPI, review: ReviewQueue) -> None:
    @app.get("/api/sightings")
    def sightings() -> JSONResponse:
        groups = review.group_sizes()
        return JSONResponse(
            {
                "pending": [_as_dict(r, groups) for r in review.pending(limit=50)],
                "labelled": [_as_dict(r, groups) for r in review.labelled(limit=50)],
                "people": review.counts(),
            }
        )

    @app.get("/api/audit")
    def audit() -> JSONResponse:
        """Enrolled faces worth a second look, and people with too few examples."""
        report = review.audit()
        return JSONResponse({"suspects": _suspect_dicts(review, report), "thin": report.thin})

    @app.post("/api/label")
    def label(body: LabelRequest) -> JSONResponse:
        """Label one sighting, or a whole group when several names are given.

        The names are read with the same parser the chat uses. Passing the raw text through
        as a single name is what produced gallery entries like ``a, yehor`` -- a person who
        does not exist, quietly competing with the real ones.
        """
        return _apply_label(review, body.sighting_id, body.name)

    @app.post("/api/dismiss")
    def dismiss(body: DismissRequest) -> JSONResponse:
        """Reject a sighting: stop offering it, and remove any reference it contributed."""
        outcome = review.dismiss(body.sighting_id)
        if not outcome.succeeded:
            raise HTTPException(status_code=404, detail=outcome.value)
        return JSONResponse({"outcome": outcome.value, "people": review.counts()})

    @app.post("/api/unlabel")
    def unlabel(body: DismissRequest) -> JSONResponse:
        """Take a label back and return the clip to the queue -- the clip is still usable."""
        outcome = review.unlabel(body.sighting_id)
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
