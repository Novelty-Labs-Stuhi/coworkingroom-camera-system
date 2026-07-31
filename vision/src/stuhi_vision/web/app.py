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
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.requests import Request

from ..names import parse_names
from ..review import ReviewQueue
from ..zones import DrawnZone, ZoneStore

_HERE = Path(__file__).parent


class LabelRequest(BaseModel):
    sighting_id: str
    name: str


class DismissRequest(BaseModel):
    sighting_id: str


class ZoneRequest(BaseModel):
    """A rectangle dragged on a camera's live frame, in fractions of it."""

    camera: str
    x1: float
    y1: float
    x2: float
    y2: float


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


def _camera_states(frames, zones: ZoneStore, drift: dict) -> list[dict]:
    """Each camera's drawn zone and whether its view has shifted since that was drawn."""
    listed = []
    for name in frames.cameras:
        age = frames.age(name)
        watch = drift.get(name)
        reading = watch.latest if watch else None
        zone = zones.get(name)
        listed.append(
            {
                "name": name,
                "zone": zone.as_tuple() if zone else None,
                # How old the picture on the page is. Stated rather than implied, because the
                # reader banks hundreds of frames and "the latest frame" has meant two very
                # different things depending on which end of the pipeline it came from.
                "frame_age": round(age, 1) if age is not None else None,
                "has_reference": bool(watch and watch.has_reference),
                "moved": bool(watch and watch.has_moved),
                "shift_px": round(reading.magnitude, 1) if reading else None,
                "shift": reading.readable if reading else None,
                # Sent so the page can call a view steady only when it is. "moved" goes
                # quiet once somebody has been told, which is not the same as recovered.
                "tolerance_px": watch.tolerance_px if watch else None,
            }
        )
    return listed


def _add_camera_routes(app: FastAPI, frames, zones: ZoneStore, drift: dict) -> None:
    """Live frames, the drawn zones, and whether a camera has been knocked.

    Frames come from the pipeline rather than the camera: these boards serve one client at a
    time, so while the pipeline is streaming nothing else can open them.
    """

    @app.get("/api/cameras")
    def cameras() -> JSONResponse:
        return JSONResponse({"cameras": _camera_states(frames, zones, drift)})

    @app.get("/frame/{camera}.jpg")
    def frame(camera: str) -> Response:
        jpeg = frames.jpeg(camera)
        if jpeg is None:
            raise HTTPException(status_code=404, detail="no frame from that camera yet")
        # No caching: the point of this endpoint is that it is current.
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )


def _add_zone_routes(app: FastAPI, frames, zones: ZoneStore, drift: dict) -> None:
    """Writing and clearing the hand-drawn zones."""

    @app.post("/api/zone")
    def set_zone(body: ZoneRequest) -> JSONResponse:
        """Store a drawn zone, and adopt the frame it was drawn on as the drift reference.

        The two belong together: a zone means something only for as long as the view it was
        drawn on holds, so saving one resets what "moved" is measured against.
        """
        image = frames.raw(body.camera)
        if image is None:
            raise HTTPException(status_code=404, detail="no frame from that camera yet")
        try:
            zone = DrawnZone.from_corners(body.x1, body.y1, body.x2, body.y2)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        zones.save(body.camera, zone)
        watch = drift.get(body.camera)
        if watch is not None:
            watch.remember(image)
        return JSONResponse(
            {
                "camera": body.camera,
                "zone": zone.as_tuple(),
                # The running pipeline is told, so it judges against the new box from the
                # next frame. Nothing to restart.
                "applies_now": True,
            }
        )

    @app.delete("/api/zone/{camera}")
    def clear_zone(camera: str) -> JSONResponse:
        """Drop a drawn zone, so the camera falls back to whatever the config says.

        Movement watching goes with it: it exists to tell you a drawn zone has gone stale, and
        with no drawn zone there is nothing to redraw.
        """
        if not zones.remove(camera):
            raise HTTPException(status_code=404, detail="that camera has no drawn zone")
        watch = drift.get(camera)
        if watch is not None:
            watch.forget()
        return JSONResponse({"camera": camera, "zone": None, "applies_now": True})


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


def create_app(review: ReviewQueue, frames=None, zones=None, drift=None) -> FastAPI:
    """Build the labelling UI over an existing review queue."""
    app = FastAPI(title="stuhi labelling")
    templates = Jinja2Templates(directory=str(_HERE / "templates"))
    app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")

    @app.get("/")
    def index(request: Request):
        return templates.TemplateResponse(request, "index.html")

    @app.get("/zones")
    def zones_page(request: Request):
        return templates.TemplateResponse(request, "zones.html")

    _add_api_routes(app, review)
    _add_media_routes(app, review)
    if frames is not None and zones is not None:
        _add_camera_routes(app, frames, zones, drift or {})
        _add_zone_routes(app, frames, zones, drift or {})
    return app


class WebUI:
    """Runs the labelling UI in a background thread for the lifetime of the pipeline."""

    def __init__(
        self,
        review: ReviewQueue,
        frames=None,
        zones=None,
        drift=None,
        host: str = "0.0.0.0",
        port: int = 8800,
    ) -> None:
        self._review = review
        self._frames = frames
        self._zones = zones
        self._drift = drift
        self._host = host
        self._port = port
        self._server = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        import uvicorn

        config = uvicorn.Config(
            create_app(self._review, self._frames, self._zones, self._drift),
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
