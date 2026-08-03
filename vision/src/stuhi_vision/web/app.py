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
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.requests import Request

from ..names import parse_names
from ..presence import (
    available_from,
    counting_from,
    covered,
    leaderboard,
    readable,
    recorded_visits,
    streaks,
    totals,
    unclosed,
    window_bounds,
)
from ..profiles import ProfileStore
from ..review import ReviewQueue
from ..zones import DrawnZone, ZoneStore
from .assets import asset_version
from .faces import gallery_view
from .profile import add_profile_routes

_HERE = Path(__file__).parent


class LabelRequest(BaseModel):
    sighting_id: str
    name: str


class DismissRequest(BaseModel):
    sighting_id: str
    # Why, in the rejecter's own words. Optional, because a rejection with no reason still
    # has to be possible -- demanding one would mean bad clips left in the gallery.
    note: str = ""
    # Who it was, if you know despite the picture being unusable. Their hours still count; the
    # picture still teaches the recogniser nothing.
    name: str = ""


class UseFaceRequest(BaseModel):
    """Whether one face is matched against: yes, no, or leave it to the rule."""

    sighting_id: str
    # None means "let the rule decide" -- the way to undo a decision without inverting it.
    wanted: bool | None = None


class RenameRequest(BaseModel):
    """Correct a name wherever it was used -- a misspelling is one mistake, not one per clip."""

    old: str
    new: str


class ZoneRequest(BaseModel):
    """A rectangle dragged on a camera's live frame, in fractions of it."""

    camera: str
    x1: float
    y1: float
    x2: float
    y2: float


def _as_dict(record, groups: dict[int, list[str]] | None = None) -> dict:
    together = (groups or {}).get(record.burst, [])
    size = len(together) or 1
    return {
        "id": record.sighting_id,
        # Why it was rejected, if it was. Kept in the record so somebody working on the
        # detector can read what it is actually getting wrong.
        "rejected_because": record.rejected_because,
        "attributed_to": record.attributed_to,
        "dismissed": record.dismissed,
        "checked": record.checked,
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
        # Every face in this group, in crossing order, so the order a group label depends on
        # can be checked against the pictures instead of taken on trust.
        "group_ids": together if size > 1 else [],
    }


def _suspect_dicts(review: ReviewQueue, report) -> list[dict]:
    """Pair each suspect reference with its sighting, so the UI can show the clip."""
    groups = review.groups()
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


def _not_yet(window: str, offset: int, epoch: float, now: float, ready: float) -> JSONResponse:
    """Counting has not begun, or has only just begun. Says which, and when it will mean
    something.

    The only remaining case: either the epoch is in the future (switched on before 04:30 today)
    or it is less than five minutes old. Everything after that shows its figures.
    """
    return JSONResponse(
        {
            "window": window,
            "offset": offset,
            "from": epoch,
            "until": now,
            "ready": False,
            "ready_at": ready,
            "days_covered": 0,
            "days_in_window": 0,
            "partial": False,
            "standings": [],
        }
    )


def _standings(window: str, offset: int, found, epoch: float, now: float):
    """The rows for one window, and the period they cover."""
    if window == "streak":
        return streaks(found, now), epoch, now
    start, end = window_bounds(window, offset, now)
    start = max(start, epoch)
    return leaderboard(found, start, end, now), start, end


def _add_presence_routes(app: FastAPI, review: ReviewQueue, history) -> None:
    """Time in the room, and the unclosed entries a correction may need to reach.

    Derived from the crossings each time rather than kept as totals: a corrected name changes
    the answer, and a stored total would be wrong until somebody remembered to rebuild it.
    """

    def visits(since: float):
        """Visits from the crossings, ignoring everything before counting began."""
        return recorded_visits(history.passages(), since)

    @app.get("/api/leaderboard")
    def board(window: str = "week", offset: int = 0) -> JSONResponse:
        """Who spent the longest in the room, over one window, ``offset`` windows back.

        Nothing before counting began is included, and every window shows what it has five
        minutes after that. A window the record does not fully cover says so -- ``days_covered``
        of ``days_in_window`` -- rather than being withheld until it does.
        """
        now = time.time()
        epoch = counting_from(review.directory)
        ready = available_from(window, epoch)
        if now < ready:
            return _not_yet(window, offset, epoch, now, ready)

        found = visits(epoch)
        standings, start, end = _standings(window, offset, found, epoch, now)
        days_covered, days_in_window = covered(window, offset, epoch, now)
        return JSONResponse(
            {
                "window": window,
                "offset": offset,
                "from": start,
                "until": end,
                "ready": True,
                "ready_at": ready,
                # How much of the asked-for window is actually behind these figures. The page
                # says "3 of 7 days so far" with it, which is the honest version of hiding the
                # board altogether.
                "days_covered": days_covered,
                "days_in_window": days_in_window,
                "partial": bool(days_in_window and days_covered < days_in_window),
                "standings": [
                    {
                        "name": standing.name,
                        "seconds": round(standing.seconds, 1),
                        "readable": (
                            f"{int(standing.seconds)} day(s)"
                            if window == "streak"
                            else readable(standing.seconds)
                        ),
                        "visits": standing.visits,
                        "days": standing.days,
                        "still_inside": standing.still_inside,
                    }
                    for standing in standings
                ],
            }
        )

    @app.get("/api/person/{name}")
    def person(name: str) -> JSONResponse:
        """One person's own figures, and every visit behind them."""
        now = time.time()
        epoch = counting_from(review.directory)
        found = [visit for visit in visits(epoch) if visit.name == name]
        run = next((s.seconds for s in streaks(found, now) if s.name == name), 0.0)
        return JSONResponse(
            {
                "name": name,
                "totals": {
                    window: {"seconds": round(seconds, 1), "readable": readable(seconds)}
                    for window, seconds in totals(found, epoch, now).items()
                },
                "streak_days": int(run),
                "inside": any(visit.left is None for visit in found),
                "visits": [
                    {
                        "entered": visit.entered,
                        "left": visit.left,
                        "seconds": round(visit.seconds(now), 1),
                        "readable": readable(visit.seconds(now)),
                    }
                    for visit in sorted(found, key=lambda v: -v.entered)[:200]
                ],
            }
        )

    @app.get("/api/inside")
    def inside() -> JSONResponse:
        """Entries with no exit after them, newest first, with the clip for each.

        What to look at when a correction is impossible: an exit can only belong to somebody
        the record has inside, so naming one otherwise means an earlier *entry* carries the
        wrong name -- and that mistake is necessarily one of these.
        """
        open_visits = sorted(
            unclosed(visits(counting_from(review.directory))), key=lambda visit: -visit.entered
        )
        listed = []
        for visit in open_visits:
            record = review.nearest(visit.entered, "in")
            listed.append(
                {
                    "name": visit.name,
                    "entered": visit.entered,
                    "for": readable(visit.seconds(time.time())),
                    "sighting_id": record.sighting_id if record else None,
                }
            )
        return JSONResponse({"inside": listed})


def _add_api_routes(app: FastAPI, review: ReviewQueue) -> None:
    @app.get("/api/sightings")
    def sightings(sort: str = "latest") -> JSONResponse:
        """Every sighting, in the pile that says what has happened to it.

        One request rather than five, because a sighting moves between piles as it is saved
        and separate requests would show it in two at once, or in neither.

        ``sort`` is "latest" or "odd" -- newest first, or furthest from that person's average
        face first, which puts the likeliest mistakes at the top.
        """
        groups = review.groups()
        reasons = {
            record.sighting_id: reason for record, reason in review.worth_rechecking(limit=50)
        }
        piles = review.piles(sort=sort)
        return JSONResponse(
            {
                **{
                    pile: [
                        {**_as_dict(record, groups), "reason": reasons.get(record.sighting_id)}
                        for record in records
                    ]
                    for pile, records in piles.items()
                },
                "people": review.counts(),
                # Most recently used first: the next person through a door is very often
                # somebody who came through recently.
                "recent_names": review.recent_names(),
                "sort": sort,
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


def _apply_rename(
    review: ReviewQueue, history, ledger, profiles: ProfileStore, body: RenameRequest
) -> JSONResponse:
    """Correct one name wherever it was used.

    Merges when the corrected name already exists, because that is what "ilari" and "Ilari"
    being one person means. The history is included deliberately: leaving both spellings there
    makes one person read as two who each came and went half the time. And whoever is inside
    right now moves with it, or their exit would arrive under a name nobody is holding.

    The bio moves too. Spelling somebody's name properly should not cost them their profile.
    """
    names = parse_names(body.new)
    if len(names) != 1:
        raise HTTPException(status_code=400, detail="give exactly one corrected name")
    corrected = names[0]
    relabelled = review.rename(body.old, corrected)
    if relabelled == 0 and body.old not in review.counts():
        raise HTTPException(status_code=404, detail=f"nobody is labelled {body.old}")
    events = history.rename(body.old, corrected) if history is not None else 0
    if ledger is not None:
        ledger.rename(body.old, corrected)
    profiles.rename(body.old, corrected)
    return JSONResponse(
        {
            "renamed": corrected,
            "clips": relabelled,
            "events": events,
            "people": review.counts(),
        }
    )


def _add_accounting_routes(app: FastAPI, review: ReviewQueue, history, passages) -> None:
    """Whether the count adds up: entries with no exit, and exits the room renamed."""

    @app.get("/api/accounting")
    def accounting(pile: str = "missing_exits") -> JSONResponse:
        """The two ways the count goes wrong, since the office day began.

        Both are pairs of a sort. A missing exit comes with the passages that were seen
        afterwards and refused -- the evidence for where it went. A renamed exit comes with the
        entry it was matched to, because if the room named it wrongly then two people are wrong
        at once, and confirming one half without the other proves nothing.
        """
        from ..accounting import renamed_exits, unmatched_entries, with_missed
        from ..presence import day_starting, office_day

        if history is None:
            return JSONResponse({"pile": pile, "since": None, "entries": []})

        from datetime import datetime

        from ..presence import OFFICE

        # The same boundary the presence figures use: the office day starts at 04:30, so
        # somebody in the room at one in the morning is still finishing the previous day --
        # which is why the day is asked for by moment rather than by calendar date. Taking
        # today's date before 04:30 puts the start of the window in the future and returns
        # nothing at all.
        now = datetime.now(OFFICE).timestamp()
        since = day_starting(office_day(now))
        until = now
        crossings = history.crossings(since, until)

        if pile == "renamed_exits":
            found = renamed_exits(crossings, since, until)
            return JSONResponse(
                {
                    "pile": pile,
                    "since": since,
                    "entries": [
                        {
                            "given": pair.given,
                            "natural": pair.natural,
                            "exit_at": pair.exit_at,
                            "entry_at": pair.entry_at,
                            "exit_frame": _frame_at(review, pair.exit_at, "out"),
                            "entry_frame": _frame_at(review, pair.entry_at, "in"),
                            "readable": pair.readable,
                        }
                        for pair in found
                    ],
                }
            )

        refused = passages.refused(since, until) if passages is not None else []
        open_entries = with_missed(unmatched_entries(crossings, since, until), refused, until)
        return JSONResponse(
            {
                "pile": pile,
                "since": since,
                "entries": [
                    {
                        "name": entry.name,
                        "entered_at": entry.entered_at,
                        "entry_frame": _frame_at(review, entry.entered_at, "in"),
                        "missed": list(entry.missed),
                        "readable": entry.readable,
                    }
                    for entry in open_entries
                ],
            }
        )


def _frame_at(review: ReviewQueue, moment: float | None, direction: str) -> str | None:
    """The sighting recorded for a crossing, so a pair can be shown as pictures.

    Matched on the moment, because that is what the two records share: the crossing wrote the
    event and the sighting from the same timestamp. A second either way covers the rounding.
    """
    if moment is None:
        return None
    for record in review.everything():
        if record.direction == direction and abs(record.timestamp - moment) <= 1.0:
            return record.sighting_id
    return None


def _add_person_routes(app: FastAPI, review: ReviewQueue) -> None:
    """One person: every face of theirs, and which ones recognition is allowed to use."""

    # Deliberately not /person: that is the presence page, and its figures are a different
    # question about the same person. This one is about their gallery.
    @app.get("/gallery")
    def gallery_page(request: Request):
        return Jinja2Templates(directory=str(_HERE / "templates")).TemplateResponse(
            request, "person.html", {"assets": asset_version()}
        )

    @app.get("/api/person/{name}/faces")
    def faces(name: str, sort: str = "latest") -> JSONResponse:
        """Every face filed under this name, with why each is or is not matched against.

        ``sort`` is "latest" or "used" -- newest first, or the ones recognition uses first and
        closest to that person's average within them. The second is the order to check the
        gallery in: it is the faces at the top that decide who somebody is.
        """
        return JSONResponse(gallery_view(review, name, sort))

    @app.post("/api/use-face")
    def use_face(body: UseFaceRequest) -> JSONResponse:
        outcome = review.use_face(body.sighting_id, body.wanted)
        if not outcome.succeeded:
            raise HTTPException(status_code=404, detail=outcome.value)
        record = review.get(body.sighting_id)
        return JSONResponse(
            {
                "sighting_id": body.sighting_id,
                "wanted": body.wanted,
                "name": record.labelled_as if record else None,
            }
        )


def _add_correction_routes(
    app: FastAPI, review: ReviewQueue, history, ledger, profiles: ProfileStore
) -> None:
    """Taking a label back, and correcting a name everywhere it was used."""

    @app.post("/api/dismiss")
    def dismiss(body: DismissRequest) -> JSONResponse:
        """Reject a sighting: stop offering it, and remove any reference it contributed."""
        outcome = review.dismiss(body.sighting_id, body.note, body.name)
        if not outcome.succeeded:
            raise HTTPException(status_code=404, detail=outcome.value)
        return JSONResponse({"outcome": outcome.value, "people": review.counts()})

    @app.post("/api/rename")
    def rename(body: RenameRequest) -> JSONResponse:
        """Correct a spelling everywhere: the gallery, every labelled clip, and the history."""
        return _apply_rename(review, history, ledger, profiles, body)

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


def create_app(
    review: ReviewQueue,
    frames=None,
    zones=None,
    drift=None,
    history=None,
    ledger=None,
    passages=None,
) -> FastAPI:
    """Build the labelling UI over an existing review queue."""
    app = FastAPI(title="stuhi labelling")
    templates = Jinja2Templates(directory=str(_HERE / "templates"))
    app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")
    # Beside the review data, like the epoch: it belongs to this deployment, not to the code.
    # Built here rather than passed in, so a deployment gains profiles without being rewired.
    profiles = ProfileStore(review.directory / "profiles.json")

    @app.get("/")
    def index(request: Request):
        return templates.TemplateResponse(
            request, "index.html", {"assets": asset_version(), "page": "label"}
        )

    @app.get("/stats")
    def stats_page(request: Request):
        return templates.TemplateResponse(
            request, "stats.html", {"assets": asset_version(), "page": "stats"}
        )

    @app.get("/zones")
    def zones_page(request: Request):
        return templates.TemplateResponse(
            request, "zones.html", {"assets": asset_version(), "page": "zones"}
        )

    _add_api_routes(app, review)
    _add_correction_routes(app, review, history, ledger, profiles)
    _add_person_routes(app, review)
    _add_accounting_routes(app, review, history, passages)
    # Registered even with no history: a profile still has a name, a bio and a picture, and the
    # dated parts of it say they have nothing rather than the page failing to open.
    add_profile_routes(app, review, profiles, history, templates)
    if history is not None:
        _add_presence_routes(app, review, history)
    _add_media_routes(app, review)
    if frames is not None and zones is not None:
        _add_camera_routes(app, frames, zones, drift or {})
        _add_zone_routes(app, frames, zones, drift or {})
    return app


class WebUI:
    """Runs the labelling UI in a background thread for the lifetime of the pipeline."""

    def __init__(self, app: FastAPI, host: str = "0.0.0.0", port: int = 8800) -> None:
        self._app = app
        self._host = host
        self._port = port
        self._server = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        import uvicorn

        config = uvicorn.Config(
            self._app,
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
