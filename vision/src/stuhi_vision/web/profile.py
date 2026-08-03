"""One person's profile: who they are, what they look like to the system, and when they were here.

The leaderboard answers "who"; this answers everything else about one of them. It is four
things on one page, and they are on one page because each is a way into correcting the same
mistake:

* their **name**, which is the label the system uses -- editable, and editing it corrects it
  everywhere rather than only here;
* their **bio**, the only thing on the page a camera did not produce;
* their **picture**, which opens the faces recognition is matching them against;
* their **year of days**, and any one day opened up into the entries and exits it is made of,
  each with the frame it came from -- because a wrong box on the chart is a wrong label on a
  frame, and this is where you can see which.

Its own module rather than more of :mod:`.app`: that one is already routing seven pages' worth
of endpoints, and a profile is a page in its own right. The answers are built by the functions
below and the routes only wire them, so each answer can be tested without a request.
"""

from __future__ import annotations

import time
from datetime import date

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.requests import Request

from ..activity import SHADES, calendar, held_on, on_day, spanning
from ..presence import (
    Passage,
    Visit,
    counting_from,
    office_day,
    readable,
    recorded_visits,
    streaks,
    totals,
)
from ..profiles import ProfileStore
from ..review import ReviewQueue
from .assets import asset_version
from .faces import gallery_view, representative

# A year of weeks, which is what the chart this copies shows. Fifty-three rather than
# fifty-two: a year is a fraction over, and the extra column is how today stays on the right.
WEEKS = 53


class BioRequest(BaseModel):
    """A person's own words about themselves. Empty clears them."""

    name: str
    bio: str = ""


def visits_of(review: ReviewQueue, history, name: str) -> tuple[list[Visit], float]:
    """One person's visits, and when counting began.

    ``history`` may be None -- a labelling-only deployment records no crossings at all, and a
    profile is still worth showing without them.
    """
    epoch = counting_from(review.directory)
    if history is None:
        return [], epoch
    found = recorded_visits(history.passages(), epoch)
    return [visit for visit in found if visit.name == name], epoch


def header_for(
    review: ReviewQueue, profiles: ProfileStore, name: str, found: list[Visit], epoch: float,
    now: float,
) -> dict:
    """The top of the profile: the name, the bio, the picture, and the figures beside them."""
    view = gallery_view(review, name, sort="used")
    run = next((standing.seconds for standing in streaks(found, now) if standing.name == name), 0.0)
    return {
        "name": name,
        "bio": profiles.bio(name),
        # The face to show, and how many are behind it -- the picture is the way into the
        # gallery, so it says what it opens.
        "picture": representative(view["frames"]),
        "faces": {"in_use": view["in_use"], "kept": view["kept"]},
        "totals": {
            window: {"seconds": round(seconds, 1), "readable": readable(seconds)}
            for window, seconds in totals(found, epoch, now).items()
        },
        "streak_days": int(run),
        "inside": any(visit.left is None for visit in found),
        "visits": len(found),
    }


def chart_for(name: str, found: list[Visit], epoch: float, now: float, weeks: int) -> dict:
    """A box per day for the last ``weeks`` weeks, shaded by how long they were here."""
    first, last = spanning(now, weeks)
    return {
        "name": name,
        "from": first.isoformat(),
        "until": last.isoformat(),
        "shades": SHADES,
        "counting_from": epoch,
        "days": [
            {
                "date": day.day.isoformat(),
                "seconds": round(day.seconds, 1),
                "readable": readable(day.seconds) if day.seconds else "",
                "visits": day.visits,
                "counted": day.counted,
                "level": day.level,
            }
            for day in calendar(found, first, last, epoch, now)
        ],
    }


def day_for(
    review: ReviewQueue, history, name: str, found: list[Visit], epoch: float, asked: date,
    now: float,
) -> dict:
    """One day opened up: every entry and exit, with the frame each was decided from.

    The frames are the point. A day that reads wrong on the chart reads wrong because a crossing
    carries the wrong name, and the only way to see that is to look at the face the system was
    looking at -- so each crossing arrives with its sighting, ready to correct through the
    ordinary labelling route.
    """
    crossings: list[Passage] = []
    if history is not None:
        mine = [
            Passage(name=who, at=at, direction=direction)
            for who, at, direction in history.passages()
            if who == name and at >= epoch
        ]
        crossings = on_day(mine, asked)

    held = held_on(found, asked, now)
    return {
        "name": name,
        "date": asked.isoformat(),
        "today": asked == office_day(now),
        "seconds": round(held, 1),
        "readable": readable(held),
        "crossings": [_crossing_dict(review, passage) for passage in crossings],
    }


def _crossing_dict(review: ReviewQueue, passage: Passage) -> dict:
    """One entry or exit, with whatever the system saw at that moment."""
    record = review.nearest(passage.at, passage.direction)
    return {
        "at": passage.at,
        "direction": passage.direction,
        "sighting_id": record.sighting_id if record else None,
        # What that frame is filed as *now*, which is not always the name on the crossing: a
        # frame relabelled without the crossing following it is exactly the disagreement worth
        # seeing here.
        "labelled_as": record.display_name if record else None,
        "has_picture": bool(record and review.crop_path(record.sighting_id)),
        "score": record.score if record else None,
    }


def _asked_date(when: str) -> date:
    try:
        return date.fromisoformat(when)
    except ValueError as error:
        raise HTTPException(status_code=400, detail="expected a date as YYYY-MM-DD") from error


def add_profile_routes(
    app: FastAPI,
    review: ReviewQueue,
    profiles: ProfileStore,
    history,
    templates: Jinja2Templates,
) -> None:
    """Wire the profile page and the three questions it asks."""

    @app.get("/profile")
    def profile_page(request: Request):
        return templates.TemplateResponse(
            request, "profile.html", {"assets": asset_version(), "page": "profile"}
        )

    @app.get("/api/profile/{name}")
    def profile(name: str) -> JSONResponse:
        now = time.time()
        found, epoch = visits_of(review, history, name)
        return JSONResponse(header_for(review, profiles, name, found, epoch, now))

    @app.post("/api/bio")
    def write_bio(body: BioRequest) -> JSONResponse:
        try:
            stored = profiles.write(body.name, body.bio)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return JSONResponse({"name": body.name.strip(), "bio": stored})

    @app.get("/api/profile/{name}/activity")
    def activity(name: str, weeks: int = WEEKS) -> JSONResponse:
        now = time.time()
        found, epoch = visits_of(review, history, name)
        return JSONResponse(chart_for(name, found, epoch, now, weeks))

    @app.get("/api/profile/{name}/day/{when}")
    def day(name: str, when: str) -> JSONResponse:
        now = time.time()
        found, epoch = visits_of(review, history, name)
        return JSONResponse(
            day_for(review, history, name, found, epoch, _asked_date(when), now)
        )
