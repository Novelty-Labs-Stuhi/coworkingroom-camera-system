"""One person's gallery, as a page needs to see it.

Two questions, both about the same set of faces:

* **what is this person's gallery made of** -- every face kept under their name, whether
  recognition is matching against it, and why. That is the frames page.
* **which single face is this person** -- the one to show as their picture. The profile page
  asks this, and the answer has to be the same face each time it is asked, or a profile would
  change its portrait every time somebody walked past a camera.

Kept out of the routing module because both answers are rules worth testing on their own, and
because the second is not obvious: see :func:`representative`.
"""

from __future__ import annotations

from ..selection import distances


def gallery_view(review, name: str, sort: str = "latest") -> dict:
    """This person's faces, in the asked-for order, each saying whether it is in use and why.

    ``sort`` is "latest" or "used" -- newest first, or the ones recognition uses first and
    closest to that person's average within them. The second is the order to check a gallery in:
    it is the faces at the top that decide who somebody is.
    """
    records = {
        record.sighting_id: record
        for record in review.everything()
        if record.labelled_as == name and not record.dismissed
    }
    picked = review.chosen.get(name)
    used = set(getattr(picked, "used", ()))
    reasons = {
        **{sighting: "too old" for sighting in getattr(picked, "too_old", ())},
        **{sighting: "unlike the average" for sighting in getattr(picked, "too_odd", ())},
        **{sighting: "you asked for it" for sighting in getattr(picked, "pinned", ())},
        **{sighting: "you excluded it" for sighting in getattr(picked, "barred", ())},
    }
    ages = {sighting: record.timestamp for sighting, record in records.items()}
    closeness = distances([r for r in review.references() if r.name == name], ages)

    frames = [
        {
            "id": sighting,
            "timestamp": record.timestamp,
            "direction": record.direction,
            "in_use": sighting in used,
            "why": reasons.get(sighting, "in use" if sighting in used else "not chosen"),
            "closeness": round(closeness.get(sighting, 0.0), 3),
            "decided": record.use_for_matching,
            "rejected_because": record.rejected_because,
            # Whether there is a picture to show. A sighting can be a perfectly good reference
            # vector with no crop saved beside it, and a portrait cannot be one of those.
            "has_picture": review.crop_path(sighting) is not None,
        }
        for sighting, record in records.items()
    ]
    if sort == "used":
        frames.sort(key=lambda frame: (not frame["in_use"], -frame["closeness"]))
    else:
        frames.sort(key=lambda frame: frame["timestamp"], reverse=True)
    return {
        "name": name,
        "frames": frames,
        "in_use": len(used),
        "kept": len(frames),
    }


def representative(frames: list[dict]) -> str | None:
    """The face to show as somebody's picture: the one closest to their own average.

    Closest to the average, rather than the newest, because the average *is* what the system
    thinks this person looks like -- so the nearest face to it is the most ordinary picture of
    them there is. The newest would be whatever the door caught this morning, which is as likely
    to be a blurred shoulder as a portrait.

    Faces recognition is actually using come first, even when an unused one sits marginally
    closer. A face is unused because it is too old or was ruled out by hand, and a profile
    should not be represented by a picture the system has stopped believing in.

    None when there is no picture to show -- a person can be entirely real and have no crop
    saved yet, and inventing a portrait for them is not available.
    """
    havable = [frame for frame in frames if frame["has_picture"]]
    if not havable:
        return None
    best = max(havable, key=lambda frame: (frame["in_use"], frame["closeness"]))
    return best["id"]
