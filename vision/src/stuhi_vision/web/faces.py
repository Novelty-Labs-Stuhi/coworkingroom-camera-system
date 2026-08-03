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


def _linkage(record, name: str) -> str | None:
    """How this frame is tied to ``name``, or None if it is not.

    Three kinds, and the difference between them is the whole point:

    * **labelled** -- a person said this is them. Only this kind becomes evidence the named
      gallery is matched against.
    * **guessed** -- the system filed the frame under a name it invented for somebody it did not
      recognise. The frame is real and is theirs as far as anything here knows, but nobody has
      confirmed it, so it teaches the named gallery nothing.
    * **attributed** -- somebody said who it was on a picture too poor to learn from. It counts
      towards their hours, never towards recognition.

    Only the first was listed until now, which is why fifty of seventy-one people on a live board
    had an empty gallery and no portrait while their frames sat on disk: an identity the system
    invents carries its name in ``name``, not in ``labelled_as``.

    A frame set aside as nobody (``labelled_as == "unknown"``) is deliberately not returned under
    the name the system had guessed for it. Somebody looked at that frame and said it belongs to
    no one; putting it back on a profile would be arguing with them.
    """
    if record.labelled_as == name:
        return "labelled"
    if record.labelled_as is None and record.name == name:
        return "guessed"
    if record.attributed_to == name:
        return "attributed"
    return None


_WHY = {
    "guessed": "the system's own guess — nobody has confirmed it",
    "attributed": "named on a picture too poor to learn from",
}


def gallery_view(review, name: str, sort: str = "latest") -> dict:
    """Every frame linked to one person, in the asked-for order, each saying what its link is.

    ``sort`` is "latest" or "used" -- newest first, or the ones recognition uses first and
    closest to that person's average within them. The second is the order to check a gallery in:
    it is the faces at the top that decide who somebody is.
    """
    links: dict[str, str] = {}
    records = {}
    for record in review.everything():
        if record.dismissed:
            continue
        link = _linkage(record, name)
        if link is None:
            continue
        records[record.sighting_id] = record
        links[record.sighting_id] = link

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
            # How the frame is tied to this name, so the page never presents a guess as though
            # somebody had confirmed it.
            "link": links[sighting],
            "confirmed": links[sighting] == "labelled",
            "why": (
                reasons.get(sighting, "in use" if sighting in used else "not chosen")
                if links[sighting] == "labelled"
                else _WHY[links[sighting]]
            ),
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
        # In use, then confirmed, then closest to the average. An unconfirmed frame cannot have a
        # closeness -- there is no average to measure it against until somebody names it -- so
        # without the middle term every guess would rank alongside a confirmed outlier.
        frames.sort(
            key=lambda frame: (not frame["in_use"], not frame["confirmed"], -frame["closeness"])
        )
    else:
        frames.sort(key=lambda frame: frame["timestamp"], reverse=True)
    return {
        "name": name,
        "frames": frames,
        "in_use": len(used),
        "kept": len(frames),
        "confirmed": sum(1 for frame in frames if frame["confirmed"]),
        "guessed": sum(1 for frame in frames if frame["link"] == "guessed"),
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

    Confirmed frames come before guesses, and among frames with no closeness to go on -- which
    is every unconfirmed one, there being no average to measure against yet -- the newest wins.
    That keeps the answer stable: asked twice about the same person it must name the same face,
    or a profile would change its portrait every time somebody walked past a camera.

    None when there is no picture to show -- a person can be entirely real and have no crop
    saved yet, and inventing a portrait for them is not available.
    """
    havable = [frame for frame in frames if frame["has_picture"]]
    if not havable:
        return None
    best = max(
        havable,
        key=lambda frame: (
            frame["in_use"],
            frame["confirmed"],
            frame["closeness"],
            frame["timestamp"],
        ),
    )
    return best["id"]
