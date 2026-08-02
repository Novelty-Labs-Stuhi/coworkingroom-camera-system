"""Sightings awaiting a human verdict, and the labelling that teaches the gallery.

Every committed crossing is filed here with the face embedding and crop of its best
frame. That turns "we do not know this person" from a dead end into the mechanism by
which the system improves: label the sighting and its embedding is enrolled into the
gallery, so the next time that person walks in they are recognised.

Labelling an already-labelled sighting *corrects* it -- the embedding is removed from the
wrong name before being added to the right one, so one mistake does not poison the
gallery forever.

Layout under the review directory, one set per sighting id::

    2026-07-27_18-04-11.npy    the face embedding
    2026-07-27_18-04-11.jpg    the face crop, for sending to a human
    index.json                 the sighting records

A label reaches the event log too, and *how far* depends on what is being said. Naming an
identity the system invented for itself corrects every crossing it ever made, because that is
what the statement means and because entries pair with exits by name -- renaming one half of a
visit would leave an entry that never closes. Correcting a name the recogniser chose from the
gallery touches only that crossing: the person it named is still themselves everywhere else.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np

from .domain import Outcome, Sighting
from .enrolment import Audit, Reference, audit
from .merges import LABELLED, merge_identity
from .names import parse_names
from .recognition.gallery import FaceGallery
from .selection import choose

_INDEX = "index.json"


_log = logging.getLogger(__name__)


class LabelOutcome(Enum):
    """What a labelling request actually did."""

    ENROLLED = "enrolled"  # first label for this sighting; one reference added
    CORRECTED = "corrected"  # moved from a previous name; no duplicate left behind
    UNCHANGED = "unchanged"  # already labelled that way, so nothing was added
    DISMISSED = "dismissed"  # marked unusable; any reference it contributed was removed
    UNLABELLED = "unlabelled"  # label taken back; the sighting returns to the pending list
    NO_FACE = "no_face"  # nothing to enrol -- no embedding was kept
    UNKNOWN_ID = "unknown_id"
    # Saved as nobody in particular. "unknown" is not a person, so nothing is enrolled under
    # it -- a gallery entry by that name would compete with the real people and match anybody
    # the recogniser was unsure about. What it does mean is that somebody has looked, which
    # is why it counts as checked and leaves the queue.
    SET_ASIDE = "set_aside"
    # Named, but nothing enrolled: the crossing had no usable face, so who it was is recorded
    # and the recogniser learns nothing. Refusing instead left the card stuck in the queue.
    ATTRIBUTED = "attributed"

    @property
    def succeeded(self) -> bool:
        return self in (
            LabelOutcome.ENROLLED,
            LabelOutcome.CORRECTED,
            LabelOutcome.UNCHANGED,
            LabelOutcome.DISMISSED,
            LabelOutcome.UNLABELLED,
            LabelOutcome.SET_ASIDE,
            LabelOutcome.ATTRIBUTED,
        )


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    """One sighting as filed for review."""

    sighting_id: str
    timestamp: float
    direction: str
    outcome: str
    name: str | None
    score: float
    labelled_as: str | None = None
    # Place within a burst -- a group of people who crossed together with no clear gap.
    # Their clips are cut from the same window and look alike, so the order they crossed in
    # is the only thing distinguishing them, and it is how a whole group gets labelled at
    # once. Defaults keep records written before this existed loadable.
    position: int = 1
    burst: int = 0
    # Marked unusable by a human -- back of a head, motion blur, nobody really there. Kept
    # rather than deleted so it stops being offered without losing the evidence.
    dismissed: bool = False
    # Why a clip was rejected, in the rejecter's own words. Kept with the record rather than
    # in a log: "back of a head", "that is the door, not a person", "two people, one box" are
    # what tell somebody working on the detector what it is actually getting wrong, and a log
    # is rotated within hours.
    rejected_because: str = ""
    # Who it was, on a picture too poor to learn from. Kept apart from ``labelled_as`` on
    # purpose: that one means "enrolled under this name", and enrolling a bad picture is
    # exactly what rejecting it is meant to prevent. This only says who came through, which is
    # what the time-in-the-room figures need.
    attributed_to: str = ""
    # What a person decided about using this face for recognition: True to use it whatever the
    # numbers say, False never to use it, None to let the rule choose. Their choice wins --
    # somebody who has looked knows things the numbers do not, such as that this is the only
    # picture of a colleague with their new beard.
    use_for_matching: bool | None = None
    # A person has looked at this and saved it. It never returns to "worth rechecking":
    # whatever the audit thinks of the numbers, somebody has judged it, and offering it back
    # would be arguing with them for ever.
    checked: bool = False

    @property
    def display_name(self) -> str:
        return self.labelled_as or self.name or "unknown"


class ReviewQueue:
    """Disk-backed sightings plus the label/correct operations over the gallery."""

    def __init__(
        self,
        directory: Path,
        gallery: FaceGallery,
        gallery_dir: Path,
        history=None,
        provisional=None,
        merges=None,
    ) -> None:
        self._dir = directory
        self._gallery = gallery
        self._gallery_dir = gallery_dir
        # The event log, so a name corrected here reaches the record the figures are derived
        # from. Without it the totals keep whatever the system guessed at the time, and no
        # amount of careful labelling would ever change them. Optional, because labelling works
        # perfectly well on its own -- it is the *figures* that need this.
        self._history = history
        # Which identities the system named itself. Needed to answer "who has nobody named
        # yet", which is a question about identities rather than about sightings -- and so
        # cannot be derived from the records alone.
        self._provisional = provisional
        # Where an identity folded into another is written down before it is folded, so a
        # mistaken merge can be taken apart again. Without it, naming the wrong person would
        # fuse two people's histories permanently on the strength of one click.
        self._merges = merges
        self._dir.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, ReviewRecord] = {}
        # Sightings are filed by the pipeline thread and labelled by the chat poller and
        # the web UI, so the record index needs guarding as much as the gallery does.
        self._lock = threading.RLock()
        self._load()
        self.refresh_matching()

    # --- recording ----------------------------------------------------------
    def labelled(self, limit: int = 50) -> list[ReviewRecord]:
        """Sightings that already carry a label, newest first, so one can be corrected."""
        with self._lock:
            done = [r for r in self._records.values() if r.labelled_as is not None]
        done.sort(key=lambda record: record.timestamp, reverse=True)
        return done[:limit]

    def record(
        self,
        sighting: Sighting,
        encode_jpeg=None,
        clip: bytes | None = None,
        position: int = 1,
        burst: int = 0,
    ) -> str:
        """File a sighting and return its id.

        ``encode_jpeg`` turns the crop into JPEG bytes; injected so this module needs no
        image library (and so tests need no OpenCV). ``clip`` is an optional MP4 of the
        moment, kept alongside so a human can see what happened rather than only a crop.
        """
        sighting_id = self._next_id(sighting.timestamp)
        if sighting.face_embedding is not None:
            np.save(self._dir / f"{sighting_id}.npy", sighting.face_embedding)
        if sighting.face_crop is not None and encode_jpeg is not None:
            jpeg = encode_jpeg(sighting.face_crop)
            if jpeg:
                (self._dir / f"{sighting_id}.jpg").write_bytes(jpeg)
        if clip:
            (self._dir / f"{sighting_id}.mp4").write_bytes(clip)
        self._records[sighting_id] = ReviewRecord(
            sighting_id=sighting_id,
            timestamp=sighting.timestamp,
            direction=sighting.direction.value,
            outcome=sighting.outcome.value,
            name=sighting.name,
            score=round(sighting.score, 4),
            position=position,
            burst=burst,
        )
        self._flush()
        return sighting_id

    # --- labelling ----------------------------------------------------------
    def label(self, sighting_id: str, name: str) -> LabelOutcome:
        """Enrol a sighting's face under ``name``, correcting any previous label.

        Labelling the same sighting the same way twice is deliberately a no-op: one
        sighting contributes exactly one reference vector, however many times it is
        labelled. Otherwise a repeated command would quietly weight that one face more
        heavily than everybody else's.

        The outcome is reported rather than a bare success flag, so the person labelling
        can tell "added" from "already like that" instead of guessing.
        """
        with self._lock:
            record = self._records.get(sighting_id)
            if record is None:
                return LabelOutcome.UNKNOWN_ID
            if _is_nobody(name):
                return self._set_aside(record)
            embedding_path = self._dir / f"{sighting_id}.npy"
            if not embedding_path.exists():
                return self._attribute(record, name)

            previous = record.labelled_as
            embedding = np.load(embedding_path)

            # Two things can make this a no-op: the record already says this name, or the
            # vector is already enrolled under it (the same sighting labelled from the chat
            # and from the web UI). Either way one sighting must count exactly once.
            if previous == name or self._gallery.contains(name, embedding):
                if previous != name:
                    self._remember(record, name)
                return LabelOutcome.UNCHANGED

            if previous is not None:
                self._gallery.discard(previous, embedding)
            self._gallery.add(name, embedding)
            self._gallery.save(self._gallery_dir)
            self._remember(record, name)
            return LabelOutcome.CORRECTED if previous is not None else LabelOutcome.ENROLLED

    def _attribute(self, record: ReviewRecord, name: str) -> LabelOutcome:
        """Record who somebody says this was, when there is no face to enrol from it.

        Ninety-seven of the sightings waiting had no face vector -- crossings where a person
        was seen but no usable face was. Saving a name on one was refused outright, so the card
        could never leave the queue: the only way past it was to reject the clip.

        Who came through and what the recogniser should learn from are different things. The
        crossing is recorded as this person, so their hours count; nothing is enrolled, because
        there is nothing to enrol. Same field the reject path uses for the same reason.
        """
        self._records[record.sighting_id] = ReviewRecord(
            **{**asdict(record), "attributed_to": name.strip(), "checked": True}
        )
        self._flush()
        return LabelOutcome.ATTRIBUTED

    def _set_aside(self, record: ReviewRecord) -> LabelOutcome:
        """Save a sighting as nobody in particular: checked, but enrolled under no name.

        "unknown" is not a person. Enrolling it would put an entry in the gallery that
        competes with the real people and matches anybody the recogniser was unsure about --
        the same fault that produced names like "a, yehor". But saying so *is* a decision, and
        a decision has to take the sighting out of the queue, or the only way to clear an
        unrecognisable frame would be to give it somebody's name.
        """
        previous = record.labelled_as
        if previous and not _is_nobody(previous):
            path = self._dir / f"{record.sighting_id}.npy"
            if path.exists():
                # Whatever it was enrolled as before is now withdrawn: the person has said
                # this face belongs to nobody, so it must stop influencing recognition.
                self._gallery.discard(previous, np.load(path))
                self._gallery.save(self._gallery_dir)
        self._records[record.sighting_id] = ReviewRecord(
            **{**asdict(record), "labelled_as": "unknown", "checked": True}
        )
        self._flush()
        # The crossing becomes unknown too. Leaving the system's guess there would credit
        # somebody with hours a person has just said belong to nobody.
        self._correct_history(record, "unknown")
        return LabelOutcome.SET_ASIDE

    def label_burst(self, sighting_id: str, names: list[str]) -> list[tuple[str, LabelOutcome]]:
        """Label everyone who crossed alongside ``sighting_id``, in crossing order.

        ``names[0]`` goes to whoever crossed first, and so on. Extra names are ignored and
        missing ones leave that person unlabelled, so a mistaken count cannot silently
        attach the wrong name to somebody.
        """
        members = self.burst_members(sighting_id)
        return [
            (member.sighting_id, self.label(member.sighting_id, name))
            for member, name in zip(members, names, strict=False)
        ]

    def burst_members(self, sighting_id: str) -> list[ReviewRecord]:
        """Everyone who crossed in the same burst as this sighting, in crossing order."""
        with self._lock:
            record = self._records.get(sighting_id)
            if record is None:
                return []
            if not record.burst:  # written before bursts existed: it stands alone
                return [record]
            members = [r for r in self._records.values() if r.burst == record.burst]
        return sorted(members, key=lambda r: r.position)

    def _remember(self, record: ReviewRecord, name: str) -> None:
        # Whether the name being replaced was one the system invented, decided *here* rather
        # than inside _correct_history. Claiming it below makes the answer change, so asking
        # afterwards would always say no and the identity would never be merged.
        invented = bool(
            self._provisional is not None
            and record.name
            and self._provisional.holds(record.name)
        )
        # A human has answered for whoever this was, so the identity the system invented for
        # them stops being an open question -- even though the gallery may still hold that
        # name until somebody merges it. Without this, labelling one of a stranger's ten
        # sightings would leave the other nine to put the same card back on the page.
        if invented:
            self._provisional.claimed(record.name)
        self._records[record.sighting_id] = ReviewRecord(
            **{**asdict(record), "labelled_as": name, "checked": True}
        )
        self._flush()
        self._correct_history(record, name, whole_identity=invented)
        # The matching set follows every change to the labelled faces, or a correction would
        # not reach recognition until the next restart.
        self.refresh_matching()

    def _correct_history(
        self, record: ReviewRecord, name: str, whole_identity: bool = False
    ) -> None:
        """Carry a label into the event log, so the presence figures follow the correction.

        Two different statements wear the same clothes here, and applying the wrong one is
        how labelling damages the record instead of repairing it:

        * naming an identity **the system invented** says who that identity *is*, and it was
          always true -- so every crossing it ever made is corrected at once. This is what
          makes one label worth all the clips nobody will ever look at. It also keeps a visit
          whole: entries pair with exits *by name*, so renaming one half and not the other
          leaves an entry that never closes and an exit belonging to nobody -- a phantom
          occupant, manufactured by the act of labelling;
        * correcting a name the recogniser **chose from the gallery** says only that *this
          crossing* was somebody else. That person is still themselves everywhere else, and
          renaming their whole history would be a far larger claim than the one being made.

        Never raises: a label must be recorded even if the history cannot be reached, because
        the label is the thing being asked for and the figures can be recomputed later.
        """
        if self._history is None:
            return
        try:
            if whole_identity and self._merges is not None and record.name:
                merge_identity(
                    self._history, self._merges, record.name, name, LABELLED, time.time()
                )
            else:
                self._history.rename_crossing(record.timestamp, record.direction, name)
        except Exception as exc:
            _log.error("could not correct the history for %s: %s", record.sighting_id, exc)

    # --- queries ------------------------------------------------------------
    def crop_path(self, sighting_id: str) -> Path | None:
        path = self._dir / f"{sighting_id}.jpg"
        return path if path.exists() else None

    def clip_path(self, sighting_id: str) -> Path | None:
        path = self._dir / f"{sighting_id}.mp4"
        return path if path.exists() else None

    def pending(self, limit: int = 20) -> list[ReviewRecord]:
        """Unlabelled sightings that have a face to enrol, newest first."""
        candidates = [
            record
            for record in self._records.values()
            if record.labelled_as is None
            and not record.dismissed
            and record.outcome != Outcome.UNIDENTIFIED.value
            and (self._dir / f"{record.sighting_id}.npy").exists()
        ]
        candidates.sort(key=lambda record: record.timestamp, reverse=True)
        return candidates[:limit]

    def counts(self) -> dict[str, int]:
        """Enrolled reference vectors per name."""
        return self._gallery.counts()

    def group_sizes(self) -> dict[int, int]:
        """How many people crossed in each burst, so a group label can be flagged."""
        sizes: dict[int, int] = {}
        with self._lock:
            for record in self._records.values():
                if record.burst:
                    sizes[record.burst] = sizes.get(record.burst, 0) + 1
        return sizes

    def regroup(self, gap_seconds: float = 3.0) -> int:
        """Recompute which sightings were a group, from the gaps between their timestamps.

        The recorded groups were assigned by a rule that ended a burst only when every clip
        had finished, and a clip's completion counter resets whenever anybody is in view -- so
        in an occupied room every crossing joined the same group. The queue still holds groups
        of forty-four, offering forty-four names in crossing order for passages minutes apart.

        The fix to the live rule cannot repair those, so this recomputes them from the one
        thing that was recorded honestly: when each crossing happened. Returns how many
        records changed group.
        """
        with self._lock:
            in_order = sorted(self._records.values(), key=lambda record: record.timestamp)
            changed = 0
            burst = 0
            position = 0
            previous: float | None = None
            for record in in_order:
                if previous is None or record.timestamp - previous > gap_seconds:
                    # The moment the group began, matching what the pipeline now issues, so
                    # repaired groups and new ones cannot collide with each other either.
                    burst = int(record.timestamp)
                    position = 0
                previous = record.timestamp
                position += 1
                if record.burst == burst and record.position == position:
                    continue
                self._records[record.sighting_id] = ReviewRecord(
                    **{**asdict(record), "burst": burst, "position": position}
                )
                changed += 1
            self._flush()
            return changed

    def groups(self) -> dict[int, list[str]]:
        """Each burst's sightings, in crossing order.

        The order is what a group label depends on -- "a, b, c" means the first, second and
        third to cross -- so the page can show every face in that order and let somebody check
        it against the pictures rather than take it on trust.
        """
        members: dict[int, list[ReviewRecord]] = {}
        with self._lock:
            for record in self._records.values():
                if record.burst:
                    members.setdefault(record.burst, []).append(record)
        return {
            burst: [record.sighting_id for record in sorted(found, key=lambda r: r.position)]
            for burst, found in members.items()
        }

    def dismiss(self, sighting_id: str, note: str = "", name: str = "") -> LabelOutcome:
        """Mark a sighting unusable, removing any reference it contributed.

        A back-of-head or blurred capture should stop being offered *and* stop influencing
        recognition. Discarding from the gallery first is the important half: leaving the
        reference behind while hiding the card would keep degrading matches invisibly.

        ``name`` separates two things that were tangled together: *who came through* and *what
        the recogniser should learn from*. An unusable picture of a known person is still
        evidence they were there, so the crossing is recorded as them and their hours count --
        while the picture itself teaches the recogniser nothing, which is the whole point of
        rejecting it.
        """
        with self._lock:
            record = self._records.get(sighting_id)
            if record is None:
                return LabelOutcome.UNKNOWN_ID
            if record.labelled_as is not None:
                path = self._dir / f"{sighting_id}.npy"
                if path.exists():
                    self._gallery.discard(record.labelled_as, np.load(path))
                    self._gallery.save(self._gallery_dir)
            self._records[sighting_id] = ReviewRecord(
                **{
                    **asdict(record),
                    # Not a label: nothing is enrolled from a rejected picture. Who it was is
                    # recorded on the crossing instead, where the hours are counted from.
                    "labelled_as": None,
                    "dismissed": True,
                    "rejected_because": note.strip(),
                    "attributed_to": name.strip(),
                    "checked": True,
                }
            )
            self._flush()
            if name.strip():
                self._correct_history(record, name.strip())
            return LabelOutcome.DISMISSED

    def unlabel(self, sighting_id: str) -> LabelOutcome:
        """Take a label back, returning the sighting to the pending list.

        Unlike :meth:`dismiss`, the clip is still considered usable -- this is for a label
        that was simply wrong, where the right answer is to look at it again.
        """
        with self._lock:
            record = self._records.get(sighting_id)
            if record is None:
                return LabelOutcome.UNKNOWN_ID
            if record.labelled_as is None:
                return LabelOutcome.UNCHANGED
            path = self._dir / f"{sighting_id}.npy"
            if path.exists():
                self._gallery.discard(record.labelled_as, np.load(path))
                self._gallery.save(self._gallery_dir)
            self._records[sighting_id] = ReviewRecord(**{**asdict(record), "labelled_as": None})
            self._flush()
            return LabelOutcome.UNLABELLED

    def use_face(self, sighting_id: str, wanted: bool | None) -> LabelOutcome:
        """Decide whether this face is matched against: yes, no, or leave it to the rule."""
        with self._lock:
            record = self._records.get(sighting_id)
            if record is None:
                return LabelOutcome.UNKNOWN_ID
            self._records[sighting_id] = ReviewRecord(
                **{**asdict(record), "use_for_matching": wanted}
            )
            self._flush()
        self.refresh_matching()
        return LabelOutcome.CORRECTED

    def refresh_matching(self) -> None:
        """Recompute what the gallery matches against, from every enrolled face.

        Called after anything that changes the labelled set -- a label, a correction, a
        rejection, a rename, a decision about one face. The gallery keeps every face; this
        decides which of them recognition is allowed to use.
        """
        references = self.references()
        with self._lock:
            ages = {r.sighting_id: r.timestamp for r in self._records.values()}
            decided = {
                r.sighting_id: r.use_for_matching
                for r in self._records.values()
                if r.use_for_matching is not None
            }
        chosen = choose(references, ages, decided)
        vectors = {r.sighting_id: r.embedding for r in references}
        self._gallery.use_only(
            {
                name: [vectors[sighting] for sighting in picked.used if sighting in vectors]
                for name, picked in chosen.items()
            }
        )
        self._chosen = chosen

    @property
    def chosen(self) -> dict[str, object]:
        """The current selection per person, for showing why a face is or is not in use."""
        return dict(getattr(self, "_chosen", {}) or {})

    def rename(self, old: str, new: str) -> int:
        """Correct a name everywhere it was used: the gallery and every sighting labelled it.

        A misspelling is one mistake, not one per sighting. Doing it by hand means unlabelling
        and relabelling each clip, which discards and re-adds reference vectors -- more work and
        more ways to lose one. If the corrected name already exists the two merge, which is what
        "ilari" and "Ilari" being the same person means.

        Returns how many sightings were relabelled.
        """
        old, new = old.strip(), new.strip()
        if not old or not new or old == new:
            return 0
        with self._lock:
            moved = 0
            for sighting_id, record in list(self._records.items()):
                if record.labelled_as != old:
                    continue
                self._records[sighting_id] = ReviewRecord(
                    **{**asdict(record), "labelled_as": new}
                )
                moved += 1
            self._gallery.rename(old, new)
            self._gallery.save(self._gallery_dir)
            self._flush()
            return moved

    def composite_labels(self) -> list[ReviewRecord]:
        """Labels that are really several names in one string.

        These are not people. They came from a route that passed free text straight through
        as a single name, so each sits in the gallery as a person of its own with a single
        reference, competing with the real entries.
        """
        with self._lock:
            records = list(self._records.values())
        return [
            record
            for record in records
            if record.labelled_as is not None and len(parse_names(record.labelled_as)) > 1
        ]

    def references(self) -> list[Reference]:
        """Every enrolled face, tied back to the sighting it came from.

        The gallery alone cannot support an audit: it stores vectors per name with no record
        of which sighting each came from, so a suspect reference could be identified but not
        shown to anyone. The review index supplies that link.
        """
        with self._lock:
            labelled = [
                record
                for record in self._records.values()
                if record.labelled_as is not None and not record.dismissed
            ]
        found = []
        for record in labelled:
            path = self._dir / f"{record.sighting_id}.npy"
            if path.exists():
                found.append(
                    Reference(
                        sighting_id=record.sighting_id,
                        name=record.labelled_as,
                        embedding=np.load(path),
                    )
                )
        return found

    def piles(self, sort: str = "latest", limit: int = 50) -> dict[str, list[ReviewRecord]]:
        """Every sighting, in the pile that describes what has happened to it.

        The system labels *everything* it sees, using "unknown" when it recognises nobody, so
        each sighting belongs somewhere from the moment it is recorded. The unknowns are kept
        apart from the named ones in both halves: they need a different action -- a name typed
        rather than a guess confirmed -- and mixing them makes a queue that cannot be worked
        through steadily.

        ``checked`` is deliberately a superset of ``recheck`` and ``checked_unknown``: it is
        everything already dealt with, which is the pile to search when looking for a past
        sighting rather than one to work through.
        """
        far = self._distances()
        with self._lock:
            records = list(self._records.values())

        def order(found: list[ReviewRecord]) -> list[ReviewRecord]:
            if sort == "odd":
                # Furthest from that person's average first: the likeliest mistakes, rather
                # than the newest. A sighting with no reference has no distance, so it sorts
                # last -- there is nothing to be suspicious of.
                found.sort(key=lambda r: far.get(r.sighting_id, 1.0))
            else:
                found.sort(key=lambda r: r.timestamp, reverse=True)
            return found[:limit]

        rechecking = {record.sighting_id for record, _ in self.worth_rechecking(limit=limit)}
        unchecked = [r for r in records if not r.checked and not r.dismissed]
        checked = [r for r in records if r.checked]
        return {
            "unnamed": self.unnamed(records, limit=limit),
            "unchecked_unknown": order([r for r in unchecked if _is_unknown(r)]),
            "unchecked_named": order([r for r in unchecked if not _is_unknown(r)]),
            "recheck": order([r for r in checked if r.sighting_id in rechecking]),
            "checked_unknown": order([r for r in checked if _is_unknown(r)]),
            "checked": order(list(checked)),
        }

    def unnamed(self, records: list[ReviewRecord] | None = None, limit: int = 50) -> list:
        """One card per identity the system named itself, best face first.

        Deliberately grouped by *person*, not by sighting. A stranger who comes through ten
        times leaves ten sightings, and offering all ten is offering the same question ten
        times: the answer to any one of them names the identity, and the other nine vanish.
        One clip is enough to say who somebody is.

        The card offered is their clearest face -- the highest-scoring, undismissed sighting
        -- because that is the one most likely to be recognisable, and a name given from a
        poor frame is the mistake this pile exists to prevent.
        """
        if self._provisional is None:
            return []
        if records is None:
            with self._lock:
                records = list(self._records.values())
        wanted = self._provisional.names(known=self._gallery.names)

        best: dict[str, ReviewRecord] = {}
        for record in records:
            name = record.labelled_as or record.name
            if name not in wanted or record.dismissed:
                continue
            held = best.get(name)
            if held is None or record.score > held.score:
                best[name] = record
        # Longest-unnamed first: somebody who has been coming through for a week without a
        # name is a worse gap in the figures than somebody who arrived a minute ago.
        return sorted(best.values(), key=lambda r: r.timestamp)[:limit]

    def _distances(self) -> dict[str, float]:
        """How much each enrolled face is unlike the rest of that person's, by sighting id."""
        return {
            suspect.sighting_id: suspect.similarity for suspect in self.audit().suspects
        }

    def everything(self) -> list[ReviewRecord]:
        """Every record, for a view that needs one person's whole history rather than a pile."""
        with self._lock:
            return list(self._records.values())

    def recent_names(self, limit: int = 20) -> list[str]:
        """Names in the order they were last used, most recent first.

        The next person through a door is very often somebody who came through recently, so
        that order puts the likely answer at the top of the list instead of alphabetically
        somewhere in the middle.
        """
        with self._lock:
            labelled = [r for r in self._records.values() if r.labelled_as]
        labelled.sort(key=lambda record: record.timestamp, reverse=True)
        seen: list[str] = []
        for record in labelled:
            if record.labelled_as not in seen:
                seen.append(record.labelled_as)
        return seen[:limit]

    def worth_rechecking(self, limit: int = 50) -> list[tuple[ReviewRecord, str]]:
        """Labels that deserve a second look, each with the reason, newest first.

        Two kinds, and both are about a label being *wrong* rather than missing:

        * a face sitting far from the rest of that person's -- the audit's judgement;
        * a name used exactly once, which is what a typo looks like. A real person accumulates
          sightings; "ilar" appears once and never again.

        Anything a person has already saved is left out for good. They have judged it, and
        putting it back because the numbers still look odd would be arguing with them.
        """
        suspects = {suspect.sighting_id: suspect.reason for suspect in self.audit().suspects}
        counts = self.counts()
        found: list[tuple[ReviewRecord, str]] = []
        with self._lock:
            records = list(self._records.values())
        for record in records:
            if record.labelled_as is None or record.checked or record.dismissed:
                continue
            if record.sighting_id in suspects:
                found.append((record, suspects[record.sighting_id]))
            elif counts.get(record.labelled_as, 0) == 1:
                found.append((record, f'"{record.labelled_as}" is used only once'))
        found.sort(key=lambda pair: pair[0].timestamp, reverse=True)
        return found[:limit]

    def audit(self) -> Audit:
        """Which enrolled faces look wrong, and who has too few examples."""
        return audit(self.references())

    @property
    def directory(self) -> Path:
        """Where this deployment's review data lives, for things kept beside it."""
        return self._dir

    def nearest(self, at: float, direction: str, window: float = 2.0) -> ReviewRecord | None:
        """The sighting for a crossing at this moment, so a clip can be shown for it.

        Matched on time and direction: the crossing and the sighting were written by different
        parts of the system and share only those.
        """
        with self._lock:
            candidates = [
                record
                for record in self._records.values()
                if record.direction == direction and abs(record.timestamp - at) <= window
            ]
        if not candidates:
            return None
        return min(candidates, key=lambda record: abs(record.timestamp - at))

    def get(self, sighting_id: str) -> ReviewRecord | None:
        with self._lock:
            return self._records.get(sighting_id)

    # --- persistence --------------------------------------------------------
    def _next_id(self, timestamp: float) -> str:
        base = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d_%H-%M-%S")
        if base not in self._records:
            return base
        suffix = 2
        while f"{base}-{suffix}" in self._records:
            suffix += 1
        return f"{base}-{suffix}"

    def _load(self) -> None:
        path = self._dir / _INDEX
        if not path.exists():
            return
        raw = json.loads(path.read_text(encoding="utf-8"))
        for entry in raw:
            record = ReviewRecord(**entry)
            self._records[record.sighting_id] = record

    def _flush(self) -> None:
        payload = [asdict(record) for record in self._records.values()]
        (self._dir / _INDEX).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _is_unknown(record: ReviewRecord) -> bool:
    """Whether this sighting stands as "unknown" -- nobody named it, or somebody said so.

    Both halves matter. The system labels everything, using "unknown" when it recognises
    nobody, and a person may deliberately save "unknown" for somebody who is not staff. The
    two look the same here on purpose: what they share is that no name is attached, which is
    what decides which pile it belongs in.
    """
    return (record.labelled_as or record.name or "unknown").lower() == "unknown"


def _is_nobody(name: str) -> bool:
    """Whether a typed name means "nobody in particular" rather than a person."""
    return name.strip().lower() in {"unknown", "unnamed", "nobody"}
