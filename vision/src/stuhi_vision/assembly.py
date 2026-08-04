"""Composition root: build the wired-up application from a config.

This is the one place that knows how every concrete component fits together, so the
modules themselves stay free of construction logic and the CLI stays thin.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

from .alignment import DriftWatch
from .attention import Attention
from .clips import ClipRecorder
from .config import CameraConfig, Config
from .continuity import RESET, DayBoundary, end_of_day, open_visits, partition, start_of_day
from .corroboration import Corroboration
from .domain import Sighting
from .doorway import DoorwayMonitor
from .gating import MotionGate
from .handlers import Doorkeeper, Enrolment, Identifier
from .heartbeat import Heartbeat
from .latest import LatestFrames
from .ledger import Ledger
from .merges import MergeLog
from .notify import TelegramNotifier
from .occlusion import Occlusion
from .ordering import OrderingRule
from .passage import PassageMonitor
from .pipeline import FrameObserver, Hooks, Pipeline
from .presence import office_day
from .provisional import Provisional
from .publishing import Publication, SightingPublisher
from .recognition.body import BodyEmbedder
from .recognition.face import FaceRecognizer
from .recognition.gallery import FaceGallery
from .region import Region
from .review import ReviewQueue
from .sessions import SessionManager
from .sources import open_source
from .sources.buffered import BufferedSource
from .store import EventStore, PassageStore
from .strangers import Strangers
from .threshold import ThresholdConfig, ThresholdMonitor
from .tracking import GatedTracker, PersonTracker
from .visualization import encode_jpeg
from .web import WebUI
from .web.app import create_app
from .witness import LeavingWitness
from .zones import ZoneStore

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _Shared:
    """The state every camera works through, rather than owning a copy of.

    A face enrolled from one camera must be recognised by the other, and occupancy is one
    fact about one room -- so these exist exactly once and each is internally locked.
    """

    gallery: FaceGallery
    # Where the gallery is written. Needed because a new identity is enrolled the moment an
    # unrecognised person walks in, and an enrolment that is not saved is lost on restart.
    gallery_dir: Path
    ledger: Ledger
    review: ReviewQueue
    notifier: TelegramNotifier | None
    frames: LatestFrames
    zones: ZoneStore
    # Names the identifying camera has seen leaving, waiting for the counting camera to
    # attach one to an exit. Shared because it is a message from one camera to the other.
    witness: LeavingWitness
    # Every episode of a doorframe being covered, refusals included. The events table holds
    # only what was committed, so "why are there fewer exits than entries" had no evidence
    # behind it: the log that held the refusals is rotated within hours.
    passages: PassageStore
    # Empties the room at the office-day boundary. Shared, because there is one room: two
    # cameras asking is two chances to notice the day turned, and the second one no-ops.
    boundary: DayBoundary
    # Where each camera stamps that it is still handling frames. One file per camera: a
    # shared stamp keeps advancing while one camera is dead, and the dead one may be the
    # camera that does the counting.
    heartbeats: Path
    # Identities the system named itself, awaiting a human's. The one pile worth interrupting
    # somebody for, and so the only thing that reaches the chat.
    provisional: Provisional
    # Their faces, kept apart from the named gallery so recognising a returning stranger
    # cannot degrade recognition of somebody with a name.
    strangers: Strangers
    # What the room camera saw, waiting for the doorway camera to ask about it: somebody
    # appearing in the room confirms an entry, somebody walking at the lens and out of frame
    # confirms an exit. Shared, because it is a message from one camera to the other.
    corroboration: Corroboration


@dataclass(slots=True)
class Camera:
    """One camera's own machinery. Nothing here is shared with another camera.

    Each camera tracks independently -- separate ByteTrack state, separate track ids,
    separate sessions -- because a track id means nothing across two views and reconciling
    them is the hard problem that giving each camera one direction avoids entirely.
    """

    name: str
    pipeline: Pipeline
    sessions: SessionManager
    publisher: SightingPublisher
    drift: DriftWatch

    def close(self) -> None:
        # Anything still waiting for its clip must go out, or a crossing just before
        # shutdown would be dropped silently.
        self.publisher.flush()
        self.sessions.close()


@dataclass(slots=True)
class Application:
    """Every camera, over the one gallery, ledger, review queue and store they share."""

    cameras: list[Camera]
    store: EventStore
    passages: PassageStore
    review: ReviewQueue
    ledger: Ledger
    frames: LatestFrames
    zones: ZoneStore
    notifier: TelegramNotifier | None = None
    web: WebUI | None = None

    def run(self) -> None:
        """Run every camera until they stop. One thread each; the last one blocks here."""
        if len(self.cameras) == 1:
            self.cameras[0].pipeline.run()
            return
        threads = [
            threading.Thread(target=camera.pipeline.run, name=f"pipeline-{camera.name}")
            for camera in self.cameras
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    def close(self) -> None:
        for camera in self.cameras:
            camera.close()
        if self.web is not None:
            self.web.stop()
        if self.notifier is not None:
            self.notifier.stop()
        self.store.close()
        self.passages.close()


def _tick(boundary: DayBoundary, heartbeat: Heartbeat):
    """What the frame loop does with no frame to show for it: stamp, then check the clock.

    Wall time, not the frame's timestamp -- a file source numbers its frames from zero, which
    would put every replay in 1970 and fire the day boundary on the second frame.

    The stamp goes first. If closing out the room ever throws, the evidence that this camera
    was alive up to that moment is already on disk.
    """

    def beat() -> None:
        heartbeat.beat()
        boundary.check(time.time())

    return beat


def _resume(ledger: Ledger, store: EventStore) -> DayBoundary:
    """Rebuild the room from the record, and arm the boundary that will empty it tonight.

    Called once, before any camera runs. Anything recorded as inside since the office day
    began is taken back so its exit can still be attributed; anything older is closed out
    now, because nobody slept here.
    """
    now = time.time()
    inside = open_visits((at, name, direction) for name, at, direction in store.passages())
    resume, stale = partition(inside, start_of_day(now))

    ledger.restore(resume)
    ledger.restore(stale)
    for name, since in sorted(stale.items(), key=lambda item: item[1]):
        # Closed at the boundary of the day it began, not now: dating it today would credit
        # the visit with every hour the system was not watching.
        ledger.close([name], end_of_day(since), RESET)
    if resume or stale:
        _log.info(
            "resumed with %d occupant(s); closed %d visit(s) older than today",
            len(resume),
            len(stale),
        )
    # Seeded with today, so a boundary that has already passed is not fired again on boot.
    return DayBoundary(ledger, day=office_day(now).toordinal())


def build(config: Config, announce, observer: FrameObserver | None = None) -> Application:
    """Assemble every component described by ``config`` into a ready pipeline.

    ``observer`` is an optional per-frame hook (e.g. an Annotator) for offline review.
    """
    thresholds = config.thresholds

    gallery = FaceGallery.load(config.paths.gallery_dir)
    store = EventStore(config.paths.database)
    ledger = Ledger(
        store,
        thresholds.exit_similarity,
        thresholds.exit_margin,
        # Exits are matched against the people inside, which is a far smaller field than the
        # door faces: a weak face can still settle it. Same gallery, so a label takes effect
        # here on the next exit too.
        faces=gallery.rank,
        face_similarity=thresholds.exit_face_match,
    )
    # Take back whoever the record says is still inside, and close whatever is older than
    # today. Both halves matter and neither is safe alone: without the restore a restart
    # orphans every occupant permanently, and without the closing a restart carries
    # yesterday's occupants into today untouched.
    boundary = _resume(ledger, store)
    provisional = Provisional(config.paths.gallery_dir.parent / "provisional")
    # Faces of people nobody has named, kept in their own store so an uncertain face cannot
    # reach the references a real person is recognised by. The count depends on a returning
    # stranger being matched to the identity they already have; this is what allows that
    # without letting their face compete against Ilari's.
    strangers = Strangers(config.paths.gallery_dir.parent / "strangers")
    # Written before any identity is folded into another, so a mistaken merge can be undone
    # rather than having fused two people's histories for good.
    merges = MergeLog(config.paths.review_dir.parent / "merges.jsonl")

    review = ReviewQueue(
        config.paths.review_dir,
        gallery,
        config.paths.gallery_dir,
        # So a corrected label reaches the crossing itself, which is what the time-in-the-room
        # figures are derived from.
        history=store,
        # So the page can offer one card per person nobody has named, rather than one per
        # sighting of them.
        provisional=provisional,
        # Naming an identity the system invented corrects every crossing it ever made, not
        # just the one on screen -- and this log is what makes that reversible.
        merges=merges,
    )

    notifier: TelegramNotifier | None = None
    if config.telegram.enabled:
        notifier = TelegramNotifier(config.telegram.bot_token, config.telegram.chat_id, review)
        notifier.start()

    # Both labelling routes share this one queue and gallery, so a label from either takes
    # effect on the next frame and the same sighting can never be counted twice.
    web: WebUI | None = None

    frames = LatestFrames(encode_jpeg)
    zones = ZoneStore(config.paths.review_dir.parent / "zones")
    passages = PassageStore(config.paths.database)
    shared = _Shared(
        gallery=gallery,
        gallery_dir=config.paths.gallery_dir,
        ledger=ledger,
        review=review,
        notifier=notifier,
        frames=frames,
        zones=zones,
        witness=LeavingWitness(),
        passages=passages,
        boundary=boundary,
        heartbeats=config.paths.review_dir.parent / "heartbeat",
        provisional=provisional,
        strangers=strangers,
        corroboration=Corroboration(),
    )
    cameras = [_build_camera(entry, config, shared, announce, observer) for entry in config.cameras]
    # The UI is started last: it serves frames and drift readings that only exist once the
    # cameras have been built.
    if config.web.enabled:
        web = WebUI(
            create_app(
                review,
                frames=frames,
                zones=zones,
                drift={camera.name: camera.drift for camera in cameras},
                # So a corrected spelling reaches the history and whoever is inside now, not
                # only the gallery: two spellings read as two half-present people.
                history=store,
                ledger=ledger,
                # The refused passages: the evidence for an exit that was never counted.
                passages=passages,
            ),
            host=config.web.host,
            port=config.web.port,
        )
        web.start()

    return Application(
        cameras=cameras,
        store=store,
        passages=passages,
        review=review,
        ledger=ledger,
        frames=frames,
        zones=zones,
        notifier=notifier,
        web=web,
    )


def _build_camera(
    entry: CameraConfig,
    config: Config,
    shared: _Shared,
    announce,
    observer: FrameObserver | None,
) -> Camera:
    """Everything one camera needs, wired to the shared gallery and ledger."""
    thresholds = config.thresholds
    performance = config.performance

    sessions = SessionManager(
        FaceRecognizer(shared.gallery, thresholds.face_match),
        BodyEmbedder(),
        thresholds.face_match,
        thresholds.face_margin,
        face_workers=performance.face_workers,
    )
    committer = _committer(entry, sessions, shared, _persistence(entry, thresholds))
    drift = DriftWatch(shared.zones.reference_path(entry.name))

    # Tapped at the source, so the clip covers the *approach* to a crossing, not just the
    # frames that happened to follow the commit.
    recorder = ClipRecorder(
        encode_jpeg,
        capacity=performance.clip_frames,
        max_clip_frames=performance.clip_max_frames,
    )
    source = _source(open_source(entry.source), entry.name, performance, shared, recorder)
    publisher = SightingPublisher(
        recorder,
        _publisher(shared.review, shared.notifier, announce),
        clear_frames=performance.clip_clear_frames,
    )

    monitor, attention, doorframe = _monitor(
        entry, shared.zones, shared.passages, shared.corroboration
    )
    _follow_zone(entry, shared.zones, monitor, attention, doorframe)
    pipeline = Pipeline(
        source=source,
        tracker=GatedTracker(
            PersonTracker(
                model_path=performance.detect_model,
                detection_conf=thresholds.detection_conf,
                imgsz=performance.detect_imgsz,
            ),
            gate=MotionGate(min_fraction=_motion(entry, performance)),
            region_builder=_region_builder(entry, performance.crop_padding, shared.zones),
        ),
        doorway=monitor,
        sessions=sessions,
        doorkeeper=committer,
        hooks=Hooks(
            announce=_directional(entry.announce, publisher, announce),
            on_frame=_frame_hook(
                publisher, observer, entry.name, shared, drift,
                shadow=_shadow(entry.name, entry, attention),
            ),
            tick=_tick(shared.boundary, Heartbeat(shared.heartbeats / entry.name)),
            doorframe=doorframe.update if doorframe is not None else None,
        ),
        attention=attention,
    )
    return Camera(
        name=entry.name,
        pipeline=pipeline,
        sessions=sessions,
        publisher=publisher,
        drift=drift,
    )


def _committer(entry: CameraConfig, sessions: SessionManager, shared: _Shared, min_age: int):
    """What this camera does with a passage it has recognised: count it, or just name it.

    Both cameras see every passage, so only one may write to the ledger. The doorway camera
    counts, since the doorframe is what tells a passage from background traffic; the room
    camera identifies, since it sees a leaver's face where the other sees the back of a head.
    """
    if entry.role == "identify":
        return Identifier(sessions, shared.witness, min_age, camera=entry.name)
    return Doorkeeper(
        sessions,
        shared.ledger,
        min_age,
        camera=entry.name,
        witness=shared.witness,
        # So an unrecognised arrival's face is enrolled under their new identity, the same
        # person coming back is matched to it rather than becoming somebody else again, and
        # the labelling page can tell an invented name from one a human chose.
        enrolment=Enrolment(shared.strangers, shared.gallery_dir, shared.provisional),
        # The room camera's second opinion on each crossing this one counts.
        corroboration=getattr(shared, "corroboration", None),
    )


def _follow_zone(
    entry: CameraConfig, zones: ZoneStore, monitor, attention, doorframe=None
) -> None:
    """Apply a redrawn zone to this running camera, rather than waiting for a restart.

    A zone is redrawn because the camera moved, so the count is wrong *now*. Removing one
    falls back to whatever the config says, which is the same path as never having drawn one.
    """
    if not isinstance(entry.detector, ThresholdConfig):
        return
    from_config = entry.detector.zone

    def apply(camera: str, drawn) -> None:
        if camera != entry.name:
            return
        zone = drawn.as_tuple() if drawn is not None else from_config
        monitor.use_zone(zone)
        if attention is not None:
            attention.use_zone(zone)
        # The pixels a "preceded" camera watches have to move with the box too, or it would go
        # on reporting the old doorframe as covered while the rule judged against the new one.
        if doorframe is not None:
            doorframe.use_zone(zone)
        _log.info("%s now judging the box %s", entry.name, [round(v, 3) for v in zone])

    zones.watch(apply)


def _persistence(entry: CameraConfig, thresholds) -> int:
    """Frames a track must have been seen for before its crossing counts, for this camera."""
    if entry.min_track_age is None:
        return thresholds.min_track_age
    return entry.min_track_age


def _motion(entry: CameraConfig, performance) -> float:
    """How much of this camera's frame must change before the detector runs.

    The right value belongs to the view, not the deployment: the dark corridor needs zero,
    because there the gate suppressed the very frames that held a person, while the lit room
    can sleep through most of a day. A camera that says nothing takes the shared default.
    """
    if entry.motion_min_fraction is None:
        return performance.motion_min_fraction
    return entry.motion_min_fraction


def _testify(entry: CameraConfig, corroboration: Corroboration | None, observation) -> None:
    """File what the room camera saw, for the doorway camera to ask about later.

    Only the identifying camera testifies: the doorway camera is the one being corroborated, and
    a witness that is also the accused is no witness. Two things are worth filing, and they are
    the two halves the doorway camera cannot see:

    * **somebody appeared** in the room's view at all -- which is what walking in looks like
      from in here, whatever they did afterwards;
    * **somebody walked at the lens and out of frame** -- growing, then gone -- which is what
      walking out looks like. Growth alone is somebody leaning towards a desk; leaving the frame
      alone is somebody stepping sideways past the edge. Together they are a departure.
    """
    if corroboration is None or entry.role != "identify":
        return
    corroboration.appeared(at=observation.began_at)
    grew = getattr(observation, "grew", 0.0)
    if getattr(observation, "left_frame", False) and grew > 0:
        corroboration.left_frame(at=observation.ended_at, grew=grew)


def _monitor(
    entry: CameraConfig,
    zones: ZoneStore,
    passages: PassageStore | None = None,
    corroboration: Corroboration | None = None,
):
    """The passage detector this camera configured, and the attention it needs, if any.

    A zone drawn in the UI wins over the one in the config: it was drawn by somebody looking
    at the actual view, which the config's numbers can only approximate.
    """
    if not isinstance(entry.detector, ThresholdConfig):
        return DoorwayMonitor(entry.detector)
    drawn = zones.get(entry.name)
    detector = entry.detector
    if drawn is not None:
        detector = replace(detector, zone=drawn.as_tuple())
    # Said out loud at startup, because a drawn zone overrides the config silently and nothing
    # in config.toml can tell you which is in force. An afternoon was spent measuring the wrong
    # box on 2026-08-03 for exactly that reason: the file said 0.30, a drawn zone said 0.152,
    # and the only trace of it was one line written whenever somebody redrew it.
    _log.info(
        "%s judging the box %s (%s)",
        entry.name,
        [round(value, 3) for value in detector.zone],
        "drawn in the UI, overriding the config" if drawn is not None else "from the config",
    )

    def report(observation) -> None:
        _log.info("%s %s", entry.name, observation.readable)
        _testify(entry, corroboration, observation)
        # Filed as well as logged: the log is rotated within hours, and a refused passage is
        # exactly the evidence needed to explain a count that looks wrong.
        if passages is not None and hasattr(observation, "coverage"):
            passages.record(time.time(), entry.name, observation)

    if entry.rule == "coverage":
        # Pixel change per vertical slice of the box decides the passage and its direction;
        # the tracker only has to confirm a person was on it. Attention runs those pixels
        # once a frame, and uses them to decide when the detector is worth waking -- so the
        # two come as a pair. See docs/design.md.
        attention = Attention(Occlusion(zone=detector.zone, config=entry.coverage))
        return PassageMonitor(detector, attention.episode, watcher=report), attention, None

    if getattr(detector, "discriminator", None) == "preceded":
        # This rule needs two things at once: the doorframe's state on each frame, and frames
        # from *before* it was covered. Attention supplies both -- it reads the box every frame
        # and replays the approach out of its buffer -- and, crucially, it also keeps the
        # detector asleep the rest of the time. Running it on every frame instead costs 200 ms
        # a frame against a camera delivering eighteen, which starves the frame reader and gets
        # the pipeline killed as stuck. That was measured, not guessed.
        attention = Attention(Occlusion(zone=detector.zone, config=entry.coverage))
        monitor = ThresholdMonitor(
            detector, report=report, covered=lambda: attention.busy
        )
        return monitor, attention, None

    return ThresholdMonitor(detector, report=report), None, None


def _region_builder(entry: CameraConfig, padding: float, zones: ZoneStore):
    """Where this camera looks: the drawn zone if it has one, else around the doorway line.

    A zone on a *counting* camera marks the doorframe, and that rule needs the whole frame --
    it is about where somebody goes after covering the box, so cropping would remove the
    evidence. A zone on an *identifying* camera means the opposite: only look here. Its view
    is a wide room full of people at desks, and everything outside the doorway is a
    distraction the detector pays for on every frame and can mistake for somebody arriving.

    Read per frame, so redrawing a zone takes effect without a restart, like everything else
    that zone touches.
    """
    if entry.role == "identify":

        def look_at_the_zone(width: int, height: int) -> Region | None:
            drawn = zones.get(entry.name)
            return None if drawn is None else Region.of(drawn.as_tuple(), width, height)

        return look_at_the_zone

    doorway = entry.doorway
    if padding <= 0 or doorway is None:
        return None

    def build_region(width: int, height: int) -> Region:
        return Region.around(doorway.line_a, doorway.line_b, width, height, padding)

    return build_region


def _source(frames, camera: str, performance, shared: _Shared, recorder: ClipRecorder):
    """Buffer the camera, keep every frame for a clip, and hand the newest one to the UI.

    The UI is fed from the *reader*, as each frame arrives. Taking it from the far end of the
    pipeline instead showed a view up to twenty seconds old -- the reader banks hundreds of
    frames so a slow machine loses nothing -- which is no use for drawing a zone or for seeing
    what a camera can see right now.
    """
    buffered = BufferedSource(
        frames,
        performance.buffer_capacity,
        name=camera,
        on_read=lambda frame: shared.frames.put(camera, frame.image),
    )
    return _recorded(buffered, recorder)


def _recorded(source, recorder: ClipRecorder):
    """Pass frames through, remembering each one so a clip can be cut from the window."""

    def frames():
        for frame in source:
            recorder.add(frame.image)
            yield frame

    return frames()


def _directional(reported: str, publisher: SightingPublisher, announce):
    """Film and announce only the crossings this camera sees faces for.

    The filter belongs *here*, before the clip is opened -- not downstream of publishing.
    Every passage is seen by both cameras, so filtering after the fact would still cut,
    encode and send a second video showing the back of someone's head.
    """
    if reported == "both":
        return publisher.hold

    def hold(sighting: Sighting) -> None:
        if sighting.direction.value == reported:
            publisher.hold(sighting)
        else:
            announce(sighting)  # counted and logged, but not filmed or sent

    return hold


def _shadow(camera: str, entry: CameraConfig, attention: Attention | None):
    """Run the ordering rule beside the live one, reporting only. Never commits.

    It resolves five of the six tracks in the recorded footage where the live rule resolves
    three, and produces both directions where the live rule produces one -- but six tracks is
    six tracks, and their ground truth was read by eye from the same ordering this rule uses,
    so the agreement is not independent evidence. Running it here costs a dictionary update a
    frame and settles the question against real traffic in a day.
    """
    if attention is None or not isinstance(entry.detector, ThresholdConfig):
        return None
    # Inherits the live rule's `passing_means` rather than flipping it. Flipping was tried and
    # was wrong: it rested on one montage read by eye, and on live traffic the flipped rule
    # disagreed with the live one on every single track both resolved -- four out of four,
    # opposite each time. Unflipped they agree, which is what makes the comparison worth
    # keeping. Whether the live rule's own sign is right is a separate question that inference
    # has now failed twice; it needs one deliberate walk in and one walk out to settle.
    rule = OrderingRule(entry.detector)

    def watch(frame, people) -> None:
        height, width = frame.image.shape[:2]
        # Non-None on exactly the frame an episode finishes, so no de-duplication is needed
        # and none is kept -- a set of every span a camera ever saw is a slow leak.
        finished = attention.episode()
        episode = rule.span_of(finished.frames) if finished is not None else None
        for verdict in rule.observe(people, width, height, episode):
            _log.info("%s [shadow ordering] %s", camera, verdict.readable)

    return watch


def _frame_hook(
    publisher: SightingPublisher,
    observer: FrameObserver | None,
    camera: str,
    shared: _Shared,
    drift: DriftWatch,
    shadow=None,
):
    """Per frame: advance the publisher, keep the frame, and watch for a moved camera."""

    def on_frame(frame, people, crossings) -> None:
        publisher.advance(people_present=bool(people))
        if shadow is not None:
            # Reporting only, and never allowed to take the pipeline down with it: a rule on
            # trial must not be able to stop the one doing the work.
            try:
                shadow(frame, people)
            except Exception:
                _log.exception("%s shadow ordering rule failed", camera)
        # Only empty frames: a person is a large moving object and would drag the
        # correlation with them, reading as a camera that had moved.
        if not people:
            drift.check(frame.image)
            if drift.has_moved:
                _warn_moved(camera, drift, shared)
        if observer is not None:
            observer(frame, people, crossings)

    return on_frame


def _warn_moved(camera: str, drift: DriftWatch, shared: _Shared) -> None:
    """Say once that a camera has moved, then stop until somebody redraws its zone."""
    reading = drift.latest.readable if drift.latest else "unknown"
    message = (
        f"{camera} appears to have moved ({reading}). Its zone was drawn on the old view, "
        f"so passages may be miscounted until it is redrawn."
    )
    _log.warning(message)
    if shared.notifier is not None:
        shared.notifier.send_note(message)
    drift.acknowledge()


def _publisher(review: ReviewQueue, notifier: TelegramNotifier | None, announce):
    """File a completed sighting with its clip, and tell the chat if it needs a name."""

    def publish(publication: Publication) -> None:
        # position and burst have to reach the record, or a group cannot be labelled by
        # crossing order later -- the review queue would see each sighting as standalone.
        sighting_id = review.record(
            publication.sighting,
            encode_jpeg=encode_jpeg,
            clip=publication.clip,
            position=publication.position,
            burst=publication.burst,
        )
        # Only somebody nobody has named yet is worth a message. Every crossing used to be
        # sent, which made the chat a feed rather than a queue -- and a feed of things needing
        # no action is one nobody reads, so the sightings that *did* need naming were lost in
        # it. `introduced` is true exactly once per person, on the crossing that invented
        # their identity, so a regular's daily arrival is filed silently.
        if notifier is not None and publication.sighting.introduced:
            notifier.announce(
                publication.sighting, sighting_id, publication.position, publication.total
            )
        announce(publication.sighting)

    return publish
