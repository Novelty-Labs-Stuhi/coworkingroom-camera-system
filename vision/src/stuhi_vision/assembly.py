"""Composition root: build the wired-up application from a config.

This is the one place that knows how every concrete component fits together, so the
modules themselves stay free of construction logic and the CLI stays thin.
"""

from __future__ import annotations

from dataclasses import dataclass

from .clips import ClipRecorder
from .config import Config
from .domain import Sighting
from .doorway import DoorwayMonitor
from .gating import MotionGate
from .handlers import Doorkeeper
from .ledger import Ledger
from .notify import TelegramNotifier
from .pipeline import FrameObserver, Pipeline
from .publishing import Publication, SightingPublisher
from .recognition.body import BodyEmbedder
from .recognition.face import FaceRecognizer
from .recognition.gallery import FaceGallery
from .region import Region
from .review import ReviewQueue
from .sessions import SessionManager
from .sources import open_source
from .sources.buffered import BufferedSource
from .store import EventStore
from .tracking import GatedTracker, PersonTracker
from .visualization import encode_jpeg
from .web import WebUI


@dataclass(slots=True)
class Application:
    pipeline: Pipeline
    store: EventStore
    review: ReviewQueue
    sessions: SessionManager
    publisher: SightingPublisher
    notifier: TelegramNotifier | None = None
    web: WebUI | None = None

    def close(self) -> None:
        # Anything still waiting for its clip to finish must go out, or a crossing right
        # before shutdown would be silently dropped.
        self.publisher.flush()
        if self.web is not None:
            self.web.stop()
        if self.notifier is not None:
            self.notifier.stop()
        self.sessions.close()
        self.store.close()


def build(config: Config, announce, observer: FrameObserver | None = None) -> Application:
    """Assemble every component described by ``config`` into a ready pipeline.

    ``observer`` is an optional per-frame hook (e.g. an Annotator) for offline review.
    """
    thresholds = config.thresholds
    performance = config.performance
    gallery = FaceGallery.load(config.paths.gallery_dir)
    faces = FaceRecognizer(gallery, thresholds.face_match)
    bodies = BodyEmbedder()
    store = EventStore(config.paths.database)
    ledger = Ledger(store, thresholds.exit_similarity, thresholds.exit_margin)
    sessions = SessionManager(
        faces,
        bodies,
        thresholds.face_match,
        thresholds.face_margin,
        face_workers=performance.face_workers,
    )
    doorkeeper = Doorkeeper(sessions, ledger, thresholds.min_track_age)
    review = ReviewQueue(config.paths.review_dir, gallery, config.paths.gallery_dir)

    notifier: TelegramNotifier | None = None
    if config.telegram.enabled:
        notifier = TelegramNotifier(config.telegram.bot_token, config.telegram.chat_id, review)
        notifier.start()

    # Both labelling routes share this one queue and gallery, so a label from either takes
    # effect on the next frame and the same sighting can never be counted twice.
    web: WebUI | None = None
    if config.web.enabled:
        web = WebUI(review, host=config.web.host, port=config.web.port)
        web.start()

    # Tapped at the source, so the clip covers the *approach* to a crossing, not just the
    # frames that happened to follow the commit.
    recorder = ClipRecorder(
        encode_jpeg,
        capacity=performance.clip_frames,
        max_clip_frames=performance.clip_max_frames,
    )
    source = _recorded(
        BufferedSource(open_source(config.source), performance.buffer_capacity), recorder
    )
    publisher = SightingPublisher(
        recorder,
        _publisher(review, notifier, announce),
        clear_frames=performance.clip_clear_frames,
    )

    pipeline = Pipeline(
        source=source,
        tracker=GatedTracker(
            PersonTracker(
                model_path=performance.detect_model,
                detection_conf=thresholds.detection_conf,
                imgsz=performance.detect_imgsz,
            ),
            gate=MotionGate(min_fraction=performance.motion_min_fraction),
            region_builder=_region_builder(config),
        ),
        doorway=DoorwayMonitor(config.doorway),
        sessions=sessions,
        doorkeeper=doorkeeper,
        announce=_directional(config.doorway.announce, publisher, announce),
        on_frame=_frame_hook(publisher, observer),
    )
    return Application(
        pipeline=pipeline,
        store=store,
        review=review,
        sessions=sessions,
        publisher=publisher,
        notifier=notifier,
        web=web,
    )


def _region_builder(config: Config):
    """Crop detection to the doorway, unless padding is zero (whole frame)."""
    padding = config.performance.crop_padding
    if padding <= 0:
        return None

    def build_region(width: int, height: int) -> Region:
        return Region.around(config.doorway.line_a, config.doorway.line_b, width, height, padding)

    return build_region


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


def _frame_hook(publisher: SightingPublisher, observer: FrameObserver | None):
    """Drive the publisher every frame, then pass the frame to any caller's observer."""

    def on_frame(frame, people, crossings) -> None:
        publisher.advance(people_present=bool(people))
        if observer is not None:
            observer(frame, people, crossings)

    return on_frame


def _publisher(review: ReviewQueue, notifier: TelegramNotifier | None, announce):
    """File a completed sighting with its clip, notify the chat, then hand it on."""

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
        if notifier is not None:
            notifier.announce(
                publication.sighting, sighting_id, publication.position, publication.total
            )
        announce(publication.sighting)

    return publish
