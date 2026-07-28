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
from .publishing import SightingPublisher
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


@dataclass(slots=True)
class Application:
    pipeline: Pipeline
    store: EventStore
    review: ReviewQueue
    sessions: SessionManager
    publisher: SightingPublisher
    notifier: TelegramNotifier | None = None

    def close(self) -> None:
        # Anything still waiting for its clip to finish must go out, or a crossing right
        # before shutdown would be silently dropped.
        self.publisher.flush()
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
        announce=publisher.hold,
        on_frame=_frame_hook(publisher, observer),
    )
    return Application(
        pipeline=pipeline,
        store=store,
        review=review,
        sessions=sessions,
        publisher=publisher,
        notifier=notifier,
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


def _frame_hook(publisher: SightingPublisher, observer: FrameObserver | None):
    """Drive the publisher every frame, then pass the frame to any caller's observer."""

    def on_frame(frame, people, crossings) -> None:
        publisher.advance(people_present=bool(people))
        if observer is not None:
            observer(frame, people, crossings)

    return on_frame


def _publisher(review: ReviewQueue, notifier: TelegramNotifier | None, announce):
    """File a completed sighting with its clip, notify the chat, then hand it on."""

    def publish(sighting: Sighting, clip: bytes | None, position: int, total: int) -> None:
        sighting_id = review.record(sighting, encode_jpeg=encode_jpeg, clip=clip)
        if notifier is not None:
            notifier.announce(sighting, sighting_id, position, total)
        announce(sighting)

    return publish
