"""Composition root: build the wired-up application from a config.

This is the one place that knows how every concrete component fits together, so the
modules themselves stay free of construction logic and the CLI stays thin.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .domain import Sighting
from .doorway import DoorwayMonitor
from .gating import MotionGate
from .handlers import Doorkeeper
from .ledger import Ledger
from .notify import TelegramNotifier
from .pipeline import FrameObserver, Pipeline
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
    notifier: TelegramNotifier | None = None

    def close(self) -> None:
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

    pipeline = Pipeline(
        source=BufferedSource(open_source(config.source), performance.buffer_capacity),
        tracker=GatedTracker(
            PersonTracker(
                detection_conf=thresholds.detection_conf,
                imgsz=performance.detect_imgsz,
            ),
            gate=MotionGate(min_fraction=performance.motion_min_fraction),
            region_builder=_region_builder(config),
        ),
        doorway=DoorwayMonitor(config.doorway),
        sessions=sessions,
        doorkeeper=doorkeeper,
        announce=_reporter(review, notifier, announce),
        on_frame=observer,
    )
    return Application(
        pipeline=pipeline,
        store=store,
        review=review,
        sessions=sessions,
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


def _reporter(review: ReviewQueue, notifier: TelegramNotifier | None, announce):
    """File every sighting for review, notify the chat, then hand it to the caller."""

    def report(sighting: Sighting) -> None:
        sighting_id = review.record(sighting, encode_jpeg=encode_jpeg)
        if notifier is not None:
            notifier.announce(sighting, sighting_id)
        announce(sighting)

    return report
