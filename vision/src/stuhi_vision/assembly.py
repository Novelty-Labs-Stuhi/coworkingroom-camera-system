"""Composition root: build the wired-up application from a config.

This is the one place that knows how every concrete component fits together, so the
modules themselves stay free of construction logic and the CLI stays thin.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .doorway import DoorwayMonitor
from .handlers import Doorkeeper
from .ledger import Ledger
from .pipeline import FrameObserver, Pipeline
from .recognition.body import BodyEmbedder
from .recognition.face import FaceRecognizer
from .recognition.gallery import FaceGallery
from .sessions import SessionManager
from .sources import open_source
from .store import EventStore
from .tracking import PersonTracker


@dataclass(slots=True)
class Application:
    pipeline: Pipeline
    store: EventStore


def build(config: Config, announce, observer: FrameObserver | None = None) -> Application:
    """Assemble every component described by ``config`` into a ready pipeline.

    ``observer`` is an optional per-frame hook (e.g. an Annotator) for offline review.
    """
    thresholds = config.thresholds
    gallery = FaceGallery.load(config.paths.gallery_dir)
    faces = FaceRecognizer(gallery, thresholds.face_match)
    bodies = BodyEmbedder()
    store = EventStore(config.paths.database)
    ledger = Ledger(store, thresholds.exit_similarity, thresholds.exit_margin)
    sessions = SessionManager(faces, bodies, thresholds.face_clarity_min)
    doorkeeper = Doorkeeper(sessions, ledger, thresholds.min_track_age)

    pipeline = Pipeline(
        source=open_source(config.source),
        tracker=PersonTracker(detection_conf=thresholds.detection_conf),
        doorway=DoorwayMonitor(config.doorway),
        sessions=sessions,
        doorkeeper=doorkeeper,
        announce=announce,
        on_frame=observer,
    )
    return Application(pipeline=pipeline, store=store)
