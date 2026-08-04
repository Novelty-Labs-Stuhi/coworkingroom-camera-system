"""The publish wiring: a sighting's place in its burst must reach the stored record.

This path had no test, and two bugs slipped through it in one evening -- arguments the
review queue did not accept (a TypeError on the next crossing), then the burst id being
dropped so a group could not be labelled by crossing order. Both were invisible until a
real person walked through a door, which is far too late to find out.
"""

from __future__ import annotations

import numpy as np

from stuhi_vision.assembly import _publisher
from stuhi_vision.corroboration import Corroboration
from stuhi_vision.domain import Direction, Outcome, Sighting
from stuhi_vision.publishing import Publication
from stuhi_vision.recognition.gallery import FaceGallery
from stuhi_vision.review import ReviewQueue


def _publication(position: int, burst: int, seed: int = 0) -> Publication:
    embedding = np.zeros(4, dtype=np.float32)
    embedding[seed % 4] = 1.0
    return Publication(
        sighting=Sighting(
            timestamp=1_760_000_000.0 + seed,
            direction=Direction.IN,
            name=None,
            score=0.2,
            outcome=Outcome.UNKNOWN,
            face_embedding=embedding,
        ),
        clip=None,
        position=position,
        total=2,
        burst=burst,
    )


def test_position_and_burst_reach_the_stored_record(tmp_path) -> None:
    gallery = FaceGallery()
    review = ReviewQueue(tmp_path / "review", gallery, tmp_path / "gallery")
    publish = _publisher(review, None, lambda sighting: None)

    publish(_publication(position=1, burst=4, seed=0))
    publish(_publication(position=2, burst=4, seed=1))

    records = sorted(review.pending(), key=lambda r: r.position)
    assert [(r.position, r.burst) for r in records] == [(1, 4), (2, 4)]


def test_a_group_stored_this_way_can_be_labelled_in_order(tmp_path) -> None:
    gallery = FaceGallery()
    review = ReviewQueue(tmp_path / "review", gallery, tmp_path / "gallery")
    publish = _publisher(review, None, lambda sighting: None)
    publish(_publication(position=1, burst=9, seed=0))
    publish(_publication(position=2, burst=9, seed=1))

    first = sorted(review.pending(), key=lambda r: r.position)[0].sighting_id
    review.label_burst(first, ["a", "b"])

    assert gallery.counts() == {"a": 1, "b": 1}


def test_the_ui_is_fed_frames_by_the_reader(tmp_path) -> None:
    """Removing the old feed without the new one landing left the UI with no picture at all.

    Deployed, that was a broken image on the zone drawing tool and an empty camera list, and
    neither end of the wiring complains on its own -- the pipeline runs happily while the page
    has nothing to show. So the collaboration is asserted, not trusted.
    """
    import numpy as np

    from stuhi_vision.assembly import _Shared, _source
    from stuhi_vision.clips import ClipRecorder
    from stuhi_vision.config import Performance
    from stuhi_vision.domain import Frame
    from stuhi_vision.latest import LatestFrames
    from stuhi_vision.zones import ZoneStore

    frames = LatestFrames(lambda image: b"jpeg")
    shared = _Shared(
        gallery=None,
        gallery_dir=tmp_path / "gallery",
        ledger=None,
        review=None,
        notifier=None,
        frames=frames,
        zones=ZoneStore(tmp_path),
        witness=None,
        passages=None,
        boundary=None,
        heartbeats=tmp_path / "heartbeat",
        provisional=None,
        strangers=None,
        corroboration=Corroboration(),
    )
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    camera = [Frame(timestamp=float(i), image=image) for i in range(3)]

    source = _source(camera, "door-in", Performance(), shared, ClipRecorder(lambda i: b"x"))
    assert [frame.timestamp for frame in source] == [0.0, 1.0, 2.0]
    assert frames.cameras == ["door-in"]
    assert frames.jpeg("door-in") == b"jpeg"


# --- what reaches the chat ----------------------------------------------------------------


def _named(introduced: bool, name: str):
    from stuhi_vision.domain import Direction, Outcome, Sighting
    from stuhi_vision.publishing import Publication

    return Publication(
        sighting=Sighting(
            timestamp=1000.0,
            direction=Direction.IN,
            name=name,
            score=0.4,
            outcome=Outcome.NAMED,
            introduced=introduced,
        ),
        clip=b"mp4",
        position=1,
        total=1,
        burst=0,
    )


class _Notifier:
    def __init__(self) -> None:
        self.sent = []

    def announce(self, sighting, sighting_id, position, total) -> None:
        self.sent.append(sighting.name)


class _Review:
    def record(self, sighting, **_) -> str:
        return "sighting-1"


def test_only_a_newly_invented_identity_reaches_the_chat() -> None:
    """Every crossing used to be sent, which made the chat a feed nobody reads."""
    from stuhi_vision.assembly import _publisher

    notifier = _Notifier()
    filed = []
    publish = _publisher(_Review(), notifier, filed.append)

    publish(_named(introduced=True, name="guest-0803-0900"))
    publish(_named(introduced=False, name="ilari"))
    publish(_named(introduced=False, name="guest-0803-0900"))

    assert notifier.sent == ["guest-0803-0900"]
    # Everything is still filed and still counted -- only the messaging is narrowed.
    assert [s.name for s in filed] == ["guest-0803-0900", "ilari", "guest-0803-0900"]


def test_a_regular_arriving_every_day_is_never_announced() -> None:
    from stuhi_vision.assembly import _publisher

    notifier = _Notifier()
    publish = _publisher(_Review(), notifier, lambda sighting: None)

    for _ in range(20):
        publish(_named(introduced=False, name="arsenii"))

    assert notifier.sent == []
