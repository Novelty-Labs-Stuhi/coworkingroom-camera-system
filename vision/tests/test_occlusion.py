"""Pixel change per vertical slice of the doorframe box: movement, and its order."""

from __future__ import annotations

import numpy as np

from stuhi_vision.occlusion import CoverageConfig, Occlusion

WIDTH, HEIGHT = 200, 400


def _empty() -> np.ndarray:
    """A still doorframe: some structure, so a change is a change and not noise."""
    rng = np.random.default_rng(4)
    image = rng.integers(90, 110, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[:, 40:60] = 200   # the frame itself, a bright vertical band
    return image


def _covered(columns: slice) -> np.ndarray:
    """The same doorframe with something dark in front of part of it."""
    image = _empty()
    image[:, columns] = 10
    return image


def _watch(**overrides) -> Occlusion:
    settings = {"slices": 5, "covered": 0.25, "min_frames": 2, "min_lag": 0.25}
    settings.update(overrides)
    return Occlusion(zone=(0.0, 0.0, 1.0, 1.0), config=CoverageConfig(**settings))


def _settle(watch: Occlusion, frames: int = 8) -> None:
    """Let the background be learned from an empty doorway."""
    for _ in range(frames):
        assert watch.update(_empty()) is None


def _play(watch: Occlusion, sweeps: list[slice]):
    """Show a covering moving across the box, then let it clear."""
    for columns in sweeps:
        episode = watch.update(_covered(columns))
        assert episode is None
    return watch.update(_empty())


def test_nothing_reported_while_the_doorway_is_empty() -> None:
    watch = _watch()
    _settle(watch)

    assert watch.update(_empty()) is None
    assert watch.busy is False


def test_a_covering_sweeping_one_way_gives_an_order() -> None:
    watch = _watch()
    _settle(watch)

    # Something moving right to left: the far slices light first.
    episode = _play(watch, [slice(150, 200), slice(100, 170), slice(40, 120), slice(0, 60)])

    assert episode is not None
    assert episode.slices >= 2
    assert episode.swept
    assert episode.lag < 0   # high-numbered slices lit first


def test_a_covering_sweeping_the_other_way_reverses_the_order() -> None:
    watch = _watch()
    _settle(watch)

    episode = _play(watch, [slice(0, 50), slice(30, 100), slice(80, 160), slice(140, 200)])

    assert episode is not None
    assert episode.swept
    assert episode.lag > 0


def test_standing_in_the_doorway_covers_the_box_without_an_order() -> None:
    watch = _watch()
    _settle(watch)

    # The same slices covered throughout: somebody at the door, not going through it.
    episode = _play(watch, [slice(70, 130)] * 5)

    assert episode is not None
    assert episode.frames == 5
    assert episode.swept is False   # movement, but no passage
    assert episode.lag == 0.0


def test_a_single_frame_of_change_is_not_an_episode() -> None:
    watch = _watch()
    _settle(watch)

    assert _play(watch, [slice(80, 120)]) is None


def test_a_person_standing_in_the_doorway_is_never_learnt_as_background() -> None:
    watch = _watch()
    _settle(watch)

    # Twenty frames of somebody standing there. If the background absorbed them, the coverage
    # would fade and their leaving would read as an event of its own.
    for _ in range(20):
        watch.update(_covered(slice(70, 130)))
    assert watch.busy
    assert watch.covered > 0.1

    episode = watch.update(_empty())
    assert episode is not None and episode.frames == 20


def test_a_light_switching_on_is_absorbed() -> None:
    watch = _watch(stuck_after=10)
    _settle(watch)

    brighter = _empty().astype(np.int16) + 40
    brighter = np.clip(brighter, 0, 255).astype(np.uint8)
    # The first frames read as covered -- everything changed at once -- and nothing would ever
    # clear it, because the background only learns from uncovered frames. So coverage that
    # outlasts any real passage is adopted as the new view instead of jamming the detector.
    for _ in range(30):
        assert watch.update(brighter) is None
    assert watch.covered < 0.25
    assert watch.busy is False
