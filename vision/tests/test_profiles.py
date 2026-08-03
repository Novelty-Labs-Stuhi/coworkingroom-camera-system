"""The bio: the one thing on a profile that is stored rather than derived."""

from __future__ import annotations

import pytest

from stuhi_vision.profiles import LONGEST, ProfileStore


def test_everybody_starts_with_no_bio(tmp_path) -> None:
    store = ProfileStore(tmp_path / "profiles.json")

    assert store.bio("ilari") == ""


def test_a_bio_is_kept_and_survives_a_restart(tmp_path) -> None:
    path = tmp_path / "profiles.json"
    ProfileStore(path).write("ilari", "  Sits by the window.  ")

    assert ProfileStore(path).bio("ilari") == "Sits by the window."


def test_saving_nothing_clears_a_bio(tmp_path) -> None:
    store = ProfileStore(tmp_path / "profiles.json")
    store.write("ilari", "Something.")

    assert store.write("ilari", "   ") == ""
    assert store.bio("ilari") == ""


def test_a_bio_that_does_not_fit_is_refused_rather_than_cut(tmp_path) -> None:
    """Silently truncating what somebody wrote about themselves is worse than saying no."""
    store = ProfileStore(tmp_path / "profiles.json")

    with pytest.raises(ValueError, match=str(LONGEST)):
        store.write("ilari", "x" * (LONGEST + 1))

    assert store.bio("ilari") == ""


def test_a_bio_belongs_to_somebody(tmp_path) -> None:
    store = ProfileStore(tmp_path / "profiles.json")

    with pytest.raises(ValueError):
        store.write("  ", "words with nobody to own them")


def test_a_corrected_spelling_keeps_the_bio(tmp_path) -> None:
    """Spelling somebody's name properly must not cost them their profile."""
    store = ProfileStore(tmp_path / "profiles.json")
    store.write("ilari", "Sits by the window.")

    assert store.rename("ilari", "Ilari") is True
    assert store.bio("Ilari") == "Sits by the window."
    assert store.bio("ilari") == ""


def test_a_merge_keeps_both_sets_of_words(tmp_path) -> None:
    """Nothing a person wrote is discarded on the strength of a click somewhere else."""
    store = ProfileStore(tmp_path / "profiles.json")
    store.write("ilari", "Sits by the window.")
    store.write("Ilari", "Learning Finnish.")

    store.rename("ilari", "Ilari")

    assert "Sits by the window." in store.bio("Ilari")
    assert "Learning Finnish." in store.bio("Ilari")


def test_a_merge_of_the_same_words_does_not_double_them(tmp_path) -> None:
    store = ProfileStore(tmp_path / "profiles.json")
    store.write("ilari", "Sits by the window.")
    store.write("Ilari", "Sits by the window.")

    store.rename("ilari", "Ilari")

    assert store.bio("Ilari") == "Sits by the window."


def test_renaming_somebody_with_no_bio_moves_nothing(tmp_path) -> None:
    store = ProfileStore(tmp_path / "profiles.json")

    assert store.rename("ilari", "Ilari") is False
