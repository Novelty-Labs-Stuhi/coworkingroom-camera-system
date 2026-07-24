"""Persistent store of clip re-ID embeddings, indexed by video id and grouped by name.

On-disk layout (JSON):

    {
      "threshold": 0.80,
      "people": {
        "Arsenii": [{"video_id": "2026-07-24_20-59-21", "emb": [...]}],
        "unknown": [{"video_id": "2026-07-24_20-46-14", "emb": [...]}]
      }
    }

A new clip is matched against every *labeled* person (everyone except ``unknown``) and
named when its best cosine similarity clears ``threshold``; otherwise it lands under
``unknown`` until a human relabels it (e.g. by replying in Telegram). The threshold is
the one tuning knob and can be changed at runtime.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from .recognition.embeddings import cosine

UNKNOWN = "unknown"


class ReidStore:
    """Labeled appearance-vector store with a tunable match threshold."""

    def __init__(self, path: Path | str, threshold: float = 0.80) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self.threshold = threshold
        self.people: dict[str, list[dict]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        data = json.loads(self._path.read_text())
        self.threshold = float(data.get("threshold", self.threshold))
        self.people = {
            name: [
                {"video_id": e["video_id"], "emb": np.asarray(e["emb"], np.float32)}
                for e in entries
            ]
            for name, entries in data.get("people", {}).items()
        }

    def _save(self) -> None:
        data = {
            "threshold": self.threshold,
            "people": {
                name: [
                    {"video_id": e["video_id"], "emb": e["emb"].tolist()} for e in entries
                ]
                for name, entries in self.people.items()
            },
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data))

    def match(self, emb: np.ndarray) -> tuple[str | None, float]:
        """Best (name, score) among labeled people. Name is None if below threshold.

        The score is always the best similarity found, even when it is below the
        threshold -- useful to show "unknown (best 0.72)" in the notification.
        """
        with self._lock:
            best_name, best_score = None, -1.0
            for name, entries in self.people.items():
                if name == UNKNOWN:
                    continue
                for entry in entries:
                    score = cosine(emb, entry["emb"])
                    if score > best_score:
                        best_name, best_score = name, score
        if best_name is not None and best_score >= self.threshold:
            return best_name, best_score
        return None, best_score

    def add(self, video_id: str, emb: np.ndarray, name: str) -> None:
        """Store an embedding under ``name`` (use ``UNKNOWN`` when unmatched)."""
        with self._lock:
            self.people.setdefault(name, []).append(
                {"video_id": video_id, "emb": np.asarray(emb, np.float32)}
            )
            self._save()

    def relabel(self, video_id: str, name: str) -> bool:
        """Move the clip's embedding (wherever it is) under ``name``. False if unknown id."""
        with self._lock:
            found = None
            for entries in self.people.values():
                for i, entry in enumerate(entries):
                    if entry["video_id"] == video_id:
                        found = entries.pop(i)
                        break
                if found is not None:
                    break
            if found is None:
                return False
            self.people.setdefault(name, []).append(found)
            self._save()
            return True

    def set_threshold(self, value: float) -> None:
        with self._lock:
            self.threshold = float(value)
            self._save()

    def summary(self) -> dict[str, int]:
        """Name -> number of stored clips, for a quick '/people' overview."""
        with self._lock:
            return {name: len(entries) for name, entries in self.people.items()}
