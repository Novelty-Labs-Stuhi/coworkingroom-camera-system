"""Audit the gallery: which enrolled faces look wrong, and who needs more examples.

Labelling is done by a human watching short clips, so mistakes are inevitable -- a name
typed against the wrong clip, a group labelled in the wrong order, the back of someone's
head enrolled as their face. A wrong reference vector is worse than a missing one: it drags
every future comparison for that person, and nothing about it looks unusual from outside.

Two checks, because they catch different mistakes:

**Outliers.** Every reference for a person is compared against that person's *own* average.
A face that sits far below the others is either somebody else or an unusable capture. This
needs at least three references to mean anything: with two, each is equidistant from their
midpoint by construction, so no outlier can exist.

**Thin enrolments.** One or two references is not enough to recognise anyone reliably, and
it is also the case where an outlier cannot be detected. So it is reported separately
rather than hidden behind the outlier check.

The similarity numbers are reported rather than just a verdict, so a human can judge a
borderline case instead of trusting a threshold.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .recognition.embeddings import cosine, normalize

# A reference this far below its person's average similarity is suspect. Relative rather
# than absolute, because how tightly a person's faces cluster varies with lighting and pose.
_OUTLIER_MARGIN = 0.12
# ...and this is suspect regardless of the rest: nothing this dissimilar is the same face.
_ABSOLUTE_FLOOR = 0.35
# Fewer references than this and recognition is unreliable -- and outliers are undetectable.
_MIN_REFERENCES = 3


@dataclass(frozen=True, slots=True)
class Reference:
    """One enrolled face: which sighting it came from, and who it is filed under."""

    sighting_id: str
    name: str
    embedding: np.ndarray


@dataclass(frozen=True, slots=True)
class Suspect:
    """An enrolled reference that does not look like the rest of that person's faces."""

    sighting_id: str
    name: str
    similarity: float  # cosine against that person's average face
    average: float  # the average similarity among that person's references
    reason: str

    @property
    def shortfall(self) -> float:
        return round(self.average - self.similarity, 4)


@dataclass(frozen=True, slots=True)
class Audit:
    """What the gallery looks like from the inside."""

    suspects: list[Suspect] = field(default_factory=list)
    thin: dict[str, int] = field(default_factory=dict)  # name -> reference count


def audit(
    references: Iterable[Reference],
    min_references: int = _MIN_REFERENCES,
    margin: float = _OUTLIER_MARGIN,
    floor: float = _ABSOLUTE_FLOOR,
) -> Audit:
    """Find suspect references and under-enrolled people."""
    by_name: dict[str, list[Reference]] = {}
    for reference in references:
        by_name.setdefault(reference.name, []).append(reference)

    suspects: list[Suspect] = []
    thin: dict[str, int] = {}
    for name, group in sorted(by_name.items()):
        if len(group) < min_references:
            thin[name] = len(group)
        suspects.extend(_suspects_in(group, margin, floor))

    suspects.sort(key=lambda suspect: suspect.similarity)
    return Audit(suspects=suspects, thin=thin)


def _suspects_in(group: Sequence[Reference], margin: float, floor: float) -> list[Suspect]:
    """Compare each reference in one person's group against the group's average face."""
    if len(group) < 2:
        return []  # nothing to compare against
    if len(group) == 2:
        return _suspect_pair(group, floor)

    average_face = normalize(np.mean([reference.embedding for reference in group], axis=0))
    similarities = [cosine(reference.embedding, average_face) for reference in group]
    average = float(np.mean(similarities))

    found = []
    for reference, similarity in zip(group, similarities, strict=True):
        reason = _reason(similarity, average, len(group), margin, floor)
        if reason is None:
            continue
        found.append(
            Suspect(
                sighting_id=reference.sighting_id,
                name=reference.name,
                similarity=round(float(similarity), 4),
                average=round(average, 4),
                reason=reason,
            )
        )
    return found


def _suspect_pair(group: Sequence[Reference], floor: float) -> list[Suspect]:
    """Two references must be compared against *each other*, not their average.

    Their mean sits midway between them, so two completely different faces each score about
    0.71 against it -- comfortably above any sane floor. Comparing the pair directly is the
    only way to notice that they are not the same person, and when they are not, either one
    could be the impostor, so both are flagged for a human to decide.
    """
    similarity = round(float(cosine(group[0].embedding, group[1].embedding)), 4)
    if similarity >= floor:
        return []
    return [
        Suspect(
            sighting_id=reference.sighting_id,
            name=reference.name,
            similarity=similarity,
            average=similarity,
            reason="not the same face",
        )
        for reference in group
    ]


def _reason(
    similarity: float, average: float, group_size: int, margin: float, floor: float
) -> str | None:
    if similarity < floor:
        return "not the same face"
    # With only two references each sits the same distance from their midpoint, so a
    # relative comparison says nothing; only the absolute floor above applies.
    if group_size >= 3 and similarity < average - margin:
        return "unlike this person's other faces"
    return None
