"""Identities the system invented that now look like somebody it knows.

Most crossings are never looked at. Naming a person repairs every clip that identity produced
-- but only for identities somebody actually opened. The rest sit as ``guest-0803-0900``
forever, even when the gallery has since learned enough to recognise them.

So after the gallery grows, every provisional identity is asked again: does this face now
clearly belong to one of the people we know? Not to *apply* an answer -- that is the one thing
this deliberately does not do -- but to offer the pair to a human as a single question worth
one click, in place of the hundreds of clips behind it.

Three rules keep it from doing harm:

* **it proposes, it never merges.** A wrong merge fuses two people's histories, and doing that
  automatically means doing it unattended and at scale;
* **the bar is higher than the door's.** Recognition at a doorway has a person walking past and
  must answer now. Here there is no hurry and no cost to saying nothing, so a claim made after
  the fact should be held to more, not less;
* **only provisional identities are ever the ones absorbed.** A person somebody named is never
  proposed for folding into somebody else.
"""

from __future__ import annotations

from dataclasses import dataclass

# Deliberately stricter than ``face_match`` at the door, and with a wider margin: this claim is
# made with time to spare, about somebody who is not standing there, and nothing is lost by
# leaving a pair unproposed.
SIMILARITY = 0.55
MARGIN = 0.10


@dataclass(frozen=True, slots=True)
class Resemblance:
    """A provisional identity that looks like a named person, offered for confirmation."""

    provisional: str      # the invented identity
    resembles: str        # the named person it looks like
    score: float          # how alike, 0..1
    runner_up: float      # the next best name, so the gap is visible
    crossings: int = 0    # how many recorded crossings would be corrected by accepting

    @property
    def readable(self) -> str:
        return (
            f"{self.provisional} looks like {self.resembles} "
            f"({self.score:.2f} vs {self.runner_up:.2f} next, {self.crossings} crossing(s))"
        )


def _reference(gallery, name: str):
    """One embedding standing for this identity, or None if it holds no faces."""
    references = gallery.references_for(name)
    return references[0] if references else None


def suggestions(
    gallery,
    provisional: set[str],
    counts: dict[str, int] | None = None,
    similarity: float = SIMILARITY,
    margin: float = MARGIN,
) -> list[Resemblance]:
    """Provisional identities that now clearly resemble a named person, best first.

    ``counts`` maps a name to how many crossings it has, so the offer can say how much is at
    stake. Absent, the pairs are still correct, just less informative.
    """
    named = [name for name in gallery.names if name not in provisional]
    if not named:
        return []

    found: list[Resemblance] = []
    for candidate in sorted(provisional):
        embedding = _reference(gallery, candidate)
        if embedding is None:
            continue
        # Only people with names compete. Ranking against other provisional identities would
        # propose folding two strangers together, which names nobody and loses both.
        ranked = [match for match in gallery.rank(embedding) if match.name in named]
        if not ranked:
            continue
        best = ranked[0]
        runner_up = ranked[1].score if len(ranked) > 1 else 0.0
        if best.score < similarity or best.score - runner_up < margin:
            continue
        found.append(
            Resemblance(
                provisional=candidate,
                resembles=best.name,
                score=float(best.score),
                runner_up=float(runner_up),
                crossings=(counts or {}).get(candidate, 0),
            )
        )
    return sorted(found, key=lambda pair: pair.score, reverse=True)
