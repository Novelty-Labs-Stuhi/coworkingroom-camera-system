"""Reading person names from whatever a human typed.

Both labelling routes -- the chat and the web UI -- take free text from a person watching a
clip, so both must read it the same way. They did not: the chat split a comma-separated
reply into a name each, while the web UI passed the whole string through as one name. The
result was a gallery containing people called ``a, yehor`` and ``art, ilar, hubertus``,
each with a single reference, silently competing with the real ones.

So parsing lives here, once, and callers use it rather than deciding for themselves.
"""

from __future__ import annotations


def parse_names(text: str) -> list[str]:
    """Read one or more names from free text, in the order they were written.

    Accepts ``ilari``, ``a, b, c`` and ``[a, b, c]`` -- people write a list either way and
    the brackets carry no meaning. Commas separate, so a name may contain spaces. Anything
    command-like is refused, so a mistyped ``/pending`` cannot enrol a face called
    "pending".
    """
    cleaned = text.strip().strip("[]").strip()
    if not cleaned or cleaned.startswith("/"):
        return []
    names = [part.strip() for part in cleaned.split(",")]
    return [name for name in names if name and not name.startswith("/")]
