"""Cache-busting for the stylesheets and scripts.

Its own module because more than one routing module serves pages now, and the alternative was
one importing the other for this -- which is a circular import waiting for whichever of them
grows a second reason to.
"""

from __future__ import annotations

from pathlib import Path

_STATIC = Path(__file__).parent / "static"


def asset_version() -> str:
    """A token that changes when the static files do, for cache-busting their URLs.

    Without it a browser keeps yesterday's stylesheet: a CSS fix that squeezed the name field
    to 41 px was deployed, served correctly, and still broken on screen. Telling somebody to
    hard-refresh is not a fix, it is a thing to remember forever.
    """
    newest = max((path.stat().st_mtime for path in _STATIC.glob("*")), default=0.0)
    return str(int(newest))
