"""Entry point for ``python -m stuhi_vision`` -- delegates to the Typer CLI."""

from __future__ import annotations

from .cli import app

if __name__ == "__main__":
    app()
