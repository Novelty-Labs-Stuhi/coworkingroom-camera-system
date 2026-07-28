"""The web labelling UI, an alternative to the Telegram route over the same queue."""

from .app import WebUI, create_app

__all__ = ["WebUI", "create_app"]
