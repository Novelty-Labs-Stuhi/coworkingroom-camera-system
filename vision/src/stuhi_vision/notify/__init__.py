"""Outbound notification and the human-in-the-loop labelling channel."""

from .telegram import TelegramNotifier

__all__ = ["TelegramNotifier"]
