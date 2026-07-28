"""Telegram channel: announce who came and went, and take labels back.

Announcing is a push; labelling is a pull. So the notifier does both:

* :meth:`TelegramNotifier.announce` posts each crossing -- the face crop when there is
  one, captioned with the name or ``unknown``, the score, and the sighting id;
* :meth:`TelegramNotifier.start` runs a background thread long-polling ``getUpdates`` for
  commands, so a reply in the chat enrols or corrects a face.

Commands (mirroring the older clip receiver, so the habits carry over)::

    /label <sighting_id> <name>   enrol that sighting's face as <name>
    /label <name>                 same, as a reply to the sighting's message
    /pending                      sightings still waiting for a label
    /people                       enrolled reference vectors per name

Labelling an already-labelled sighting corrects it. The bot token and chat id come from
the environment, never from the config file, so they cannot be committed.
"""

from __future__ import annotations

import threading
from pathlib import Path

from ..domain import Outcome, Sighting
from ..names import parse_names
from ..review import LabelOutcome, ReviewQueue

_API = "https://api.telegram.org/bot{token}/{method}"
_POLL_TIMEOUT = 30  # seconds held open by getUpdates; a long poll, not a busy loop
_ID_PREFIX = "id: "

_HELP = (
    "to label: reply to a video and type the name.\n"
    "several people at once: reply with their names in crossing order, e.g. a, b, c\n"
    "\n"
    "/pending - sightings waiting for a label\n"
    "/people - enrolled faces per name"
)


class TelegramNotifier:
    """Posts sightings to a chat and applies labels replied back from it."""

    def __init__(self, token: str, chat_id: str, review: ReviewQueue) -> None:
        self._token = token
        self._chat_id = chat_id
        self._review = review
        self._http = None  # httpx is an optional extra; imported on first use
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def _client(self):
        if self._http is None:
            import httpx

            self._http = httpx.Client(timeout=_POLL_TIMEOUT + 30)
        return self._http

    # --- outbound -----------------------------------------------------------
    def announce(
        self, sighting: Sighting, sighting_id: str, position: int = 1, total: int = 1
    ) -> None:
        """Post one crossing as a clip. Never raises -- a chat outage must not stop work.

        One message per crossing: the clip shows which way the person went and whether
        anyone came through with them, which is what makes it judgeable. The face crop is
        still saved to disk for inspection, but sending it too doubled the traffic in the
        chat for little gain. The still is only used when no clip could be made.
        """
        caption = _caption(sighting, sighting_id, position, total)
        clip = self._review.clip_path(sighting_id)
        crop = self._review.crop_path(sighting_id)
        try:
            if clip is not None:
                self._send_video(clip, caption)
            elif crop is not None:
                self._send_photo(crop, caption)
            else:
                self._send_message(caption)
        except Exception as exc:
            print(f"  -> telegram send failed: {exc}")

    def _send_video(self, path: Path, caption: str) -> None:
        self._client.post(
            _API.format(token=self._token, method="sendVideo"),
            data={
                "chat_id": self._chat_id,
                "caption": caption,
                "supports_streaming": "true",
            },
            files={"video": (path.name, path.read_bytes(), "video/mp4")},
        )

    def send_test(self, text: str) -> None:
        """Send one message, letting errors surface -- used to verify the credentials."""
        response = self._client.post(
            _API.format(token=self._token, method="sendMessage"),
            data={"chat_id": self._chat_id, "text": text},
        )
        if response.status_code != 200:
            raise RuntimeError(f"telegram rejected the request: {response.text}")

    def _send_photo(self, path: Path, caption: str) -> None:
        self._client.post(
            _API.format(token=self._token, method="sendPhoto"),
            data={"chat_id": self._chat_id, "caption": caption},
            files={"photo": (path.name, path.read_bytes(), "image/jpeg")},
        )

    def _send_message(self, text: str) -> None:
        self._client.post(
            _API.format(token=self._token, method="sendMessage"),
            data={"chat_id": self._chat_id, "text": text},
        )

    # --- inbound ------------------------------------------------------------
    def start(self) -> None:
        """Begin polling for label commands in a background thread."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._poll, name="telegram-poll", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=_POLL_TIMEOUT + 5)
            self._thread = None
        if self._http is not None:
            self._http.close()
            self._http = None

    def _poll(self) -> None:
        offset = 0
        while not self._stop.is_set():
            try:
                response = self._client.get(
                    _API.format(token=self._token, method="getUpdates"),
                    params={"offset": offset, "timeout": _POLL_TIMEOUT},
                )
                for update in response.json().get("result", []):
                    offset = update["update_id"] + 1
                    message = update.get("message")
                    if message:
                        self._handle(message)
            except Exception as exc:
                print(f"  -> telegram poll error: {exc}")
                self._stop.wait(5)

    def _handle(self, message: dict) -> None:
        text = (message.get("text") or "").strip()
        if not text:
            return
        if not text.startswith("/"):
            # A plain reply to a sighting is a label. Typing a name is the whole gesture --
            # no command to remember -- and a comma-separated list names a whole group in
            # the order they crossed. Plain text that is not a reply is ignored, so ordinary
            # chatter cannot accidentally enrol a face.
            self._handle_reply_label(message, text)
            return
        parts = text.split()
        command, args = parts[0].lstrip("/").lower(), parts[1:]

        if command == "label":
            self._handle_label(message, args)
        elif command == "pending":
            self._handle_pending()
        elif command == "people":
            self._handle_people()
        else:
            self._send_message(_HELP)

    def _handle_reply_label(self, message: dict, text: str) -> None:
        """Label from a plain reply: one name, or several in crossing order."""
        sighting_id = _sighting_id_from_reply(message)
        if sighting_id is None:
            return  # not a reply to a sighting; stay silent rather than guess
        names = parse_names(text)
        if not names:
            return
        if len(names) == 1:
            self._report_label(sighting_id, names[0])
            return

        results = self._review.label_burst(sighting_id, names)
        if not results:
            self._send_message(f"unknown sighting id: {sighting_id}")
            return
        counts = self._review.counts()
        lines = [
            f"{position}. {name}: {outcome.value}"
            for position, ((_id, outcome), name) in enumerate(zip(results, names, strict=False), 1)
        ]
        if len(names) > len(results):
            lines.append(f"only {len(results)} crossed together; extra names ignored")
        lines.append(", ".join(f"{n}={counts.get(n, 0)}" for n in dict.fromkeys(names)))
        self._send_message("\n".join(lines))

    def _report_label(self, sighting_id: str, name: str) -> None:
        outcome = self._review.label(sighting_id, name)
        counts = self._review.counts()
        replies = {
            LabelOutcome.ENROLLED: f"enrolled as {name} ({counts.get(name, 0)} reference(s))",
            LabelOutcome.CORRECTED: f"corrected to {name} ({counts.get(name, 0)} reference(s))",
            LabelOutcome.UNCHANGED: (
                f"already labelled {name} -- nothing added, one sighting counts once"
            ),
            LabelOutcome.NO_FACE: f"no face was kept for {sighting_id} - nothing to enrol",
            LabelOutcome.UNKNOWN_ID: f"unknown sighting id: {sighting_id}",
        }
        self._send_message(replies[outcome])

    def _handle_label(self, message: dict, args: list[str]) -> None:
        if len(args) >= 2:
            sighting_id, name = args[0], " ".join(args[1:])
        elif len(args) == 1:
            replied_id = _sighting_id_from_reply(message)
            if replied_id is None:
                self._send_message("reply to a sighting, or: /label <sighting_id> <name>")
                return
            sighting_id, name = replied_id, args[0]
        else:
            self._send_message("usage: /label <sighting_id> <name>")
            return

        self._report_label(sighting_id, name)

    def _handle_pending(self) -> None:
        records = self._review.pending()
        if not records:
            self._send_message("nothing waiting for a label")
            return
        lines = [
            f"{record.sighting_id}  {record.direction}  {record.display_name} ({record.score:.2f})"
            for record in records
        ]
        self._send_message("\n".join(lines))

    def _handle_people(self) -> None:
        counts = self._review.counts()
        if not counts:
            self._send_message("gallery is empty - label a sighting to enrol someone")
            return
        self._send_message("\n".join(f"{name}: {count}" for name, count in counts.items()))


def _caption(sighting: Sighting, sighting_id: str, position: int = 1, total: int = 1) -> str:
    """The message body: who, which way, how sure, and the id to reply with.

    When several people come through together their clips are cut from the same window and
    look nearly identical, so the crossing order is the only thing distinguishing them --
    it is stated first, because it is what you label by.
    """
    if sighting.outcome is Outcome.NAMED:
        who = f"{sighting.name} ({sighting.score:.2f})"
    elif sighting.name is not None:
        # An exit shows no face, so the name came from the ledger linking it to an occupant.
        who = f"{sighting.name} (linked)"
    elif sighting.outcome is Outcome.UNKNOWN:
        who = f"unknown (best {sighting.score:.2f})"
    else:
        who = "no face seen"
    lines = []
    if total > 1:
        lines.append(f"person {position} of {total} to cross")
    lines.append(f"{sighting.direction.value}: {who}")
    lines.append(f"{_ID_PREFIX}{sighting_id}")
    if sighting.outcome is Outcome.UNKNOWN:
        lines.append("reply with the name" + (", or names in order" if total > 1 else ""))
    return "\n".join(lines)


def _sighting_id_from_reply(message: dict) -> str | None:
    """Pull the sighting id out of the caption of the message being replied to."""
    replied = message.get("reply_to_message", {})
    body = replied.get("caption") or replied.get("text") or ""
    for line in body.splitlines():
        if line.startswith(_ID_PREFIX):
            return line[len(_ID_PREFIX) :].strip()
    return None
