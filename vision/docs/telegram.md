# Telegram setup

The chat is how the system tells you who came and went, and how you teach it who people
are. A face nobody recognised is announced with its crop; replying enrols it.

## 1. Create the bot

1. Message [@BotFather](https://t.me/BotFather) on Telegram, send `/newbot`, and follow the
   prompts. It replies with a **bot token** that looks like `123456789:AA...`.
2. Send your new bot any message (a bot cannot start a conversation with you).
3. Get your **chat id** by opening, in a browser:
   `https://api.telegram.org/bot<TOKEN>/getUpdates` — look for `"chat":{"id":...}`.

## 2. Put the credentials in the environment

They are read from the environment, never from `config.toml`, so they cannot be committed:

```bash
export TELEGRAM_BOT_TOKEN=123456789:AA...
export TELEGRAM_CHAT_ID=987654321
```

Add those two lines to `~/.profile` on the machine that runs the pipeline so they survive
a login. With them unset, everything still runs -- just with no announcements.

## 3. Verify before relying on it

```bash
stuhi-vision telegram-check
```

Sends a test message and prints who is enrolled. If the credentials are wrong this fails
here, rather than silently swallowing every later notification.

## 4. Using it

Each doorway crossing arrives as **one video** -- the approach, the crossing, and enough
afterwards that it ends on an empty doorway -- captioned with one of:

| Caption | Meaning |
|---|---|
| `in: ilari (0.61)` | matched the gallery confidently |
| `in: unknown (best 0.22)` | a face was seen but matched nobody — **label it** |
| `out: ilari (linked)` | no face at the exit; linked to an occupant by body |
| `in: no face seen` | crossing counted, but no usable face |

The face crop is still written to the review directory for inspection; it is not sent, so
each crossing is exactly one message.

Commands:

```
/label <sighting_id> <name>   enrol that sighting's face as <name>
/label <name>                 same, as a reply to the sighting's message
/pending                      sightings still waiting for a label
/people                       enrolled reference vectors per name
```

**Labelling is the enrolment.** There is no separate setup session: walk through the door,
then label the crops that arrive. Each label adds one more reference vector for that
person, so accuracy improves as you use it.

**Labelling again corrects it.** The embedding is removed from the wrong name before being
added to the right one, so one mistake does not stay in the gallery.

> Relabelling teaches the gallery and fixes the review record, but it does **not** rewrite
> the SQLite occupancy event, which keeps the name it was recorded with.

## Practical notes

- Expect **every** early crossing to be `unknown`. That is the gallery being empty, not a
  fault. Label a handful and it settles down.
- Prefer labelling crops that are large and face-on. A blurry crop enrolled as someone
  makes every later match worse, and there is no way to tell which reference vector is
  responsible.
- `/pending` is the backlog. If it grows, either faces are too small at the mounting
  position, or the match threshold is too strict for your camera.

## The web UI

The same queue is also labelled from a browser, served from inside the pipeline process:

```
http://<server>:8800
```

It shows each waiting sighting as its clip, with a name box; already-labelled sightings
appear below so a name can be corrected. Enrolled people and their reference counts are
listed at the top — that count is the honest check that a label added a reference.

Both routes drive one `ReviewQueue` and one `FaceGallery`, so:

- a sighting labelled in the chat shows as labelled in the UI, and vice versa;
- labelling the same sighting from **both** still adds exactly one reference vector;
- a label takes effect on the next frame, with no restart.

Running in the pipeline process is what makes that last point true: the gallery lives in
memory inside the recogniser, so a separate process could only write files and the running
pipeline would not see the new face until it restarted.

Disable it with `enabled = false` under `[web]`.
