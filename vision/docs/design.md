# Design

## Goal

A live ledger of **who is currently inside** the stuhi office, from a single camera
at the doorway. Identity must survive a change of clothes and a cap, so it is based
on the **face**, not on clothing.

## The core idea

Two different signals do two different jobs:

| Signal | Model | Job | Invariance |
|--------|-------|-----|------------|
| **Face** | InsightFace / ArcFace | *Name* a person on entry | clothes- and cap-invariant (durable, across days) |
| **Body** | torchvision backbone | *Link* an exit back to an entry | same-day only (same clothes that visit) |

A person entering walks toward the camera, so their face is visible → we name them.
A person leaving walks away, so there is no face → we match the back-of-body against
the body embedding we captured at entry, while they still wear today's clothes. The
face gives the durable identity; the body carries that identity to the exit.

## Flow

```
frame source ─▶ person tracker ─▶ TrackSession (per track, every frame)
(file/webcam/     (YOLO +           • RunningIdentity → best face-vs-gallery score so far
 ESP32 stream)     ByteTrack)        • sharpest body crop → lighting-normalised embedding
                          │
                          ▼
                  doorway monitor ─┬─ IN  ─▶ Doorkeeper ─▶ ledger.enter ─▶ store
                  (foot crosses    │        (name + entry embedding)   └▶ ReviewQueue
                   the line +      └─ OUT ─▶ Doorkeeper ─▶ ledger.exit  ─▶ store
                   persistence)             (body → best-pair match)      └▶ Telegram
```

* **Recognition runs online, every frame** — not at the crossing. A `TrackSession` keeps a
  `RunningIdentity`: the **highest** face-vs-gallery cosine seen across the track, plus the
  embedding and crop of the frame that produced it. There is no separate clarity gate — for
  the correct person a clearer, more frontal face simply scores higher, so the running
  maximum already prefers the best look. Clarity is still measured and stored as diagnostic
  metadata for tuning the thresholds.
* **Only a crossing commits** an in/out. Direction is pure geometry (the foot point's
  side of the line flips), so the head-count is right even when identity is uncertain.
  A **persistence gate** (min track age) rejects one-frame flicker. Sessions that never
  cross are pruned.
* **Three identity outcomes**, decided at the crossing: `NAMED` (cleared the threshold *and*
  beat the runner-up by `face_margin`), `UNKNOWN` (a face was seen but matched nobody), and
  `UNIDENTIFIED` (no usable face). The margin is what separates "this is a stranger", who is
  mediocre against everyone, from "a bad look at someone I know". An unmatched person is
  never force-matched to the closest name.
* **Lighting normalization** (CLAHE) is applied to every body crop before embedding, so
  the same person embeds similarly under different lighting.
* **Exit attribution**: *elimination* when one person is inside; otherwise the closest
  occupant wins only if it clears a **similarity threshold** AND beats the runner-up by
  a **margin** — else the exit is left unattributed rather than guessed.

## The labelling loop

An unrecognised face is not a dead end — it is how the gallery grows. Every crossing is
filed in a `ReviewQueue` with its face embedding and crop, and announced to Telegram. A
reply enrols it:

```
/label <sighting_id> <name>    or, as a reply to the sighting:  /label <name>
/pending                       sightings still waiting for a label
/people                        enrolled reference vectors per name
```

Labelling an already-labelled sighting **corrects** it: the embedding is removed from the
wrong name before being added to the right one, so a single mistake does not poison the
gallery permanently. This is why `FaceGallery.save` also deletes the files of names that
have been emptied — otherwise a reload would resurrect the wrong label.

The bot token and chat id are read from `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` in the
environment, never from the config file, so they cannot be committed. With them unset the
pipeline runs exactly as before, just without announcements.

One honest limitation: relabelling teaches the gallery and fixes the review record, but it
does **not** rewrite the SQLite occupancy event, which keeps the name it was recorded with.

## Why these boundaries

Identity is deliberately isolated. `recognition/` holds the models and the matching;
`ledger.py` holds the occupancy rules; `doorway.py` holds the geometry. Each can be
tested or replaced alone — e.g. swapping the body backbone for a stronger re-ID model
(OSNet) touches only `recognition/body.py`, and adding a second camera (CAM-OUT for
real exit faces) is just another frame source feeding the same ledger.

## Honest limitations

* **Front-enrol vs. back-exit** is the weakest case; elimination covers most of it in
  a small office, but two similarly-built people leaving together can be mis-attributed.
* **Body embeddings are same-day only** — a person in different clothes tomorrow is a
  new body vector, re-linked to their durable face identity at the next entry.
* A pure side/profile camera mount undermines face recognition; angle the camera so
  entrants present a roughly frontal face.

## Known next steps (not built)

* A second doorway camera aimed to catch faces on exit (turns exits from
  body-matching into face-recognition — the biggest robustness win).
* Multi-view / multi-visit face enrolment to strengthen the gallery.
* Nightly ledger reset so any accumulated exit error does not persist.
