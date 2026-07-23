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
(file/webcam/     (YOLO +           • best-face-so-far → "good enough" → recognise + lock
 ESP32 stream)     ByteTrack)        • sharpest body crop → lighting-normalised embedding
                          │
                          ▼
                  doorway monitor ─┬─ IN  ─▶ Doorkeeper ─▶ ledger.enter ─▶ store
                  (foot crosses    │        (name + entry embedding)
                   the line +      └─ OUT ─▶ Doorkeeper ─▶ ledger.exit  ─▶ store
                   persistence)             (body → best-pair match)
```

* **Recognition runs online, every frame** — not at the crossing. A `TrackSession`
  accumulates per track: the first face clear enough ("good enough" clarity) locks the
  identity early (low latency), and its frame's body crop becomes the stored entry
  embedding; the sharpest body crop is kept as the exit query / fallback.
* **Only a crossing commits** an in/out. Direction is pure geometry (the foot point's
  side of the line flips), so the head-count is right even when identity is uncertain.
  A **persistence gate** (min track age) rejects one-frame flicker. Sessions that never
  cross are pruned.
* **Lighting normalization** (CLAHE) is applied to every body crop before embedding, so
  the same person embeds similarly under different lighting.
* **Exit attribution**: *elimination* when one person is inside; otherwise the closest
  occupant wins only if it clears a **similarity threshold** AND beats the runner-up by
  a **margin** — else the exit is left unattributed rather than guessed.

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
