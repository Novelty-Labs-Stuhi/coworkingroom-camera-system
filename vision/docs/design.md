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

## Keeping up on a slow machine

Measured on the deployment server (Xeon E5-1650, 2012, **no AVX2** — torch logs
`Could not initialize NNPACK! Reason: Unsupported hardware` and falls back to its slowest
kernels):

| Stage | Before | After |
|---|---|---|
| YOLO @ imgsz 640 | 440 ms | — |
| YOLO @ imgsz 320 | — | 204 ms |
| InsightFace detection | 400 ms | 219 ms |
| Live stream, end to end | 2.25 fps | — |

At ~2 fps a doorway crossing lasting a second is seen once or twice, and `min_track_age`
would reject it. The response is deliberately **not** to approximate the vision, but to
stop wasting frames and stop looking where nothing happens:

* **`BufferedSource`** — a reader thread drains the camera at full rate into a bounded
  queue. This is the important one: previously every frame arriving *during* processing was
  lost, so a one-second crossing yielded whatever we happened to be free for. Now it banks
  ~12 frames and we may spend several seconds on them. Frames stay in order, so tracking is
  unaffected; the cost is latency, which occupancy logging does not care about. The queue
  drops its **oldest** frame when full, and counts drops.
* **`MotionGate`** — an unchanged frame cannot contain a crossing, so it never reaches the
  models. Idle cost falls to about a millisecond a frame. Hysteresis (`linger_frames`) keeps
  the gate open briefly after movement so a person pausing mid-stride does not flicker it
  shut and break their track. The reference is the previous frame, not a learned background:
  a background model would absorb a stationary person, who is exactly who matters.
* **`Region`** — detection runs on the crop around the doorway line. Boxes are translated
  back to full-frame coordinates immediately, so nothing downstream can tell. That
  translation is the one thing here that can silently corrupt results — get it wrong and
  every box shifts by the crop offset, which looks exactly like a mis-drawn doorway line —
  hence its own module and tests.
* **Parallel face embedding** — per-person work is independent, so it runs on a thread pool
  (onnxruntime releases the GIL). Tracking stays sequential because it must.
* **`allowed_modules=['detection', 'recognition']`** — `buffalo_l` also ships 3D landmarks
  (137 MB), 2D landmarks and gender/age, and `FaceAnalysis.get()` ran all of them on every
  face.
* **`tools/export_onnx.py`** — exports YOLO to ONNX, which avoids the NNPACK fallback
  entirely and is usually faster on this class of CPU.

`GatedTracker` composes the gate and the crop around any `Detector`, so the pipeline itself
stays ignorant of all of it.

**Still unmeasured:** ArcFace recognition (`w600k_r50`, 166 MB) has never run, because no
frame with a face in it has reached the server yet. Per-person cost will be higher than the
219 ms detection figure. Also unverified: the `det_size` reduction to 480, which landed at
the same time as the module restriction, so their contributions are not separable.

## One camera per direction

A camera at a doorway sees faces going one way and the backs of heads going the other. With
a camera on each side, **both cameras see every passage** — so each is given the one
direction it can actually judge (`announce = "in" | "out"`) and ignores the other. Two
consequences follow, and they are the whole reason for the arrangement:

* **No passage is counted twice**, without any cross-camera matching. Comparing tracks
  between two views is the hard problem in multi-camera vision; assigning directions avoids
  it completely. Track ids stay meaningless across cameras and nothing tries to reconcile
  them.
* **Exits get named by a face**, exactly as entries do. The body-embedding exit match still
  exists for a single-camera deployment, but with two cameras it is no longer load-bearing —
  which removes the clothing-dependence that broke the first system.

The filter is applied *before* the clip is opened. Filtering after publication would still
cut, encode and send a second video showing the back of someone's head.

**The doorway line belongs to the camera, not the room.** Each has its own view, so its own
pixel coordinates and its own idea of which side is inside — `inside_side` **flips** between
opposed cameras. Copying one camera's block to the other and forgetting that makes it label
entries as exits, and the symptom (occupancy running backwards) looks like a logic bug
rather than a config error.

**Shared exactly once, each internally locked:** the face gallery (a face enrolled from one
camera must be recognised by the other), the occupancy ledger (one room, one truth), the
event store, and the review queue. Everything else — tracker, sessions, motion gate, clip
recorder, publisher — is per camera. Each camera runs its own pipeline in its own thread.

**Every event records which camera saw it**, so a drifting count can be traced to the
camera responsible instead of guessed at. Databases written before this migrate in place;
recreating them would throw away the history that is the point of keeping them.

**An exit that cannot be attributed is now recorded as `unknown`** rather than dropped.
Silently writing nothing meant occupancy only ever grew — the count drifted upward
permanently and no amount of walking out could correct it.

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
