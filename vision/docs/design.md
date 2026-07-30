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

## Recognising a passage: the doorframe, not a line

The original rule drew a line across the doorway and watched a person's foot point cross it.
It failed repeatedly against this doorway, and the reasons are worth keeping:

* **The foot point sat pinned to the bottom edge of the frame.** People pass close to the
  camera, so their feet leave the picture. A horizontal line was therefore never crossed —
  measured foot y-values were 477–480 out of 480 for every frame of a passage.
* **The line had to be redrawn from a still whenever a camera moved**, and it is drawn in
  pixels, so it silently stops matching.
* **`inside_side` reversed the count when set wrongly**, and the symptom — occupancy running
  backwards — looks like a logic bug rather than a configuration one.

The replacement uses a sturdier fact about the scene, and it came from watching the footage:
**the doorframe is visible, and a person passing through occludes it.** Their pixels are in
front of the frame; somebody merely moving in the corridor beyond is seen *through* the
opening and never overlaps it. So a *finished* track is judged by where it began and ended:

| Track | Verdict |
|---|---|
| touched the doorframe zone, **last** seen at the far edge | passed through, `passing_means` |
| touched the zone, **first** seen at that edge | came from beyond the door, the opposite |
| never touched the zone, or never tall enough | background traffic — ignored |
| both or neither | stepped in and back out — not a passage |

Three properties matter more than the specific thresholds:

**The decision is deferred until the track ends.** Whether somebody passed *through* is only
knowable once they stop being visible — and that disappearance is precisely the signal. This
is why the detector owns track lifetimes and emits nothing while a person is still in view.

**Nothing is a hairline.** The zone is a broad strip and the edge has a margin, so a knocked
camera degrades gradually rather than silently counting nothing.

**Track ids do the work of separating people.** No cross-camera identity matching is needed,
because each camera judges only its own direction.

The same shape of rule serves a room-facing camera with `edge = "bottom"`: it sits at the
door looking inward, so somebody leaving walks toward the lens and out of the bottom of the
frame. There is no doorframe in that view to occlude, so `min_height` does the work of
separating a person at the door from people seated across the room — that view has a person
in *every* frame, which is why plain presence is useless there.

`tools/replay_threshold.py` replays recorded frames through the rule with the parameters on
the command line. Tuning it against the live system is impractical: each attempt costs
somebody a walk down the corridor, and a negative result says only "nothing happened".

## One camera counts, the other names

A camera at a doorway sees faces going one way and the backs of heads going the other, and
**both cameras see every passage** — so exactly one of them may commit it. `role` settles
which, and the split follows what each camera is actually good at rather than being symmetric:

* **The doorway camera counts** (`role = "count"`), in both directions. It is the only one
  that can see the doorframe, which is what separates a passage from somebody crossing the
  room behind it. It is a poor witness to *who*: it watches leavers from behind.
* **The room camera names** (`role = "identify"`) and commits nothing. A leaver walks straight
  at its lens, face first, so it recognises them outright. It is a poor judge of *whether*
  anyone passed, because everyone in its view sits at the near edge the whole time.

The handover is a name and a timestamp (`witness.LeavingWitness`), nothing more — the
recogniser has already decided who somebody is, and a second opinion here would mean two
answers to one question. The counting camera claims the most recent name when its own face
match came up empty. **A claim consumes the name**, so two people leaving one after another
cannot both be recorded as the first, and a name expires after fifteen seconds rather than
surviving to mislabel some later exit.

Evidence for an exit is used in order of how direct it is: a face seen by the counting camera,
then the other camera's name, then the body embedding. The body-embedding match still exists
for a single-camera deployment but is no longer load-bearing — which removes the
clothing-dependence that broke the first system.

**Nothing matches tracks between views.** Comparing tracks across two cameras is the hard
problem in multi-camera vision; a role per camera avoids it entirely. Track ids stay
meaningless across cameras and nothing tries to reconcile them.

`announce` is a narrower question — which directions a camera *films and sends* — and is
applied before the clip is opened, so a filtered direction costs no encoding. The doorway
camera films arrivals only: its exit clip would show the back of a head, while the room camera
films that same exit face-first, which is the clip worth labelling.

**The doorway line belongs to the camera, not the room.** Each has its own view, so its own
pixel coordinates and its own idea of which side is inside — `inside_side` **flips** between
opposed cameras. Copying one camera's block to the other and forgetting that makes it label
entries as exits, and the symptom (occupancy running backwards) looks like a logic bug
rather than a config error.

**Shared exactly once, each internally locked:** the face gallery (a face enrolled from one
camera must be recognised by the other), the occupancy ledger (one room, one truth), the
event store, the review queue, and the leaving witness (a message from one camera to the
other). Everything else — tracker, sessions, motion gate, clip
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
