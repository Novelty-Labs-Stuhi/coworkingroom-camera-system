# Deployment & operations

How the system is deployed on the office server, and how to run and calibrate it.

## The server

- Host: `stuhinet` (user `markp`), reached as `ssh ssh.porkkanat.com` via a Cloudflare
  Access tunnel. Key-based auth.
- Ubuntu 24.04, Python 3.12, **CPU only (no GPU)**, and a **locked-down** box: no
  passwordless `sudo`, and `python3-venv`/`ensurepip` are not installed.
- Project lives at `~/stuhi_vision`, with its virtualenv at `~/stuhi_vision/.venv`.

## Install notes (why it isn't a plain `pip install`)

Three environment quirks were worked around (see `install.sh` for the exact steps):

1. **No `ensurepip`** → the venv is created with `python3 -m venv --without-pip`, then
   pip is bootstrapped with `get-pip.py`.
2. **`libGL` missing and no sudo** → OpenCV must be the **headless** build, and pinned
   below 5 (`opencv-python-headless>=4.9,<5`) because the opencv-5 headless wheel still
   links `libGL`.
3. **No GPU** → `torch` is the CPU wheel; `face.py` selects the CPU ONNX runtime
   automatically. Expect reduced frame rate on live streams (fine for a doorway).

Everything runs inside the venv: `~/stuhi_vision/.venv/bin/python -m stuhi_vision ...`

## Updating the code

There is no git remote; the repo is shipped as a clean archive:

```bash
# on the dev machine
git archive --format=tar.gz -o /tmp/sv.tgz HEAD
scp /tmp/sv.tgz ssh.porkkanat.com:~/sv.tgz
ssh ssh.porkkanat.com "tar xzf ~/sv.tgz -C ~/stuhi_vision && rm ~/sv.tgz"
```

The venv and any `data/`, `gallery/` are outside the archive, so they are preserved.

## Running

```bash
cd ~/stuhi_vision
.venv/bin/python -m stuhi_vision run          # process the configured source
.venv/bin/python -m stuhi_vision occupancy    # who is inside right now
```

## Calibrating without a live camera

Because there is no camera attached, place the doorway line and check detection on a
**recorded clip** (any short video of the doorway works):

```bash
# overlay the current line on one frame -> a PNG you can eyeball
.venv/bin/python -m stuhi_vision calibrate --source clip.mp4 -o doorway.png

# full annotated video: boxes, track ids, the line, and in/out events burned in
.venv/bin/python -m stuhi_vision review --source clip.mp4 -o annotated.mp4
```

Adjust `line_a` / `line_b` in `config.toml` until the line sits across the threshold and
`review` shows `in`/`out` firing at the right moments. `review` writes to a throwaway
database, so it never affects real occupancy counts.

## Enrolling faces

```bash
.venv/bin/python -m stuhi_vision enroll "Ilari" photos/ilari_*.jpg
```

A few clear, roughly-frontal photos per person. Until someone is enrolled, their entries
are labelled `guest-N` (still counted, just unnamed).

## Health check

```bash
.venv/bin/python -m pytest -q          # 19 pure-logic tests, no models needed
```
