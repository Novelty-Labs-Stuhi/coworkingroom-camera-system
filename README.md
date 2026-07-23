# coworkingroom-camera-system

Doorway camera for the stuhi office: an **ESP32-CAM detects motion and uploads photos**,
and a **server identifies who was seen** using face recognition.

```
[ESP32-CAM] --motion? POST JPEG--> [server.py /upload] --> save photo
 (firmware,                                             --> detect person + recognise face
  root of repo)                                         --> log sighting (who, when)
                                                        GET /sightings -> the log
```

## Two parts

### 1. Camera firmware (repo root — Arduino/ESP32)
Grayscale QVGA motion detector. On motion (with a cooldown) it JPEG-encodes the frame
and POSTs it to the server. Files: `esp32cam.ino`, `camera.*`, `motion.*`, `netwifi.*`,
`uploader.*`, `board_config.h`, `camera_pins.h`, `partitions.csv`.

- Copy `secrets.h.example` → `secrets.h` and set WiFi credentials.
- Set the server URL in `uploader.cpp`.
- Select your board in `board_config.h`, flash with the Arduino IDE.

### 2. Vision server (`vision/` + `server.py` — Python)
`server.py` is a FastAPI receiver. On each uploaded photo it saves the JPEG, detects
people (YOLO), recognises faces against an enrolled gallery (InsightFace/ArcFace), and
logs each **sighting**. The vision engine lives in `vision/` (the `stuhi_vision`
package; it also contains a fuller video/stream pipeline for a future streaming camera).

```bash
pip install -e vision[server]                          # engine + FastAPI/uvicorn
python -m stuhi_vision enroll "Ilari" faces/ilari_*.jpg  # build ./gallery/
uvicorn server:app --host 0.0.0.0 --port 3400          # 0.0.0.0 + port 3400 = the camera's upload target
curl localhost:3400/sightings                          # recent identifications
```

## What it does and does not do

- **Does:** for every motion photo, say whether a person is present and — when the face
  is clear enough — name them (else `unknown`); keep a timestamped log.
- **Does not:** track in/out direction or count occupancy. The camera sends single
  motion-triggered **grayscale QVGA** frames, so there is no video to track across and
  no direction. Recognition quality is limited by that small, grayscale image.

The `vision/` engine *does* contain a full occupancy pipeline (tracking, doorway
line-crossing, in/out ledger) for the day a **streaming** camera is used; see
`vision/docs/design.md`.
