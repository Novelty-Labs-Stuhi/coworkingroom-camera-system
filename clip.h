#pragma once
#include <stddef.h>
#include <stdint.h>

// How many JPEG frames we can hold for a single clip. At ~10 fps over 1.5 s we
// expect ~15; the cap is a safety limit so a slow loop can't overrun the arrays.
static const int CLIP_MAX_FRAMES = 24;

// One short burst of motion, held in PSRAM as already-encoded JPEG frames.
struct Clip {
  uint8_t *jpeg[CLIP_MAX_FRAMES];  // each frame2jpg() buffer (we own/free them)
  size_t len[CLIP_MAX_FRAMES];     // byte length of each JPEG
  int count;                       // how many frames were actually captured
  int fps;                         // measured capture rate, for the server's video
};

// Capture a ~1.5 s burst of JPEG frames into `clip`. Returns the frame count.
int recordClip(Clip &clip);

// Free every JPEG buffer the clip holds and reset its count.
void freeClip(Clip &clip);
