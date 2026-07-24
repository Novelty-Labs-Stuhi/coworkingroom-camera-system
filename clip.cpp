#include "clip.h"
#include "esp_camera.h"
#include <Arduino.h>

// --- Tuning knobs -----------------------------------------------------------
// How long a motion clip lasts. We grab frames as fast as the loop allows until
// this much wall-clock time has passed (or we hit CLIP_MAX_FRAMES).
static const unsigned long CLIP_DURATION_MS = 1500;
// JPEG quality (0-100) for each software-encoded frame.
static const int JPEG_QUALITY = 80;
// ---------------------------------------------------------------------------

int recordClip(Clip &clip) {
  clip.count = 0;
  int captured = 0;
  unsigned long start = millis();

  while (millis() - start < CLIP_DURATION_MS && captured < CLIP_MAX_FRAMES) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      delay(10);
      continue;
    }

    // The sensor gives us a raw grayscale frame; encode it to JPEG in software
    // (the OV7725 has no hardware JPEG). frame2jpg allocates the buffer we keep.
    uint8_t *jpg = nullptr;
    size_t jlen = 0;
    if (frame2jpg(fb, JPEG_QUALITY, &jpg, &jlen)) {
      clip.jpeg[captured] = jpg;
      clip.len[captured] = jlen;
      captured++;
    }
    esp_camera_fb_return(fb);
  }

  clip.count = captured;

  // Report the rate we actually achieved so the server plays the clip at real speed.
  unsigned long elapsed = millis() - start;
  clip.fps = elapsed > 0 ? (int)((captured * 1000UL) / elapsed) : captured;
  if (clip.fps < 1) clip.fps = 1;

  Serial.printf("Recorded clip: %d frames @ ~%d fps\n", captured, clip.fps);
  return captured;
}

void freeClip(Clip &clip) {
  for (int i = 0; i < clip.count; i++) {
    free(clip.jpeg[i]);  // frame2jpg allocated these
  }
  clip.count = 0;
}
