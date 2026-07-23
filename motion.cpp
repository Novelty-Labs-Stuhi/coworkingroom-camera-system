#include "motion.h"
#include <Arduino.h>

// --- Tuning knobs -----------------------------------------------------------
// How much a single pixel's brightness (0-255) must change to "count".
static const int PIXEL_DIFF_THRESHOLD = 25;
// What fraction of all pixels must change before we call it motion (0.05 = 5%).
static const float MOTION_FRACTION = 0.05f;
// ---------------------------------------------------------------------------

// Copy of the previous frame, kept in PSRAM between loop() iterations.
static uint8_t *prevFrame = nullptr;
static size_t prevLen = 0;

bool detectMotion(camera_fb_t *frame) {
  if (!frame || frame->format != PIXFORMAT_GRAYSCALE) {
    return false;
  }

  // First frame (or size changed): store it and report no motion yet.
  if (prevFrame == nullptr || prevLen != frame->len) {
    if (prevFrame) free(prevFrame);
    prevFrame = (uint8_t *)ps_malloc(frame->len);
    if (!prevFrame) {
      Serial.println("prevFrame alloc failed (no PSRAM?)");
      return false;
    }
    memcpy(prevFrame, frame->buf, frame->len);
    prevLen = frame->len;
    return false;
  }

  // Count how many pixels changed by more than the threshold.
  uint32_t changed = 0;
  for (size_t i = 0; i < frame->len; i++) {
    int d = (int)frame->buf[i] - (int)prevFrame[i];
    if (d < 0) d = -d;  // absolute difference
    if (d > PIXEL_DIFF_THRESHOLD) changed++;
  }

  // Remember this frame for next time.
  memcpy(prevFrame, frame->buf, frame->len);

  float fraction = (float)changed / (float)frame->len;
  Serial.printf("motion fraction: %.3f\n", fraction);
  return fraction > MOTION_FRACTION;
}
