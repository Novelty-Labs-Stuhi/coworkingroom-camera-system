#include "uploader.h"
#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>

// >>> EDIT THIS ONE LINE <<<  Point it at the server's IP and port.
// Port 3400 is inside Arsenii's ufw-allowed range (3400-3499) on Mark's PC.
static const char *SERVER_URL = "http://192.168.8.238:3400/clip";

// Write a little-endian uint32 into buf and advance the offset.
static void putU32LE(uint8_t *buf, size_t &off, uint32_t v) {
  buf[off++] = v & 0xff;
  buf[off++] = (v >> 8) & 0xff;
  buf[off++] = (v >> 16) & 0xff;
  buf[off++] = (v >> 24) & 0xff;
}

void sendClip(const Clip &clip) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi not connected, skipping upload");
    return;
  }
  if (clip.count <= 0) {
    Serial.println("Empty clip, nothing to send");
    return;
  }

  // One contiguous body: [len][jpeg][len][jpeg]... Build it in PSRAM so we send
  // the whole clip in a single POST rather than one request per frame.
  size_t total = 0;
  for (int i = 0; i < clip.count; i++) total += 4 + clip.len[i];

  uint8_t *body = (uint8_t *)ps_malloc(total);
  if (!body) {
    Serial.println("clip body alloc failed (no PSRAM?)");
    return;
  }

  size_t off = 0;
  for (int i = 0; i < clip.count; i++) {
    putU32LE(body, off, (uint32_t)clip.len[i]);
    memcpy(body + off, clip.jpeg[i], clip.len[i]);
    off += clip.len[i];
  }

  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/octet-stream");
  http.addHeader("X-Frame-Count", String(clip.count));
  http.addHeader("X-Fps", String(clip.fps));
  int code = http.POST(body, total);
  if (code > 0) {
    Serial.printf("Upload HTTP %d (%d frames, %u bytes)\n", code, clip.count,
                  (unsigned)total);
  } else {
    Serial.printf("Upload failed: %s\n", http.errorToString(code).c_str());
  }
  http.end();

  free(body);
}
