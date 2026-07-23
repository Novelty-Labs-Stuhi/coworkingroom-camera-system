#include "uploader.h"
#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>

// >>> EDIT THIS ONE LINE <<<  Point it at your Mac's IP and the server port.
// Find your Mac's IP with:  ipconfig getifaddr en0
static const char *SERVER_URL = "http://192.168.8.203:8000/upload";

void sendPhoto(camera_fb_t *frame) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi not connected, skipping upload");
    return;
  }

  // The camera gives us a raw grayscale frame; encode it to JPEG in software.
  uint8_t *jpg = nullptr;
  size_t jpgLen = 0;
  if (!frame2jpg(frame, 80 /* quality 0-100 */, &jpg, &jpgLen)) {
    Serial.println("JPEG encode failed");
    return;
  }

  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "image/jpeg");
  int code = http.POST(jpg, jpgLen);
  if (code > 0) {
    Serial.printf("Upload HTTP %d (%u bytes)\n", code, (unsigned)jpgLen);
  } else {
    Serial.printf("Upload failed: %s\n", http.errorToString(code).c_str());
  }
  http.end();

  free(jpg);  // frame2jpg allocated this buffer; we own it
}
