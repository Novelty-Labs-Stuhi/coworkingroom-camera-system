#include <Arduino.h>
#include "esp_camera.h"

// Our own modules (one feature each)
#include "camera.h"    // initCamera()
#include "netwifi.h"   // connectWiFi()
#include "motion.h"    // detectMotion()
#include "uploader.h"  // sendPhoto()

// Don't send more often than this, even if motion keeps going (milliseconds).
static const unsigned long COOLDOWN_MS = 10000;  // 10 seconds
static unsigned long lastSend = 0;

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  if (!initCamera()) {
    Serial.println("Camera init failed. Halting.");
    return;  // nothing else can run without the camera
  }
  connectWiFi();
  Serial.println("Ready. Watching for motion...");
}

void loop() {
  // Grab one frame from the camera.
  camera_fb_t *frame = esp_camera_fb_get();
  if (!frame) {
    Serial.println("Frame capture failed");
    delay(100);
    return;
  }

  // Ask the motion module whether this frame differs enough from the last one.
  if (detectMotion(frame)) {
    unsigned long now = millis();
    if (now - lastSend > COOLDOWN_MS) {
      Serial.println("Motion detected -> sending photo");
      sendPhoto(frame);
      lastSend = now;
    }
  }

  // Give the frame buffer back to the driver, then pace the loop (~10 fps).
  esp_camera_fb_return(frame);
  delay(100);
}
