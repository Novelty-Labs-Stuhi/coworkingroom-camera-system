#include "netwifi.h"
#include <Arduino.h>
#include <WiFi.h>
#include "secrets.h"  // WIFI_SSID / WIFI_PASSWORD (gitignored; see secrets.h.example)

// Your WiFi credentials come from secrets.h.
static const char *ssid = WIFI_SSID;
static const char *password = WIFI_PASSWORD;

void connectWiFi() {
  WiFi.begin(ssid, password);
  WiFi.setSleep(false);

  Serial.print("WiFi connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("WiFi connected. IP: ");
  Serial.println(WiFi.localIP());
}
