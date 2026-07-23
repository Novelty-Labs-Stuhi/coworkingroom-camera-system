#pragma once
#include "esp_camera.h"

// Initialise the camera in grayscale (small, fast) mode for motion detection.
// Returns true on success.
bool initCamera();
