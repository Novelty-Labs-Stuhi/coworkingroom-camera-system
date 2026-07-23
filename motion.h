#pragma once
#include "esp_camera.h"

// Compare this grayscale frame against the previous one.
// Returns true when enough pixels changed to count as motion.
// The first call just stores the frame and returns false.
bool detectMotion(camera_fb_t *frame);
