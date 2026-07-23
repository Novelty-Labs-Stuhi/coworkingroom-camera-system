#pragma once
#include "esp_camera.h"

// Encode the frame to JPEG and HTTP-POST it to the server endpoint.
void sendPhoto(camera_fb_t *frame);
