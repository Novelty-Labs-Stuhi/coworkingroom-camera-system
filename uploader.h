#pragma once
#include "clip.h"

// HTTP-POST a recorded clip to the server. The body is a length-prefixed
// stream of JPEG frames: for each frame, a little-endian uint32 length
// followed by that many JPEG bytes. The server stitches them into a video.
void sendClip(const Clip &clip);
