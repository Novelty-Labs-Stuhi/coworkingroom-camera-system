"""Body appearance embedding -- the same-day link that resolves exits.

A person walking out shows the back of their head, so the face signal is gone. But
within a single visit their clothes do not change, so a whole-body appearance vector
captured at entry still matches them at exit. This is deliberately *not* a cross-day
identity (clothes change tomorrow) -- it only links today's exit to today's entry.

Backed by a pretrained torchvision backbone with its classifier removed, giving a
generic appearance feature. It sits behind the same ``embed`` shape as the face
recogniser, so a stronger re-ID model (e.g. OSNet) can replace it without touching
callers. Loaded lazily to keep torch out of the pure logic and tests.
"""

from __future__ import annotations

import numpy as np

from ..domain import Box
from .embeddings import normalize

_INPUT_SIZE = (256, 128)  # (height, width) -- the usual person-crop aspect ratio


class BodyEmbedder:
    """Turn a person crop into an L2-normalised appearance vector."""

    def __init__(self) -> None:
        self._model = None
        self._transform = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from torchvision import transforms
        from torchvision.models import ResNet50_Weights, resnet50

        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        backbone.fc = torch.nn.Identity()  # keep the 2048-d feature, drop the classifier
        backbone.eval()
        self._model = backbone
        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(_INPUT_SIZE, antialias=True),
                transforms.Normalize(
                    mean=ResNet50_Weights.IMAGENET1K_V2.transforms().mean,
                    std=ResNet50_Weights.IMAGENET1K_V2.transforms().std,
                ),
            ]
        )

    def embed(self, image: np.ndarray, box: Box) -> np.ndarray | None:
        crop = box.crop(image)
        if crop.size == 0:
            return None
        self._ensure_loaded()
        import torch

        from ..preprocessing import normalize_lighting

        crop = normalize_lighting(crop)  # even out exposure so lighting doesn't drift the vector
        rgb = crop[:, :, ::-1].copy()  # OpenCV BGR -> RGB
        tensor = self._transform(rgb).unsqueeze(0)
        with torch.no_grad():
            features = self._model(tensor).squeeze(0).cpu().numpy()
        return normalize(features.astype(np.float32))
