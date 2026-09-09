"""Convert MemoryVLA's DINO-normalized image tensor for VGGT."""

from __future__ import annotations

import torch
import torch.nn.functional as F

_DINO_MEAN = (0.485, 0.456, 0.406)
_DINO_STD = (0.229, 0.224, 0.225)


def preprocess_memoryvla_images(pixel_values, image_size: int = 518) -> torch.Tensor:
    """Return RGB values in ``[0, 1]`` with shape ``[B,1,3,518,518]``.

    MemoryVLA supplies separate DINO/SigLIP tensors.  DINO has ImageNet
    normalization and is therefore inverted before VGGT's own normalization.
    """
    if not isinstance(pixel_values, dict) or "dino" not in pixel_values:
        raise ValueError("Spatial teacher requires MemoryVLA pixel_values['dino']")
    images = pixel_values["dino"].float()
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected DINO images [B,3,H,W], got {tuple(images.shape)}")
    mean = images.new_tensor(_DINO_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(_DINO_STD).view(1, 3, 1, 1)
    images = (images * std + mean).clamp_(0.0, 1.0)
    images = F.interpolate(images, size=(image_size, image_size), mode="bicubic", align_corners=False)
    return images.unsqueeze(1)
