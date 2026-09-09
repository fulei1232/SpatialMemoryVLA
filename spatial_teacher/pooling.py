"""Pooling VGGT patch features to the VLA patch grid."""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def interpolate_pooling(hidden: torch.Tensor, target_tokens: int) -> torch.Tensor:
    """Bilinearly resample ``[B,S,P,D]`` VGGT tokens to ``[B,target_tokens,D]``.

    Frames are concatenated in token order, matching the one-camera first
    version of SpatialMemoryVLA.  The target grid must be square (256 = 16²).
    """
    if hidden.ndim != 4:
        raise ValueError(f"Expected VGGT features [B,S,P,D], got {tuple(hidden.shape)}")
    batch, frames, patches, dim = hidden.shape
    source_hw, target_hw = math.isqrt(patches), math.isqrt(target_tokens)
    if source_hw * source_hw != patches or target_hw * target_hw != target_tokens:
        raise ValueError("VGGT and VLA token counts must represent square patch grids")
    x = hidden.reshape(batch * frames, source_hw, source_hw, dim).permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(target_hw, target_hw), mode="bilinear", align_corners=True)
    return x.permute(0, 2, 3, 1).reshape(batch, frames * target_tokens, dim)


def custom_pooling(hidden: torch.Tensor, target_tokens: int) -> torch.Tensor:
    """Compatibility name for the Spatial-Forcing bilinear pooling operation."""
    return interpolate_pooling(hidden, target_tokens)
