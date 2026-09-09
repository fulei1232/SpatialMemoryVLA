"""Spatial representation alignment utilities for SpatialMemoryVLA.

The projector deliberately lives outside ``prismatic``: it is a trainable
MemoryVLA extension, whereas VGGT is an external, frozen training-only
teacher.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAlignProjector(nn.Module):
    """Project VLA visual hidden states into the VGGT feature space."""

    def __init__(
        self,
        llm_dim: int = 4096,
        teacher_dim: int = 2048,
        use_vlm_norm: bool = False,
    ) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.teacher_dim = teacher_dim
        self.vlm_norm = nn.LayerNorm(llm_dim) if use_vlm_norm else nn.Identity()
        self.fc1 = nn.Linear(llm_dim, teacher_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(teacher_dim, teacher_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def align_dimension(self, vision_hidden: torch.Tensor) -> torch.Tensor:
        if vision_hidden.ndim != 3 or vision_hidden.shape[-1] != self.llm_dim:
            raise ValueError(
                f"Expected visual hidden states [B,N,{self.llm_dim}], got {tuple(vision_hidden.shape)}"
            )
        return self.fc2(self.act(self.fc1(self.vlm_norm(vision_hidden))))

    def alignment_loss(self, projected: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if projected.shape != target.shape:
            raise ValueError(f"Spatial feature shapes must match: {tuple(projected.shape)} != {tuple(target.shape)}")
        return (1.0 - F.cosine_similarity(projected.float(), target.float(), dim=-1)).mean()

    def forward(self, vision_hidden: torch.Tensor, target: torch.Tensor | None = None):
        projected = self.align_dimension(vision_hidden)
        return (projected, self.alignment_loss(projected, target)) if target is not None else projected


def extract_visual_hidden(
    hidden_states: Sequence[torch.Tensor], layer_idx: int, num_patches: int, *, start_idx: int = 1
) -> torch.Tensor:
    """Extract the contiguous visual-token block from Prismatic LLM states."""
    try:
        hidden = hidden_states[layer_idx]
    except IndexError as exc:
        raise ValueError(f"spatial_align_layer={layer_idx} is unavailable ({len(hidden_states)} hidden states)") from exc
    visual = hidden[:, start_idx : start_idx + num_patches, :]
    if visual.shape[1] != num_patches:
        raise ValueError(f"Expected {num_patches} visual tokens, got {visual.shape[1]}")
    return visual
