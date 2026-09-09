"""External frozen VGGT teacher for Geometry-Aligned Temporal Memory."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

import torch
import torch.nn as nn

from .pooling import custom_pooling
from .preprocess import preprocess_memoryvla_images


class SpatialTeacher(nn.Module):
    """Frozen, checkpointed VGGT feature extractor kept outside MemoryVLA."""

    def __init__(self, vggt: nn.Module, feature_layer: int = -1, target_tokens: int = 256) -> None:
        super().__init__()
        self.vggt = vggt
        self.feature_layer = feature_layer
        self.target_tokens = target_tokens
        self.vggt.requires_grad_(False)
        self.vggt.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, **kwargs: Any) -> "SpatialTeacher":
        # Spatial-Forcing's vendored VGGT source uses absolute ``vggt.*``
        # imports.  Expose this package directory at import time without
        # modifying upstream files.
        vendor_root = str(Path(__file__).resolve().parent)
        if vendor_root not in sys.path:
            sys.path.insert(0, vendor_root)
        try:
            from vggt.models.vggt import VGGT
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "VGGT sources must be vendored at spatial_teacher/vggt before enabling spatial forcing. "
                "They are available in /home/Spatial-Forcing/openvla-SF/vggt."
            ) from exc
        vggt = VGGT(enable_camera=False, enable_point=False, enable_depth=False, enable_track=False, feature_only=True)
        state = torch.load(Path(checkpoint_path), map_location="cpu")
        vggt.load_state_dict(state.get("state_dict", state), strict=False)
        return cls(vggt, **kwargs)

    def train(self, mode: bool = True):  # teacher must remain evaluation-only
        super().train(False)
        self.vggt.eval()
        return self

    @torch.no_grad()
    def forward(self, pixel_values) -> torch.Tensor:
        images = preprocess_memoryvla_images(pixel_values).to(next(self.vggt.parameters()).device)
        output = self.vggt(images)
        features = output["features"][self.feature_layer]
        patch_start_idx = output["patch_start_idx"]
        return custom_pooling(features[:, :, patch_start_idx:, :], self.target_tokens)
