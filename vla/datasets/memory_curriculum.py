"""Deterministic, episode-aware image corruptions for temporal-memory training."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
from PIL import Image


def _episode_uniform(episode_id: int, seed: int) -> float:
    digest = hashlib.blake2b(f"{seed}:{episode_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") / float(2**64)


@dataclass(frozen=True)
class MemoryCurriculumConfig:
    enabled: bool = False
    curriculum_type: str = "normal"
    probability: float = 0.5
    start_ratio: float = 0.4
    duration_ratio: float = 0.2
    recovery_ratio: Optional[float] = None
    strength: str = "full"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.curriculum_type not in {"normal", "occlusion"}:
            raise ValueError(f"Unsupported memory curriculum: {self.curriculum_type}")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        if not 0.0 < self.start_ratio < 1.0:
            raise ValueError("start_ratio must be in (0, 1)")
        if not 0.0 <= self.duration_ratio <= 1.0:
            raise ValueError("duration_ratio must be in [0, 1]")
        if self.recovery_ratio is not None:
            if not 0.0 < self.recovery_ratio <= 1.0:
                raise ValueError("recovery_ratio must be in (0, 1]")
            if self.recovery_ratio <= self.start_ratio + self.duration_ratio:
                raise ValueError("recovery_ratio must follow the occlusion ramp")
        if self.strength not in {"light", "partial", "full"}:
            raise ValueError("strength must be light, partial, or full")

    def episode_is_occluded(self, episode_id: int) -> bool:
        return self.enabled and self.curriculum_type == "occlusion" and (
            _episode_uniform(episode_id, self.seed) < self.probability
        )


def apply_memory_curriculum(
    image: Image.Image,
    sample: Mapping[str, Any],
    cfg: MemoryCurriculumConfig,
) -> tuple[Image.Image, bool, float]:
    """Apply a deterministic late-episode mask and return image, flag, severity.

    The episode manifest injects ``episode_position`` and ``episode_length``.
    A full-strength corruption replaces the complete current RGB observation,
    making recovery of object position from that frame impossible.
    """
    if not cfg.enabled or cfg.curriculum_type == "normal":
        return image, False, 0.0

    episode_id = int(np.asarray(sample["epis_idx"]).reshape(-1)[0])
    if not cfg.episode_is_occluded(episode_id):
        return image, False, 0.0
    if "episode_position" not in sample or "episode_length" not in sample:
        raise ValueError("Episode-aware occlusion requires an episode manifest")

    position = int(sample["episode_position"])
    length = max(1, int(sample["episode_length"]))
    progress = position / max(1, length - 1)
    if progress < cfg.start_ratio:
        return image, False, 0.0
    if cfg.recovery_ratio is not None and progress >= cfg.recovery_ratio:
        return image, False, 0.0

    if cfg.duration_ratio == 0:
        ramp = 1.0
    else:
        ramp = min(1.0, (progress - cfg.start_ratio) / cfg.duration_ratio)
    maximum = {"light": 0.30, "partial": 0.65, "full": 1.0}[cfg.strength]
    severity = maximum * max(ramp, 1.0 / max(1, length))

    array = np.asarray(image, dtype=np.uint8).copy()
    height, width = array.shape[:2]
    if severity >= 0.999:
        array[...] = 0
    else:
        mask_h = max(1, int(round(height * np.sqrt(severity))))
        mask_w = max(1, int(round(width * np.sqrt(severity))))
        y0 = (height - mask_h) // 2
        x0 = (width - mask_w) // 2
        array[y0 : y0 + mask_h, x0 : x0 + mask_w] = 0
    return Image.fromarray(array), True, float(severity)
