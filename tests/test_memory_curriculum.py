import importlib.util
from pathlib import Path
import sys

import numpy as np
from PIL import Image


MODULE_PATH = Path(__file__).parents[1] / "vla" / "datasets" / "memory_curriculum.py"
SPEC = importlib.util.spec_from_file_location("memory_curriculum", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_occlusion_starts_after_visible_history_and_reaches_full_mask():
    cfg = MODULE.MemoryCurriculumConfig(
        enabled=True,
        curriculum_type="occlusion",
        probability=1.0,
        start_ratio=0.4,
        duration_ratio=0.2,
        strength="full",
    )
    image = Image.fromarray(np.full((32, 32, 3), 255, dtype=np.uint8))
    visible, visible_flag, _ = MODULE.apply_memory_curriculum(
        image, {"epis_idx": np.asarray([7]), "episode_position": 3, "episode_length": 10}, cfg
    )
    hidden, hidden_flag, severity = MODULE.apply_memory_curriculum(
        image, {"epis_idx": np.asarray([7]), "episode_position": 9, "episode_length": 10}, cfg
    )
    assert not visible_flag
    assert np.asarray(visible).min() == 255
    assert hidden_flag and severity == 1.0
    assert np.asarray(hidden).max() == 0


def test_episode_assignment_is_deterministic():
    cfg = MODULE.MemoryCurriculumConfig(enabled=True, curriculum_type="occlusion", probability=0.5, seed=9)
    assert [cfg.episode_is_occluded(i) for i in range(20)] == [
        cfg.episode_is_occluded(i) for i in range(20)
    ]


def test_occlusion_can_recover_after_full_mask():
    cfg = MODULE.MemoryCurriculumConfig(
        enabled=True,
        curriculum_type="occlusion",
        probability=1.0,
        start_ratio=0.3,
        duration_ratio=0.2,
        recovery_ratio=0.75,
        strength="full",
    )
    image = Image.fromarray(np.full((32, 32, 3), 255, dtype=np.uint8))
    full, full_flag, full_strength = MODULE.apply_memory_curriculum(
        image, {"epis_idx": np.asarray([7]), "episode_position": 6, "episode_length": 11}, cfg
    )
    recovered, recovered_flag, recovered_strength = MODULE.apply_memory_curriculum(
        image, {"epis_idx": np.asarray([7]), "episode_position": 9, "episode_length": 11}, cfg
    )
    assert full_flag and full_strength == 1.0 and np.asarray(full).max() == 0
    assert not recovered_flag and recovered_strength == 0.0
    assert np.asarray(recovered).min() == 255
