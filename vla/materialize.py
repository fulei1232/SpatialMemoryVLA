"""
materialize.py

Factory class for initializing Open-X RLDS-backed datasets, given specified data mixture parameters; provides and
exports individual functions for clear control flow.
"""

import json
from pathlib import Path
from typing import Tuple, Type, Union

import numpy as np

from transformers import PreTrainedTokenizerBase
from torch.utils.data import Dataset

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from vla.datasets import (
    EpisodicRLDSDataset,
    GroupRLDSDataset,
    RLDSBatchTransform,
    RLDSDataset,
    LiberoRelocationBatchTransform,
    LiberoRelocationNPZDataset,
    RoboMMEBatchTransform,
    RoboMMEPickleDataset,
    StreamRLDSDataset,
)
from vla.action_tokenizer import ActionTokenizer
from vla.datasets.memory_curriculum import MemoryCurriculumConfig


def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    image_aug: bool = False,
    future_action_window_size: int = 15,
    load_all_data_for_training: bool = True,  # Load all data for training, or only a subset
    dataloader_type: str = "group",
    group_size: int = 16,
    seed: int = 42,
    episode_manifest_path: Union[str, Path, None] = None,
    memory_curriculum_enabled: bool = False,
    memory_curriculum_type: str = "normal",
    occlusion_probability: float = 0.5,
    occlusion_start_ratio: float = 0.4,
    occlusion_duration_ratio: float = 0.2,
    occlusion_recovery_ratio: Union[float, None] = None,
    occlusion_strength: str = "full",
) -> Tuple[Dataset, ActionTokenizer, PaddedCollatorForActionPrediction]:
    """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""

    action_tokenizer = ActionTokenizer(tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        tokenizer,
        image_transform,
        prompt_builder_fn,
        predict_stop_token=predict_stop_token,
        memory_curriculum=MemoryCurriculumConfig(
            enabled=memory_curriculum_enabled,
            curriculum_type=memory_curriculum_type,
            probability=occlusion_probability,
            start_ratio=occlusion_start_ratio,
            duration_ratio=occlusion_duration_ratio,
            recovery_ratio=occlusion_recovery_ratio,
            strength=occlusion_strength,
            seed=seed,
        ),
    )

    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side=padding_side,
    )

    if data_mix == "libero_relocation_npz":
        stats = json.loads((Path(data_root_dir) / "action_stats.json").read_text())
        transform = LiberoRelocationBatchTransform(
            base_tokenizer=tokenizer,
            image_transform=image_transform,
            prompt_builder_fn=prompt_builder_fn,
            action_q01=np.asarray(stats["action"]["q01"], dtype=np.float32),
            action_q99=np.asarray(stats["action"]["q99"], dtype=np.float32),
            action_horizon=future_action_window_size + 1,
        )
        dataset = LiberoRelocationNPZDataset(
            data_root_dir=data_root_dir,
            batch_transform=transform,
            seed=seed,
        )
        return dataset, action_tokenizer, collator

    if data_mix == "robomme":
        norm_stats_path = Path(data_root_dir) / "meta" / "norm_stats.json"
        if not norm_stats_path.is_file():
            raise FileNotFoundError(f"Missing RoboMME normalization statistics: {norm_stats_path}")
        raw_stats = json.loads(norm_stats_path.read_text())
        action_stats = raw_stats.get("norm_stats", raw_stats)["actions"]
        robomme_transform = RoboMMEBatchTransform(
            base_tokenizer=tokenizer,
            image_transform=image_transform,
            prompt_builder_fn=prompt_builder_fn,
            action_q01=np.asarray(action_stats["q01"], dtype=np.float32),
            action_q99=np.asarray(action_stats["q99"], dtype=np.float32),
            action_horizon=future_action_window_size + 1,
            image_aug=image_aug,
            memory_curriculum=MemoryCurriculumConfig(
                enabled=memory_curriculum_enabled,
                curriculum_type=memory_curriculum_type,
                probability=occlusion_probability,
                start_ratio=occlusion_start_ratio,
                duration_ratio=occlusion_duration_ratio,
                recovery_ratio=occlusion_recovery_ratio,
                strength=occlusion_strength,
                seed=seed,
            ),
        )
        dataset = RoboMMEPickleDataset(
            data_root_dir=data_root_dir,
            batch_transform=robomme_transform,
            group_size=group_size,
            seed=seed,
            episode_manifest_path=episode_manifest_path,
        )
        return dataset, action_tokenizer, collator

    # Build RLDS Iterable Dataset
    if dataloader_type == "normal":
        dataset = RLDSDataset(
            data_root_dir,
            data_mix,
            batch_transform,
            resize_resolution=default_image_resolution[1:],
            shuffle_buffer_size=shuffle_buffer_size,
            train=train,
            future_action_window_size=future_action_window_size,
            image_aug=image_aug,
            load_all_data_for_training=load_all_data_for_training,
            seed=seed,
        )
    elif dataloader_type == "group":
        assert group_size > 1, "Group size must be greater than 1 for grouped dataset"
        dataset = GroupRLDSDataset(
            data_root_dir,
            data_mix,
            batch_transform,
            resize_resolution=default_image_resolution[1:],
            shuffle_buffer_size=shuffle_buffer_size,
            train=train,
            future_action_window_size=future_action_window_size,
            image_aug=image_aug,
            load_all_data_for_training=load_all_data_for_training,
            group_size=group_size,
            seed=seed,
        )
    elif dataloader_type == "stream":
        dataset = StreamRLDSDataset(
            data_root_dir,
            data_mix,
            batch_transform,
            resize_resolution=default_image_resolution[1:],
            shuffle_buffer_size=shuffle_buffer_size,
            train=train,
            future_action_window_size=future_action_window_size,
            image_aug=image_aug,
            load_all_data_for_training=load_all_data_for_training,
            seed=seed,
        )

    else:
        raise NotImplementedError(f"Dataset type {dataloader_type} not implemented.")

    return dataset, action_tokenizer, collator
