"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

from dataclasses import dataclass
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple, Type
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from torchvision import transforms as tv_transforms
from transformers import PreTrainedTokenizerBase
import tensorflow as tf

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from vla.action_tokenizer import ActionTokenizer
from vla.datasets.rlds import make_interleaved_dataset, make_single_dataset, \
    make_interleaved_episodic_dataset
from vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from vla.datasets.rlds.utils.data_utils import NormalizationType
from vla.datasets.memory_curriculum import MemoryCurriculumConfig, apply_memory_curriculum

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    memory_curriculum: MemoryCurriculumConfig = MemoryCurriculumConfig()

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        # dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        
        # For future action predictions
        if rlds_batch["action"].shape[0] > 1:
            dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"]
        else:
            dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]

        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        img, occlusion_flag, occlusion_strength = apply_memory_curriculum(
            img, rlds_batch, self.memory_curriculum
        )
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        # Construct Chat-based Prompt
        prompt_builder = self.prompt_builder_fn("openvla")

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": ""},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids

        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # Add future actions to batch
        if rlds_batch["action"].shape[0] > 1:
            action = torch.tensor(action, dtype=torch.float32)
            action_mask = None
            if "action_mask" in rlds_batch:
                action_mask = torch.tensor(rlds_batch["action_mask"], dtype=torch.bool)

        # Mask prompt tokens
        labels[: int(torch.where(input_ids==2)[0][0])] = IGNORE_INDEX

        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        timesteps = rlds_batch['observation']['timestep']

        return dict(pixel_values=pixel_values,
                    input_ids=input_ids,
                    labels=labels,
                    dataset_name=dataset_name,
                    actions=action,
                    action_masks=action_mask,
                    timesteps=timesteps,
                    episode_ids=None,
                    occlusion_flags=np.asarray([occlusion_flag], dtype=np.bool_),
                    occlusion_strengths=np.asarray([occlusion_strength], dtype=np.float32),
                    )


@dataclass
class RoboMMEBatchTransform:
    """Convert one official RoboMME preprocessed pickle into MemoryVLA inputs."""

    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    action_q01: np.ndarray
    action_q99: np.ndarray
    action_horizon: int = 16
    image_aug: bool = False
    memory_curriculum: MemoryCurriculumConfig = MemoryCurriculumConfig()

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image = Image.fromarray(np.asarray(sample["image"], dtype=np.uint8))
        if self.image_aug:
            image = tv_transforms.Compose(
                [
                    tv_transforms.RandomResizedCrop(image.size[::-1], scale=(0.9, 0.9), ratio=(1.0, 1.0)),
                    tv_transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                ]
            )(image)
        image, occlusion_flag, occlusion_strength = apply_memory_curriculum(
            image, sample, self.memory_curriculum
        )
        instruction = str(sample["prompt"]).lower()

        prompt_builder = self.prompt_builder_fn("openvla")
        for turn in (
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": ""},
        ):
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = torch.tensor(
            self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        )
        labels = input_ids.clone()
        eos_positions = torch.where(input_ids == 2)[0]
        if len(eos_positions) == 0:
            raise ValueError("RoboMME prompt is missing the expected Llama EOS token (id=2)")
        labels[: int(eos_positions[0])] = IGNORE_INDEX

        actions = np.asarray(sample["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 8:
            raise ValueError(f"Expected RoboMME actions [T,8], got {actions.shape}")
        state = np.asarray(sample["state"], dtype=np.float32)
        if state.shape != (8,):
            raise ValueError(f"Expected RoboMME state [8], got {state.shape}")

        # RoboMME stores absolute joint targets. Its official RoboMMEDataConfig
        # converts the first seven joints to deltas and keeps the gripper
        # command (dimension 8) absolute before applying action normalization.
        actions = actions.copy()
        actions[:, :7] -= state[None, :7]
        if actions.shape[0] < self.action_horizon:
            actions = np.concatenate(
                [actions, np.repeat(actions[-1:], self.action_horizon - actions.shape[0], axis=0)], axis=0
            )
        actions = actions[: self.action_horizon]
        actions = 2.0 * (actions - self.action_q01) / (self.action_q99 - self.action_q01 + 1e-8) - 1.0
        actions = np.clip(actions, -1.0, 1.0)

        return {
            "pixel_values": self.image_transform(image),
            "input_ids": input_ids,
            "labels": labels,
            "dataset_name": "robomme",
            "actions": torch.from_numpy(actions.astype(np.float32, copy=False)),
            "action_masks": torch.ones(self.action_horizon, dtype=torch.bool),
            "timesteps": np.asarray(sample["step_idx"], dtype=np.int64).reshape(1),
            "episode_ids": np.asarray(sample["epis_idx"], dtype=np.int64).reshape(1),
            "occlusion_flags": np.asarray([occlusion_flag], dtype=np.bool_),
            "occlusion_strengths": np.asarray([occlusion_strength], dtype=np.float32),
        }


class RoboMMEPickleDataset(IterableDataset):
    """Grouped, rank-sharded reader for the official RoboMME preprocessed dataset."""

    def __init__(
        self,
        data_root_dir: Path,
        batch_transform: RoboMMEBatchTransform,
        group_size: int = 16,
        seed: int = 42,
        episode_manifest_path: Path | None = None,
    ) -> None:
        self.data_root_dir = Path(data_root_dir)
        self.data_dir = self.data_root_dir / "data"
        stats_path = self.data_root_dir / "meta" / "stats.json"
        if not stats_path.is_file() or not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"RoboMME dataset must contain data/part_*.zip (or data/*.pkl) and "
                f"meta/stats.json under {self.data_root_dir}"
            )
        stats = json.loads(stats_path.read_text())
        self.dataset_length = int(stats.get("execution_samples", stats.get("total_samples", 0)))
        if self.dataset_length <= 0:
            raise ValueError(f"Invalid RoboMME sample count in {stats_path}: {stats}")

        self.archive_paths = sorted(self.data_dir.glob("part_*.zip"))
        self.sample_to_archive = None
        if self.archive_paths:
            sample_to_archive = np.full(self.dataset_length, -1, dtype=np.int16)
            for archive_idx, archive_path in enumerate(self.archive_paths):
                with ZipFile(archive_path) as archive:
                    for member in archive.namelist():
                        if not member.endswith(".pkl"):
                            continue
                        sample_idx = int(Path(member).stem)
                        if 0 <= sample_idx < self.dataset_length:
                            sample_to_archive[sample_idx] = archive_idx
            missing = np.flatnonzero(sample_to_archive < 0)
            if len(missing):
                raise ValueError(
                    f"RoboMME ZIP archives are missing {len(missing)} samples; first missing index is {missing[0]}"
                )
            self.sample_to_archive = sample_to_archive
        elif not (self.data_dir / "0.pkl").is_file():
            raise FileNotFoundError(
                f"No RoboMME part_*.zip archives or extracted pickle samples found in {self.data_dir}"
            )
        self.batch_transform = batch_transform
        self.group_size = group_size
        self.seed = seed
        self.episode_manifest_path = Path(episode_manifest_path) if episode_manifest_path else None
        self.manifest = None
        if self.episode_manifest_path is not None:
            if not self.episode_manifest_path.is_file():
                raise FileNotFoundError(f"RoboMME episode manifest not found: {self.episode_manifest_path}")
            manifest = np.load(self.episode_manifest_path)
            required = {
                "sample_indices", "episode_ids", "timesteps", "episode_positions", "episode_lengths"
            }
            missing_keys = required - set(manifest.files)
            if missing_keys:
                raise ValueError(f"Episode manifest is missing keys: {sorted(missing_keys)}")
            lengths = {key: len(manifest[key]) for key in required}
            if len(set(lengths.values())) != 1:
                raise ValueError(f"Episode manifest arrays have different lengths: {lengths}")
            if lengths["sample_indices"] % self.group_size:
                raise ValueError("Episode manifest length must be divisible by group_size")
            self.manifest = {key: np.asarray(manifest[key]) for key in required}
            if "source_timesteps" in manifest.files:
                if len(manifest["source_timesteps"]) != lengths["sample_indices"]:
                    raise ValueError("source_timesteps has the wrong length")
                self.manifest["source_timesteps"] = np.asarray(manifest["source_timesteps"])
        self.dataset_statistics = {
            "robomme": {
                "action": {
                    "q01": batch_transform.action_q01.tolist(),
                    "q99": batch_transform.action_q99.tolist(),
                }
            }
        }

    def __len__(self) -> int:
        return len(self.manifest["sample_indices"]) if self.manifest is not None else self.dataset_length

    def _load_sample(self, sample_idx: int, open_archives: Dict[int, ZipFile]) -> Dict[str, Any]:
        if self.sample_to_archive is None:
            with (self.data_dir / f"{sample_idx}.pkl").open("rb") as handle:
                return pickle.load(handle)
        archive_idx = int(self.sample_to_archive[sample_idx])
        archive = open_archives.get(archive_idx)
        if archive is None:
            archive = ZipFile(self.archive_paths[archive_idx])
            open_archives[archive_idx] = archive
        with archive.open(f"{sample_idx}.pkl") as handle:
            return pickle.load(handle)

    def __iter__(self):
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        num_groups = (self.dataset_length + self.group_size - 1) // self.group_size
        open_archives = {}
        epoch = 0
        while True:
            active_length = len(self) if self.manifest is not None else self.dataset_length
            num_groups = (active_length + self.group_size - 1) // self.group_size
            groups = np.arange(num_groups)
            np.random.default_rng(self.seed + epoch).shuffle(groups)
            for group_idx in groups[rank::world_size]:
                start = int(group_idx) * self.group_size
                stop = min(start + self.group_size, active_length)
                for manifest_idx in range(start, stop):
                    sample_idx = (
                        int(self.manifest["sample_indices"][manifest_idx])
                        if self.manifest is not None else manifest_idx
                    )
                    sample = self._load_sample(sample_idx, open_archives)
                    if self.manifest is not None:
                        expected_episode = int(self.manifest["episode_ids"][manifest_idx])
                        expected_timestep = int(self.manifest["timesteps"][manifest_idx])
                        source_timestep = int(
                            self.manifest.get("source_timesteps", self.manifest["timesteps"])[manifest_idx]
                        )
                        actual_episode = int(np.asarray(sample["epis_idx"]).reshape(-1)[0])
                        actual_timestep = int(np.asarray(sample["step_idx"]).reshape(-1)[0])
                        if (actual_episode, actual_timestep) != (expected_episode, source_timestep):
                            raise ValueError(
                                f"Manifest/sample mismatch for {sample_idx}: "
                                f"{(expected_episode, source_timestep)} != {(actual_episode, actual_timestep)}"
                            )
                        # Memory timestep is a contiguous index within the retained
                        # execution episode; preserve the source timestep separately.
                        sample["source_step_idx"] = actual_timestep
                        sample["step_idx"] = np.asarray([expected_timestep], dtype=np.int64)
                        sample["episode_position"] = int(self.manifest["episode_positions"][manifest_idx])
                        sample["episode_length"] = int(self.manifest["episode_lengths"][manifest_idx])
                    yield self.batch_transform(sample)
            epoch += 1


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        future_action_window_size: int = 0,
        train: bool = True,
        image_aug: bool = False,
        load_all_data_for_training: bool = True,
        load_depth=False,
        load_proprio=False,
        seed: int = 42,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=load_depth,
            load_proprio=load_proprio,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        # TFDS file shuffling has no exposed seed in dlimp. Disable it and use
        # the explicitly seeded episode/frame shuffle below. Offset by global
        # rank so ranks see distinct but repeatable streams across matched runs.
        rank = int(os.environ.get("RANK", "0"))
        rlds_seed = int(seed) + rank
        tf.random.set_seed(rlds_seed)
        for dataset_kwargs in per_dataset_kwargs:
            dataset_kwargs["shuffle"] = False
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                    # If we wanted to feed / predict more than one step
                future_action_window_size=future_action_window_size,                        # For action chunking
                skip_unlabeled=True,                                                        # Skip trajectories without language labels
                #goal_relabeling_strategy="uniform",                                        # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
            load_all_data_for_training=load_all_data_for_training,
            seed=rlds_seed,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2], # real not, TODO
                random_hue=[0.05], # TODO
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
            # load_all_data_for_training=rlds_config["load_all_data_for_training"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)


class GroupRLDSDataset(RLDSDataset):
    def __init__(self, *args,
                 group_size: int = 16,
                 **kwargs):
        self.group_size = group_size
        super().__init__(*args, **kwargs)

    def make_dataset(self, rlds_config):
        return make_interleaved_episodic_dataset(
            **rlds_config,
            group_size=self.group_size,
            use_optim_group_sample=True,
        )

    def __iter__(self) -> Dict[str, Any]:
        episode_id = -1
        for rlds_batch in self.dataset.as_numpy_iterator():
            episode_id += 1
            indices = range(rlds_batch["action"].shape[0])
            for i in indices:
                raw_frame = tree_map(lambda x: x[i], rlds_batch)
                raw_frame["epis_idx"] = np.asarray([episode_id], dtype=np.int64)
                raw_frame["episode_position"] = i
                raw_frame["episode_length"] = rlds_batch["action"].shape[0]
                frame = self.batch_transform(raw_frame)
                frame["episode_ids"] = np.array([episode_id])
            yield frame


@dataclass
class LiberoRelocationBatchTransform:
    """Convert one frame from a successful relocation rollout into VLA inputs."""

    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    action_q01: np.ndarray
    action_q99: np.ndarray
    action_horizon: int = 16

    def __call__(self, episode: Dict[str, Any], position: int) -> Dict[str, Any]:
        image = Image.fromarray(np.asarray(episode["images"][position], dtype=np.uint8))
        instruction = str(np.asarray(episode["instruction"]).item()).lower()
        prompt_builder = self.prompt_builder_fn("openvla")
        for turn in (
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": ""},
        ):
            prompt_builder.add_turn(turn["from"], turn["value"])
        input_ids = torch.tensor(
            self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        )
        labels = input_ids.clone()
        eos_positions = torch.where(input_ids == 2)[0]
        if len(eos_positions) == 0:
            raise ValueError("Relocation prompt is missing the expected Llama EOS token (id=2)")
        labels[: int(eos_positions[0])] = IGNORE_INDEX

        raw_actions = np.asarray(episode["actions"], dtype=np.float32)
        standardized = raw_actions.copy()
        # Raw LIBERO uses -1=open, +1=close. Match libero_dataset_transform:
        # clip to [0,1], invert, yielding 1=open and 0=close.
        standardized[:, 6] = 1.0 - np.clip(standardized[:, 6], 0.0, 1.0)
        normalized = standardized.copy()
        normalized[:, :6] = np.clip(
            2.0 * (standardized[:, :6] - self.action_q01[None, :6])
            / (self.action_q99[None, :6] - self.action_q01[None, :6] + 1e-8)
            - 1.0,
            -1.0,
            1.0,
        )
        end = min(len(normalized), position + self.action_horizon)
        valid = normalized[position:end]
        actions = np.zeros((self.action_horizon, 7), dtype=np.float32)
        actions[: len(valid)] = valid
        if len(valid) < self.action_horizon:
            actions[len(valid) :, 6] = valid[-1, 6]
        action_mask = np.zeros(self.action_horizon, dtype=np.bool_)
        action_mask[: len(valid)] = True

        return {
            "pixel_values": self.image_transform(image),
            "input_ids": input_ids,
            "labels": labels,
            "dataset_name": "libero_relocation_npz",
            "actions": torch.from_numpy(actions),
            "action_masks": torch.from_numpy(action_mask),
            "timesteps": np.asarray([position], dtype=np.int64),
            "episode_ids": np.asarray([int(episode["runtime_episode_id"])], dtype=np.int64),
            "occlusion_flags": np.asarray([False], dtype=np.bool_),
            "occlusion_strengths": np.asarray([0.0], dtype=np.float32),
        }


class LiberoRelocationNPZDataset(IterableDataset):
    """Rank-sharded infinite stream that preserves complete episode ordering."""

    def __init__(self, data_root_dir: Path, batch_transform: LiberoRelocationBatchTransform, seed: int = 42) -> None:
        self.data_root_dir = Path(data_root_dir)
        self.episode_paths = sorted((self.data_root_dir / "episodes").glob("episode-*.npz"))
        if not self.episode_paths:
            raise FileNotFoundError(f"No relocation episodes found under {self.data_root_dir / 'episodes'}")
        self.batch_transform = batch_transform
        self.seed = int(seed)
        lengths = []
        for path in self.episode_paths:
            with np.load(path, allow_pickle=False) as episode:
                length = len(episode["actions"])
                if length <= 0 or not np.array_equal(episode["timesteps"], np.arange(length)):
                    raise ValueError(f"Non-contiguous or empty episode: {path}")
                if not bool(np.any(episode["relocation_flags"])):
                    raise ValueError(f"Episode never reaches relocation phase: {path}")
                lengths.append(length)
        self.dataset_length = int(sum(lengths))
        stats = json.loads((self.data_root_dir / "action_stats.json").read_text())
        self.dataset_statistics = {"libero_spatial_no_noops": stats}

    def __len__(self) -> int:
        return self.dataset_length

    def __iter__(self) -> Dict[str, Any]:
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        epoch = 0
        while True:
            indices = np.arange(len(self.episode_paths))
            np.random.default_rng(self.seed + epoch).shuffle(indices)
            rank_indices = indices[rank::world_size]
            for local_index in rank_indices:
                path = self.episode_paths[int(local_index)]
                with np.load(path, allow_pickle=False) as loaded:
                    episode = {key: loaded[key] for key in loaded.files}
                episode["runtime_episode_id"] = epoch * len(self.episode_paths) + int(local_index)
                for position in range(len(episode["actions"])):
                    yield self.batch_transform(episode, position)
            epoch += 1


class StreamRLDSDataset(RLDSDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_dataset(self, rlds_config):
        return make_interleaved_episodic_dataset(
            **rlds_config,
            use_optim_group_sample=False,
        )

    def __iter__(self) -> Dict[str, Any]:
        episode_id = -1
        for rlds_batch in self.dataset.as_numpy_iterator():
            episode_id += 1
            T = rlds_batch["action"].shape[0]
            for i in range(T):
                raw_frame = tree_map(lambda x: x[i], rlds_batch)
                raw_frame["epis_idx"] = np.asarray([episode_id], dtype=np.int64)
                raw_frame["episode_position"] = i
                raw_frame["episode_length"] = T
                frame = self.batch_transform(raw_frame)
                frame["episode_ids"] = np.array([episode_id])
                yield frame


def _frame_generator(batch_dict, batch_transform):
    length = batch_dict["action"].shape[0]
    for i in range(length):
        yield batch_transform(tree_map(lambda x: x[i], batch_dict))
