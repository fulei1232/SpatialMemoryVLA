#!/usr/bin/env python3
"""Build compact, contiguous RoboMME windows for temporal-memory experiments."""

from __future__ import annotations

import argparse
import pickle
import warnings
from collections import OrderedDict
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/media/fulei/jlu/SpatialMemoryVLA/datasets/robomme_preprocessed_data"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=200)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--dependency-steps", type=int, default=8)
    parser.add_argument("--occlusion-start-ratio", type=float, default=0.4)
    parser.add_argument("--source", choices=("zip-indexed", "extracted", "zip"), default="zip-indexed")
    return parser.parse_args()


def build_archive_index(data_dir: Path, sample_count: int) -> tuple[list[Path], np.ndarray]:
    archives = sorted(data_dir.glob("part_*.zip"), key=lambda path: int(path.stem.split("_")[-1]))
    if not archives:
        raise FileNotFoundError(f"No part_*.zip files found under {data_dir}")
    mapping = np.full(sample_count, -1, dtype=np.int16)
    for archive_index, archive_path in enumerate(archives):
        with ZipFile(archive_path) as archive:
            for name in archive.namelist():
                if name.endswith(".pkl"):
                    sample_index = int(Path(name).stem)
                    if 0 <= sample_index < sample_count:
                        mapping[sample_index] = archive_index
    missing = np.flatnonzero(mapping < 0)
    if len(missing):
        raise RuntimeError(f"Archives are missing sample {int(missing[0])} and {len(missing) - 1} others")
    return archives, mapping


def read_episode_prefix(
    data_dir: Path, archives: list[Path], mapping: np.ndarray, max_episodes: int
) -> OrderedDict[int, list[tuple[int, int]]]:
    handles: dict[int, ZipFile] = {}
    episodes: OrderedDict[int, list[tuple[int, int]]] = OrderedDict()
    completed = 0
    previous_episode = None
    try:
        for sample_index in range(len(mapping)):
            archive_index = int(mapping[sample_index])
            archive = handles.get(archive_index)
            if archive is None:
                archive = ZipFile(archives[archive_index])
                handles[archive_index] = archive
            with archive.open(f"{sample_index}.pkl") as handle:
                sample = pickle.load(handle)
            episode_id = int(np.asarray(sample["epis_idx"]).reshape(-1)[0])
            source_timestep = int(np.asarray(sample["step_idx"]).reshape(-1)[0])
            if previous_episode is not None and episode_id != previous_episode:
                completed += 1
                if completed >= max_episodes:
                    break
            episodes.setdefault(episode_id, []).append((sample_index, source_timestep))
            previous_episode = episode_id
    finally:
        for archive in handles.values():
            archive.close()
    while len(episodes) > max_episodes:
        episodes.popitem(last=True)
    return episodes


def read_extracted_episode_prefix(
    data_dir: Path, max_episodes: int, minimum_length: int
) -> OrderedDict[int, list[tuple[int, int]]]:
    paths = sorted(data_dir.glob("*.pkl"), key=lambda path: int(path.stem))
    episodes: OrderedDict[int, list[tuple[int, int]]] = OrderedDict()
    previous_episode = None
    completed_valid = 0
    skipped = 0
    for path in paths:
        try:
            with path.open("rb") as handle:
                sample = pickle.load(handle)
        except (EOFError, OSError, pickle.UnpicklingError) as exc:
            skipped += 1
            if skipped <= 5:
                warnings.warn(f"Skipping unreadable extracted sample {path}: {exc}")
            continue
        episode_id = int(np.asarray(sample["epis_idx"]).reshape(-1)[0])
        source_timestep = int(np.asarray(sample["step_idx"]).reshape(-1)[0])
        if previous_episode is not None and episode_id != previous_episode:
            if len(episodes[previous_episode]) >= minimum_length:
                completed_valid += 1
                if completed_valid >= max_episodes:
                    break
            else:
                episodes.pop(previous_episode)
        episodes.setdefault(episode_id, []).append((int(path.stem), source_timestep))
        previous_episode = episode_id
    valid = OrderedDict((key, value) for key, value in episodes.items() if len(value) >= minimum_length)
    while len(valid) > max_episodes:
        valid.popitem(last=True)
    if len(valid) < max_episodes:
        raise RuntimeError(f"Only found {len(valid)} extracted episodes with at least {minimum_length} frames")
    return valid


def make_windows(
    episodes: OrderedDict[int, list[tuple[int, int]]],
    group_size: int,
    dependency_steps: int,
    start_ratio: float,
) -> dict[str, np.ndarray]:
    if not 1 <= dependency_steps < group_size:
        raise ValueError("dependency_steps must be in [1, group_size)")
    rows = []
    for episode_id, frames in episodes.items():
        frames.sort(key=lambda item: item[1])
        length = len(frames)
        boundary = min(length - 1, max(1, int(round(start_ratio * max(1, length - 1)))))
        start = boundary - (group_size - dependency_steps)
        start = min(max(0, start), max(0, length - group_size))
        selected = list(range(start, min(length, start + group_size)))
        selected.extend([selected[-1]] * (group_size - len(selected)))
        for position in selected:
            sample_index, source_timestep = frames[position]
            rows.append((sample_index, episode_id, position, source_timestep, position, length))
    keys = (
        "sample_indices", "episode_ids", "timesteps", "source_timesteps", "episode_positions", "episode_lengths"
    )
    return {key: np.asarray([row[index] for row in rows], dtype=np.int64) for index, key in enumerate(keys)}


class ZipMetadataReader:
    def __init__(self, archives: list[Path], mapping: np.ndarray):
        self.archives = archives
        self.mapping = mapping
        self.handles: dict[int, ZipFile] = {}
        self.cache: dict[int, tuple[int, int]] = {}

    def read(self, sample_index: int) -> tuple[int, int]:
        if sample_index in self.cache:
            return self.cache[sample_index]
        archive_index = int(self.mapping[sample_index])
        archive = self.handles.get(archive_index)
        if archive is None:
            archive = ZipFile(self.archives[archive_index])
            self.handles[archive_index] = archive
        with archive.open(f"{sample_index}.pkl") as handle:
            sample = pickle.load(handle)
        value = (
            int(np.asarray(sample["epis_idx"]).reshape(-1)[0]),
            int(np.asarray(sample["step_idx"]).reshape(-1)[0]),
        )
        self.cache[sample_index] = value
        return value

    def lower_bound_episode(self, episode_id: int) -> int:
        low, high = 0, len(self.mapping)
        while low < high:
            middle = (low + high) // 2
            middle_episode, _ = self.read(middle)
            if middle_episode < episode_id:
                low = middle + 1
            else:
                high = middle
        return low

    def close(self) -> None:
        for archive in self.handles.values():
            archive.close()


def make_zip_indexed_windows(
    archives: list[Path],
    mapping: np.ndarray,
    max_episodes: int,
    group_size: int,
    dependency_steps: int,
    start_ratio: float,
) -> dict[str, np.ndarray]:
    reader = ZipMetadataReader(archives, mapping)
    rows = []
    try:
        for episode_id in range(max_episodes):
            episode_start = reader.lower_bound_episode(episode_id)
            episode_end = reader.lower_bound_episode(episode_id + 1)
            length = episode_end - episode_start
            if length < group_size:
                raise RuntimeError(f"Episode {episode_id} has only {length} execution samples")
            boundary = min(length - 1, max(1, int(round(start_ratio * max(1, length - 1)))))
            window_start = boundary - (group_size - dependency_steps)
            window_start = min(max(0, window_start), length - group_size)
            for position in range(window_start, window_start + group_size):
                sample_index = episode_start + position
                actual_episode, source_timestep = reader.read(sample_index)
                if actual_episode != episode_id:
                    raise RuntimeError(
                        f"Non-monotonic episode index at sample {sample_index}: {actual_episode} != {episode_id}"
                    )
                rows.append((sample_index, episode_id, position, source_timestep, position, length))
    finally:
        reader.close()
    keys = (
        "sample_indices", "episode_ids", "timesteps", "source_timesteps", "episode_positions", "episode_lengths"
    )
    return {key: np.asarray([row[index] for row in rows], dtype=np.int64) for index, key in enumerate(keys)}


def main() -> None:
    args = parse_args()
    stats = __import__("json").loads((args.data_root / "meta" / "stats.json").read_text())
    sample_count = int(stats["execution_samples"])
    data_dir = args.data_root / "data"
    if args.source == "extracted":
        episodes = read_extracted_episode_prefix(data_dir, args.max_episodes, args.group_size)
        arrays = make_windows(
            episodes, args.group_size, args.dependency_steps, args.occlusion_start_ratio
        )
    elif args.source == "zip":
        archives, mapping = build_archive_index(data_dir, sample_count)
        episodes = read_episode_prefix(data_dir, archives, mapping, args.max_episodes)
        arrays = make_windows(
            episodes, args.group_size, args.dependency_steps, args.occlusion_start_ratio
        )
    else:
        archives, mapping = build_archive_index(data_dir, sample_count)
        arrays = make_zip_indexed_windows(
            archives,
            mapping,
            args.max_episodes,
            args.group_size,
            args.dependency_steps,
            args.occlusion_start_ratio,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f"wrote {args.output}: {args.max_episodes} episodes, {len(arrays['sample_indices'])} samples")


if __name__ == "__main__":
    main()
