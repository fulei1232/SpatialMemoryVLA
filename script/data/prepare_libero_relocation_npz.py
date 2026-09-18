#!/usr/bin/env python3
"""Audit successful rollout NPZ files and materialize a fixed relocation dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-stats", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    candidates = sorted(args.staging_root.rglob("episode-*.npz"))
    valid: list[tuple[Path, dict]] = []
    errors: list[str] = []
    for path in candidates:
        try:
            with np.load(path, allow_pickle=False) as episode:
                length = len(episode["actions"])
                if episode["images"].shape != (length, 256, 256, 3):
                    raise ValueError(f"bad image shape {episode['images'].shape}")
                if episode["actions"].shape != (length, 7):
                    raise ValueError(f"bad action shape {episode['actions'].shape}")
                if not np.array_equal(episode["timesteps"], np.arange(length)):
                    raise ValueError("non-contiguous timesteps")
                flags = np.asarray(episode["relocation_flags"], dtype=np.bool_)
                transitions = np.flatnonzero(flags[1:] != flags[:-1]) + 1
                if not (len(transitions) == 1 and not flags[0] and flags[-1]):
                    raise ValueError(f"invalid relocation flag lifecycle: {transitions.tolist()}")
                displacement = float(np.linalg.norm(episode["new_pose"][:3] - episode["old_pose"][:3]))
                if not np.isclose(displacement, 0.10, atol=1e-4):
                    raise ValueError(f"unexpected displacement {displacement}")
                if not np.isfinite(episode["actions"]).all():
                    raise ValueError("non-finite actions")
                metadata = {
                    "path": str(path),
                    "steps": length,
                    "relocation_step": int(transitions[0]),
                    "seed": int(episode["seed"]),
                    "initial_state_id": int(episode["initial_state_id"]),
                    "source_episode_index": int(episode["source_episode_index"]),
                    "displacement_m": displacement,
                }
            valid.append((path, metadata))
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    if len(valid) < args.episodes:
        raise RuntimeError(
            f"Only {len(valid)} valid successful episodes found; requested {args.episodes}. Errors: {errors[:5]}"
        )
    selected = valid[: args.episodes]
    episodes_dir = args.output_root / "episodes"
    if episodes_dir.exists() and any(episodes_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite populated dataset: {episodes_dir}")
    episodes_dir.mkdir(parents=True, exist_ok=True)
    for index, (source, metadata) in enumerate(selected):
        destination = episodes_dir / f"episode-{index:04d}.npz"
        shutil.copy2(source, destination)
        metadata["dataset_episode_id"] = index
        metadata["dataset_path"] = str(destination)

    stats = json.loads(args.source_stats.read_text())
    (args.output_root / "action_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    summary = {
        "dataset_type": "successful-policy-rollout-relocation",
        "episodes": len(selected),
        "transitions": sum(row[1]["steps"] for row in selected),
        "candidate_episodes": len(candidates),
        "valid_candidates": len(valid),
        "rejected_candidates": len(errors),
        "seeds": sorted({row[1]["seed"] for row in selected}),
        "episode_records": [row[1] for row in selected],
        "rejections": errors,
    }
    (args.output_root / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "episode_records"}, indent=2))


if __name__ == "__main__":
    main()
