#!/usr/bin/env python3
"""Print a digest of the first seeded LIBERO RLDS episodes for fairness checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from vla.datasets.datasets import StreamRLDSDataset


class IdentityTransform:
    def __call__(self, frame):
        return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/libero-rlds"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--shuffle-buffer-size", type=int, default=32)
    parser.add_argument("--image-aug", action="store_true")
    args = parser.parse_args()

    dataset = StreamRLDSDataset(
        args.data_root,
        "libero_spatial_no_noops",
        IdentityTransform(),
        resize_resolution=(224, 224),
        shuffle_buffer_size=args.shuffle_buffer_size,
        future_action_window_size=15,
        train=True,
        image_aug=args.image_aug,
        load_all_data_for_training=True,
        seed=args.seed,
    )
    iterator = dataset.dataset.as_numpy_iterator()
    records = []
    for _ in range(args.episodes):
        episode = next(iterator)
        action_bytes = np.asarray(episode["action"]).tobytes()
        image_bytes = np.asarray(episode["observation"]["image_primary"]).tobytes()
        language_items = np.asarray(episode["task"]["language_instruction"]).reshape(-1).tolist()
        language_bytes = b"\0".join(
            item if isinstance(item, bytes) else str(item).encode() for item in language_items
        )
        digest = hashlib.sha256(action_bytes + image_bytes + language_bytes)
        records.append(
            {
                "steps": int(episode["action"].shape[0]),
                "sha256": digest.hexdigest(),
                "action_sha256": hashlib.sha256(action_bytes).hexdigest(),
                "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "language_sha256": hashlib.sha256(language_bytes).hexdigest(),
            }
        )
    print(json.dumps({"seed": args.seed, "rank": 0, "episodes": records}, sort_keys=True))


if __name__ == "__main__":
    main()
