#!/usr/bin/env python3
"""Verify that completed LIBERO-10 A/B/C run configs differ only by ablation mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MATCHED_FIELDS = (
    "pretrained_checkpoint",
    "vla.data_mix",
    "vla.max_steps",
    "vla.learning_rate",
    "vla.global_batch_size",
    "vla.shuffle_buffer_size",
    "image_aug",
    "future_action_window_size",
    "action_model_type",
    "repeated_diffusion_steps",
    "dataloader_type",
    "mem_length",
    "seed",
    "save_interval",
)
EXPECTED_MODES = {
    "A": ("memoryvla", False, False),
    "B": ("spatial_forcing", True, False),
    "C": ("spatial_memory", True, True),
}


def nested(config: dict, dotted_key: str):
    value = config
    for key in dotted_key.split("."):
        value = value[key]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("a_config", type=Path)
    parser.add_argument("b_config", type=Path)
    parser.add_argument("c_config", type=Path)
    args = parser.parse_args()
    configs = {
        model: json.loads(path.read_text())
        for model, path in zip(("A", "B", "C"), (args.a_config, args.b_config, args.c_config))
    }

    errors = []
    for field in MATCHED_FIELDS:
        values = {model: nested(config, field) for model, config in configs.items()}
        if len({json.dumps(value, sort_keys=True) for value in values.values()}) != 1:
            errors.append(f"{field} differs: {values}")

    for model, (mode, forcing, memory) in EXPECTED_MODES.items():
        config = configs[model]
        actual = (
            config["experiment_mode"],
            config["use_spatial_forcing"],
            config["use_spatial_memory"],
        )
        if actual != (mode, forcing, memory):
            errors.append(f"{model} mode mismatch: expected {(mode, forcing, memory)}, got {actual}")

    if errors:
        print("LIBERO-10 A/B/C fairness check: FAIL")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("LIBERO-10 A/B/C fairness check: PASS")
    for field in MATCHED_FIELDS:
        print(f"{field}: {nested(configs['A'], field)}")
    print("mode-only differences: A=memoryvla, B=spatial_forcing, C=spatial_memory")


if __name__ == "__main__":
    main()
