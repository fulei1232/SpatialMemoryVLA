#!/usr/bin/env python3
"""Summarize matched LIBERO-Spatial B/C1/C16 temporal-occlusion smoke runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


METRICS = {
    "action_loss": "VLA Train/Action Loss",
    "spatial_loss": "VLA Train/Spatial Loss",
    "total_loss": "VLA Train/Total Loss",
    "grad_norm": "VLA Train/Grad Norm",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=20)
    parser.add_argument("--run-tag", default="det_noaug")
    args = parser.parse_args()

    runs = {
        "B": f"libero_spatial_occ_B_{args.expected_steps}step_{args.run_tag}",
        "C1": f"libero_spatial_occ_C_mem1_{args.expected_steps}step_{args.run_tag}",
        "C16": f"libero_spatial_occ_C_mem16_{args.expected_steps}step_{args.run_tag}",
    }

    result: dict[str, dict] = {}
    configs: dict[str, dict] = {}
    for label, dirname in runs.items():
        run_dir = args.root / dirname
        config = json.loads((run_dir / "config.json").read_text())
        configs[label] = config
        metric_files = sorted(run_dir.glob("*.jsonl"))
        metric_file = next(path for path in metric_files if path.name not in {"run-metrics.jsonl", "checkpoint-events.jsonl"})
        rows = read_jsonl(metric_file)

        checkpoints = list((run_dir / "checkpoints").glob("*.pt"))
        optimizers = list((run_dir / "checkpoints").glob("*.optimizer"))
        diagnostics = []
        diagnostic_sequence = []
        for path in sorted((run_dir / "diagnostics").glob("gate*.csv")):
            with path.open(newline="") as handle:
                path_rows = list(csv.DictReader(handle))
            diagnostics.extend(path_rows)
            diagnostic_sequence.extend(
                f"{path.name}:{row['episode_id']}:{row['timestep']}:"
                f"{row['occlusion_flag']}:{row['occlusion_strength']}"
                for row in path_rows
            )

        summary = {
            "steps": len(rows),
            "complete": len(rows) == args.expected_steps and len(checkpoints) == 1 and len(optimizers) == 1,
            "checkpoint": str(checkpoints[0]) if len(checkpoints) == 1 else None,
            "optimizer": str(optimizers[0]) if len(optimizers) == 1 else None,
            "mem_length": config["mem_length"],
            "use_spatial_memory": config["use_spatial_memory"],
        }
        for short_name, key in METRICS.items():
            values = [float(row[key]) for row in rows]
            summary[short_name] = {
                "first": values[0] if values else None,
                "last": values[-1] if values else None,
                "mean": sum(values) / len(values) if values else None,
                "mean_last_50": sum(values[-50:]) / len(values[-50:]) if values else None,
                "finite": all(math.isfinite(value) for value in values),
                "max": max(values) if values else None,
            }
        strengths = [float(row["occlusion_strength"]) for row in diagnostics]
        history_sizes = [int(row["history_size"]) for row in diagnostics]
        summary["diagnostics"] = {
            "rows": len(diagnostics),
            "occluded_fraction": sum(value > 0 for value in strengths) / len(strengths) if strengths else None,
            "full_occlusion_fraction": sum(value >= 0.999 for value in strengths) / len(strengths) if strengths else None,
            "max_history_size": max(history_sizes) if history_sizes else None,
            "occlusion_sequence_sha256": hashlib.sha256(
                "\n".join(diagnostic_sequence).encode()
            ).hexdigest(),
        }
        result[label] = summary

    fairness_fields = (
        "pretrained_checkpoint", "seed", "data_root_dir", "dataloader_type",
        "future_action_window_size", "action_dim", "repeated_diffusion_steps",
        "memory_curriculum_enabled", "memory_curriculum_type", "occlusion_probability",
        "occlusion_start_ratio", "occlusion_duration_ratio", "occlusion_recovery_ratio",
        "occlusion_strength", "spatial_align_coeff", "spatial_teacher_path",
    )
    fairness = {
        field: {json.dumps(config[field], sort_keys=True, default=str) for config in configs.values()}
        for field in fairness_fields
    }
    matched = all(len(values) == 1 for values in fairness.values())
    occlusion_hashes = {
        row["diagnostics"]["occlusion_sequence_sha256"] for row in result.values()
    }
    matched_occlusion_sequence = len(occlusion_hashes) == 1
    payload = {
        "complete": all(item["complete"] for item in result.values()),
        "matched_shared_configuration": matched,
        "matched_occlusion_sequence": matched_occlusion_sequence,
        "runs": result,
    }
    output_stem = f"libero_spatial_occlusion_{args.expected_steps}step_{args.run_tag}_summary"
    (args.root / f"{output_stem}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )

    lines = [
        "# LIBERO-Spatial occlusion training smoke",
        "",
        f"- Complete: {payload['complete']}",
        f"- Matched shared configuration: {matched}",
        f"- Matched per-rank occlusion sequence: {matched_occlusion_sequence}",
        "",
        "| Run | Steps | Action first → last | Action last-50 | Spatial first → last | Total first → last | Max grad | Occluded | Max history |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, row in result.items():
        def pair(name: str) -> str:
            metric = row[name]
            return f"{metric['first']:.4f} → {metric['last']:.4f}"

        diagnostic = row["diagnostics"]
        lines.append(
            f"| {label} | {row['steps']} | {pair('action_loss')} | "
            f"{row['action_loss']['mean_last_50']:.4f} | {pair('spatial_loss')} | "
            f"{pair('total_loss')} | {row['grad_norm']['max']:.4f} | "
            f"{100 * diagnostic['occluded_fraction']:.1f}% | {diagnostic['max_history_size']} |"
        )
    note = (
        "Twenty steps validate the pipeline only; they are not a model-quality comparison."
        if args.expected_steps <= 20
        else "Training losses are diagnostic only; model quality must be decided by matched simulator rollouts."
    )
    lines.extend(["", note, ""])
    (args.root / f"{output_stem.upper()}.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
