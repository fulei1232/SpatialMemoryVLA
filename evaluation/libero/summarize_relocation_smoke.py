#!/usr/bin/env python3
"""Validate and summarize paired B/C1/C16 relocation smoke rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    result: dict[str, dict] = {}
    state_sets: dict[str, set[int]] = {}
    for label in ("B", "C1", "C16"):
        model_root = args.root / label
        rollouts = []
        traces = []
        for path in model_root.rglob("rollouts.jsonl"):
            rollouts.extend(read_jsonl(path))
        for path in model_root.rglob("failure_trace.jsonl"):
            traces.extend(read_jsonl(path))
        interventions = [row.get("intervention") for row in traces]
        interventions = [row for row in interventions if row and row.get("type") == "relocation"]
        def unexpected_contacts(row: dict) -> list:
            if "unexpected_contacts_after" in row:
                return row["unexpected_contacts_after"]
            return [
                pair for pair in row.get("contacts_after", [])
                if not any(
                    marker in geom.lower()
                    for geom in pair for marker in ("table", "floor")
                )
            ]

        def safe_relocation(row: dict) -> bool:
            return bool(
                row.get("xy_in_workspace")
                and row.get("z_unchanged")
                and float(row.get("pose_error", 1.0)) < 1e-6
                and not unexpected_contacts(row)
            )
        state_sets[label] = {int(row["initial_state_id"]) for row in rollouts}
        result[label] = {
            "episodes": len(rollouts),
            "successes": sum(bool(row.get("success")) for row in rollouts),
            "success_rate": (
                sum(bool(row.get("success")) for row in rollouts) / len(rollouts)
                if rollouts else None
            ),
            "exceptions": sum(row.get("exception") is not None for row in traces),
            "interventions": len(interventions),
            "safe_relocations": sum(safe_relocation(row) for row in interventions),
            "pose_verified": sum(float(row.get("pose_error", 1.0)) < 1e-6 for row in interventions),
            "observation_refreshed": sum(
                float(row.get("observation_mean_abs_diff", 0.0)) > 0.0 for row in interventions
            ),
            "timestep_continuous": sum(
                row.get("timestep_before") == row.get("timestep_after") for row in interventions
            ),
            "no_unexpected_contacts_after": sum(not unexpected_contacts(row) for row in interventions),
            "initial_state_ids": sorted(state_sets[label]),
            "mean_min_eef_distance_to_old": (
                sum(row["min_eef_distance_to_old"] for row in interventions if row.get("min_eef_distance_to_old") is not None)
                / max(1, sum(row.get("min_eef_distance_to_old") is not None for row in interventions))
            ),
            "mean_min_eef_distance_to_new": (
                sum(row["min_eef_distance_to_new"] for row in interventions if row.get("min_eef_distance_to_new") is not None)
                / max(1, sum(row.get("min_eef_distance_to_new") is not None for row in interventions))
            ),
        }

    result["paired_initial_states"] = bool(
        state_sets["B"] and state_sets["B"] == state_sets["C1"] == state_sets["C16"]
    )
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# LIBERO dynamic relocation smoke",
        "",
        f"Paired initial states: **{result['paired_initial_states']}**",
        "",
        "| Model | Episodes | Success | Safe pose | Obs refreshed | No unexpected contact | Exceptions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in ("B", "C1", "C16"):
        row = result[label]
        rate = "n/a" if row["success_rate"] is None else f"{100 * row['success_rate']:.1f}%"
        lines.append(
            f"| {label} | {row['episodes']} | {rate} | {row['safe_relocations']}/{row['interventions']} "
            f"| {row['observation_refreshed']}/{row['interventions']} | {row['no_unexpected_contacts_after']}/{row['interventions']} "
            f"| {row['exceptions']} |"
        )
    lines.extend([
        "",
        "Protocol: task 8 (`next_to_the_plate`), official states 0–9, seed 7, relocation before policy call 5, target bowl translated (+0.0954 m X, +0.03 m Y), total 0.10 m.",
    ])
    (args.root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
