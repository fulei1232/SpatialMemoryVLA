#!/usr/bin/env python3
"""Aggregate paired B/C1/C16 LIBERO normal and temporal-occlusion rollouts."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


MODELS = ("B", "C1", "C16")
CONDITIONS = ("normal", "temporal_occlusion")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def key(row: dict) -> tuple[int, int]:
    return int(row["runtime_task_index"]), int(row["initial_state_id"])


def wilson(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if total == 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - margin, center + margin]


def exact_mcnemar_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def summarize(rows: list[dict]) -> dict:
    successes = sum(bool(row["success"]) for row in rows)
    recovery_rows = [row for row in rows if bool(row.get("reached_recovery", False))]
    recovery_successes = sum(bool(row.get("recovery_success", False)) for row in recovery_rows)
    successful_calls = [int(row["policy_calls"]) for row in rows if row["success"]]
    return {
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows) if rows else None,
        "success_rate_wilson95": wilson(successes, len(rows)),
        "mean_policy_calls": sum(int(row["policy_calls"]) for row in rows) / len(rows) if rows else None,
        "mean_policy_calls_successes": (
            sum(successful_calls) / len(successful_calls) if successful_calls else None
        ),
        "reached_full_occlusion": sum(bool(row.get("reached_full_occlusion", False)) for row in rows),
        "reached_recovery": len(recovery_rows),
        "recovery_successes": recovery_successes,
        "recovery_success_rate": recovery_successes / len(recovery_rows) if recovery_rows else None,
        "success_phases": dict(Counter(row.get("success_phase") for row in rows if row["success"])),
        "timeouts": sum(bool(row["timeout"]) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-states-per-task", type=int, default=50)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--task-ids", nargs="+", type=int, default=[0, 8])
    args = parser.parse_args()
    conditions = tuple(args.conditions)
    task_ids = tuple(args.task_ids)

    records: dict[tuple[str, str], list[dict]] = {}
    for model in MODELS:
        for condition in conditions:
            rows = []
            for path in sorted((args.root / model / condition).glob("task-*/rollouts.jsonl")):
                rows.extend(read_jsonl(path))
            records[(model, condition)] = sorted(rows, key=key)

    expected = len(task_ids) * args.expected_states_per_task
    complete = all(len(rows) == expected for rows in records.values())
    paired = {}
    for condition in conditions:
        key_sets = [{key(row) for row in records[(model, condition)]} for model in MODELS]
        paired[condition] = len({frozenset(keys) for keys in key_sets}) == 1

    metrics = {
        model: {
            condition: {
                "overall": summarize(records[(model, condition)]),
                "tasks": {
                    str(task_id): summarize(
                        [row for row in records[(model, condition)] if int(row["runtime_task_index"]) == task_id]
                    )
                    for task_id in task_ids
                },
            }
            for condition in conditions
        }
        for model in MODELS
    }

    paired_tests = {}
    for condition in conditions:
        paired_tests[condition] = {}
        maps = {
            model: {key(row): bool(row["success"]) for row in records[(model, condition)]}
            for model in MODELS
        }
        for baseline in ("B", "C1"):
            shared = sorted(set(maps["C16"]) & set(maps[baseline]))
            wins = sum(maps["C16"][item] and not maps[baseline][item] for item in shared)
            losses = sum(maps[baseline][item] and not maps["C16"][item] for item in shared)
            paired_tests[condition][f"C16_vs_{baseline}"] = {
                "pairs": len(shared),
                "C16_wins": wins,
                "C16_losses": losses,
                "ties": len(shared) - wins - losses,
                "success_rate_delta": (
                    metrics["C16"][condition]["overall"]["success_rate"]
                    - metrics[baseline][condition]["overall"]["success_rate"]
                ),
                "exact_mcnemar_p": exact_mcnemar_p(wins, losses),
            }

    result = {
        "protocol": {
            "complete": complete,
            "expected_episodes_per_model_condition": expected,
            "paired_task_initial_states": paired,
            "models": list(MODELS),
            "conditions": list(conditions),
        },
        "metrics": metrics,
        "paired_tests": paired_tests,
    }
    (args.root / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    def pct(value: float | None) -> str:
        return "NA" if value is None else f"{100 * value:.1f}%"

    lines = [
        "# LIBERO-Spatial B/C1/C16 matched rollout summary",
        "",
        f"- Complete: {complete}",
        *(f"- Paired {condition} states: {paired[condition]}" for condition in conditions),
        "",
        "| Model | Condition | Success | SR (95% Wilson CI) | Recovery success | Mean calls | Mean calls (success) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        for condition in conditions:
            row = metrics[model][condition]["overall"]
            interval = row["success_rate_wilson95"]
            interval_text = "NA" if interval is None else f"{pct(interval[0])}–{pct(interval[1])}"
            recovery = (
                "NA"
                if row["reached_recovery"] == 0
                else f"{row['recovery_successes']}/{row['reached_recovery']} ({pct(row['recovery_success_rate'])})"
            )
            successful_calls = row["mean_policy_calls_successes"]
            lines.append(
                f"| {model} | {condition} | {row['successes']}/{row['episodes']} | "
                f"{pct(row['success_rate'])} ({interval_text}) | {recovery} | "
                f"{row['mean_policy_calls']:.2f} | "
                f"{'NA' if successful_calls is None else f'{successful_calls:.2f}'} |"
            )
    lines.extend(["", "## Paired C16 comparisons", ""])
    for condition in conditions:
        for baseline in ("B", "C1"):
            row = paired_tests[condition][f"C16_vs_{baseline}"]
            lines.append(
                f"- {condition}, C16 vs {baseline}: ΔSR={pct(row['success_rate_delta'])}, "
                f"wins/losses={row['C16_wins']}/{row['C16_losses']}, "
                f"exact McNemar p={row['exact_mcnemar_p']:.4g}."
            )
    lines.extend(
        [
            "",
            "Recovery success is success among episodes that reached the recovered-visible phase.",
            "Paired significance and effect size, not training loss alone, determine whether history is effective.",
            "",
        ]
    )
    (args.root / "SUMMARY.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
