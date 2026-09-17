#!/usr/bin/env python3
"""Summarize paired LIBERO normal/temporal-occlusion memory checks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


CONDITIONS = {
    "memory_normal": Path("memory/normal"),
    "memory_occlusion": Path("memory/temporal_occlusion"),
    "reset_occlusion": Path("reset/temporal_occlusion"),
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def episode_key(record: dict) -> tuple[int, int]:
    return int(record["runtime_task_index"]), int(record["initial_state_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-per-condition", type=int, default=4)
    args = parser.parse_args()

    groups: dict[str, list[dict]] = {}
    for name, relative_root in CONDITIONS.items():
        records: list[dict] = []
        for path in sorted((args.root / relative_root).glob("task-*/rollouts.jsonl")):
            records.extend(read_jsonl(path))
        groups[name] = sorted(records, key=episode_key)

    key_sets = {name: {episode_key(row) for row in rows} for name, rows in groups.items()}
    complete = all(len(rows) == args.expected_per_condition for rows in groups.values())
    paired = bool(key_sets) and len({frozenset(keys) for keys in key_sets.values()}) == 1

    phase_totals: dict[str, Counter] = {}
    lifecycle_complete: dict[str, bool] = {}
    for name in ("memory_occlusion", "reset_occlusion"):
        totals: Counter = Counter()
        traces: list[dict] = []
        for path in sorted((args.root / CONDITIONS[name]).glob("task-*/failure_trace.jsonl")):
            traces.extend(read_jsonl(path))
        for trace in traces:
            totals.update(trace.get("occlusion_phase_counts", {}))
        phase_totals[name] = totals
        lifecycle_complete[name] = (
            len(traces) == len(groups[name])
            and all(
                trace.get("occlusion_phase_counts", {}).get("full_occlusion", 0) > 0
                and trace.get("occlusion_phase_counts", {}).get("recovered_visible", 0) > 0
                for trace in traces
            )
        )

    metrics = {}
    for name, rows in groups.items():
        successes = sum(bool(row["success"]) for row in rows)
        metrics[name] = {
            "episodes": len(rows),
            "successes": successes,
            "success_rate": successes / len(rows) if rows else None,
            "mean_policy_calls": (
                sum(int(row["policy_calls"]) for row in rows) / len(rows) if rows else None
            ),
        }

    memory_rate = metrics["memory_occlusion"]["success_rate"]
    reset_rate = metrics["reset_occlusion"]["success_rate"]
    result = {
        "protocol": {
            "complete": complete,
            "paired_task_initial_states": paired,
            "expected_episodes_per_condition": args.expected_per_condition,
            "occlusion_lifecycle_complete": lifecycle_complete,
            "occlusion_phase_totals": {k: dict(v) for k, v in phase_totals.items()},
        },
        "metrics": metrics,
        "memory_advantage_under_occlusion": (
            memory_rate - reset_rate if memory_rate is not None and reset_rate is not None else None
        ),
        "episodes": groups,
    }
    (args.root / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    def render_rate(value: float | None) -> str:
        return "NA" if value is None else f"{100 * value:.1f}%"

    lines = [
        "# LIBERO temporal-occlusion memory smoke",
        "",
        f"- Complete: {complete}",
        f"- Paired task/initial states: {paired}",
        f"- Memory occlusion lifecycle complete: {lifecycle_complete['memory_occlusion']}",
        f"- Reset occlusion lifecycle complete: {lifecycle_complete['reset_occlusion']}",
        "",
        "| Condition | Success | Rate | Mean policy calls |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in CONDITIONS:
        metric = metrics[name]
        mean_calls = metric["mean_policy_calls"]
        lines.append(
            f"| {name} | {metric['successes']}/{metric['episodes']} | "
            f"{render_rate(metric['success_rate'])} | "
            f"{'NA' if mean_calls is None else f'{mean_calls:.2f}'} |"
        )
    lines.extend(
        [
            "",
            "Memory advantage under temporal occlusion: "
            + render_rate(result["memory_advantage_under_occlusion"]),
            "",
            "This is a technical smoke test; four paired episodes are not a performance claim.",
        ]
    )
    (args.root / "SUMMARY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
