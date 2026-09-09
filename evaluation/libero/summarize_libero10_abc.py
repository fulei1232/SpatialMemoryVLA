#!/usr/bin/env python3
"""Aggregate matched LIBERO-10 A/B/C rollout records."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


MODELS = ("A", "B", "C")
ROLLOUT_FIELDS = (
    "model",
    "task_name",
    "runtime_task_index",
    "initial_state_id",
    "success",
    "policy_calls",
    "timeout",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-states", type=int, default=50)
    args = parser.parse_args()

    records = {}
    for model in MODELS:
        for path in sorted((args.root / model).glob("**/rollouts.jsonl")):
            with path.open() as handle:
                for line in handle:
                    record = json.loads(line)
                    key = (
                        record["model"],
                        int(record["runtime_task_index"]),
                        int(record["initial_state_id"]),
                    )
                    records[key] = record

    ordered = sorted(
        records.values(),
        key=lambda row: (row["model"], int(row["runtime_task_index"]), int(row["initial_state_id"])),
    )
    with (args.root / "libero10_rollouts.jsonl").open("w") as handle:
        for record in ordered:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (args.root / "libero10_rollouts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROLLOUT_FIELDS)
        writer.writeheader()
        writer.writerows(ordered)

    grouped = defaultdict(list)
    names = {}
    for record in ordered:
        model = record["model"]
        task_index = int(record["runtime_task_index"])
        grouped[(model, task_index)].append(bool(record["success"]))
        names[task_index] = record["task_name"]

    summary_rows = []
    for task_index in sorted(names):
        rates = {}
        counts = {}
        for model in MODELS:
            values = grouped[(model, task_index)]
            counts[model] = len(values)
            rates[model] = sum(values) / len(values) if values else None
        summary_rows.append(
            {
                "Task": names[task_index],
                "runtime_task_index": task_index,
                "A": rates["A"],
                "B": rates["B"],
                "C": rates["C"],
                "C-A": None if rates["A"] is None or rates["C"] is None else rates["C"] - rates["A"],
                "C-B": None if rates["B"] is None or rates["C"] is None else rates["C"] - rates["B"],
                "A_rollouts": counts["A"],
                "B_rollouts": counts["B"],
                "C_rollouts": counts["C"],
            }
        )

    summary_fields = (
        "Task",
        "runtime_task_index",
        "A",
        "B",
        "C",
        "C-A",
        "C-B",
        "A_rollouts",
        "B_rollouts",
        "C_rollouts",
    )
    with (args.root / "libero10_task_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    def overall(model: str) -> tuple[int, int, float | None]:
        values = [
            bool(record["success"])
            for record in ordered
            if record["model"] == model
        ]
        return len(values), sum(values), (sum(values) / len(values) if values else None)

    lines = [
        "# LIBERO-10 matched A/B/C rollout summary",
        "",
        "Protocol: 10 tasks × 50 official initial states; matched seed, CFG, DDIM, action chunk, horizon, and normalization.",
        "",
        "| Task | A | B | C | C-A | C-B |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        def rate(key: str) -> str:
            return "NA" if row[key] is None else f"{100 * row[key]:.2f}%"

        lines.append(
            f"| {row['Task']} | {rate('A')} | {rate('B')} | {rate('C')} | "
            f"{rate('C-A')} | {rate('C-B')} |"
        )
    lines.append("")
    for model in MODELS:
        count, successes, rate = overall(model)
        rendered = "NA" if rate is None else f"{100 * rate:.2f}%"
        completeness = "complete" if count == 10 * args.expected_states else f"incomplete ({count}/500)"
        lines.append(f"- {model} overall SR: {rendered} ({successes}/{count}; {completeness})")
    lines.append("")
    (args.root / "LIBERO10_ABC_SUMMARY.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
