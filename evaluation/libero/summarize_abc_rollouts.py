#!/usr/bin/env python3
"""Build live CSV/Markdown summaries for the scheduled A/B/C LIBERO rollouts."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


EPISODES_RE = re.compile(r"# episodes completed so far: (\d+)")
SUCCESSES_RE = re.compile(r"# successes: (\d+)")


def parse_counts(log_dir: Path) -> tuple[int, int]:
    logs = sorted(log_dir.glob("*.txt"), key=lambda path: path.stat().st_mtime)
    if not logs:
        return 0, 0
    text = logs[-1].read_text(errors="replace")
    episodes = EPISODES_RE.findall(text)
    successes = SUCCESSES_RE.findall(text)
    return (int(episodes[-1]) if episodes else 0, int(successes[-1]) if successes else 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    status_path = args.root / "task_status.tsv"
    records: dict[tuple[str, str, int], dict[str, str]] = {}
    if status_path.exists():
        with status_path.open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                key = (row["model"], row["suite"], int(row["task_id"]))
                records[key] = row

    aggregate: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"tasks_attempted": 0, "tasks_ok": 0, "tasks_skipped": 0, "episodes": 0, "successes": 0}
    )
    detail_rows = []
    for (model, suite, task_id), row in sorted(records.items()):
        episodes, successes = parse_counts(Path(row["log_dir"]))
        status = row["status"]
        bucket = aggregate[(model, suite)]
        bucket["tasks_attempted"] += 1
        bucket["tasks_ok"] += int(status == "ok")
        bucket["tasks_skipped"] += int(status != "ok")
        bucket["episodes"] += episodes
        bucket["successes"] += successes
        detail_rows.append(
            {
                "model": model,
                "suite": suite,
                "task_id": task_id,
                "status": status,
                "exit_code": row["exit_code"],
                "episodes": episodes,
                "successes": successes,
                "success_rate": f"{100 * successes / episodes:.2f}" if episodes else "",
                "log_dir": row["log_dir"],
            }
        )

    csv_path = args.root / "task_results.csv"
    fields = ["model", "suite", "task_id", "status", "exit_code", "episodes", "successes", "success_rate", "log_dir"]
    with csv_path.with_suffix(".csv.tmp").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(detail_rows)
    csv_path.with_suffix(".csv.tmp").replace(csv_path)

    suites = sorted({suite for _, suite, _ in records})
    if not suites:
        suites = ["libero_spatial"]
    expected_tasks = {suite: 90 if suite == "libero_90" else 10 for suite in suites}
    rendered_suites = ", ".join(suites)
    lines = [
        "# A/B/C LIBERO rollout summary",
        "",
        f"Protocol: {rendered_suites}; 50 episodes/task using each official initial state once; seed 7; CFG 1.5; action chunk 8.",
        "",
        "| Model | Suite | Tasks attempted/expected | Skipped | Episodes | Successes | Success rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in ("A", "B", "C"):
        for suite, expected in expected_tasks.items():
            values = aggregate[(model, suite)]
            rate = 100 * values["successes"] / values["episodes"] if values["episodes"] else 0.0
            lines.append(
                f"| {model} | {suite} | {values['tasks_attempted']}/{expected} | "
                f"{values['tasks_skipped']} | {values['episodes']} | {values['successes']} | {rate:.2f}% |"
            )
    lines.extend(["", "Per-task details: `task_results.csv`", ""])
    report_path = args.root / "SUMMARY.md"
    report_path.with_suffix(".md.tmp").write_text("\n".join(lines))
    report_path.with_suffix(".md.tmp").replace(report_path)


if __name__ == "__main__":
    main()
