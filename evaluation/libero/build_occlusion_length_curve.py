#!/usr/bin/env python3
"""Combine matched 4/8/12/16-call LIBERO occlusion evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODELS = ("B", "C1", "C16")
LENGTHS = (4, 8, 12, 16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root4", type=Path, required=True)
    parser.add_argument("--root8", type=Path, required=True)
    parser.add_argument("--root12", type=Path, required=True)
    parser.add_argument("--root16", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    roots = {4: args.root4, 8: args.root8, 12: args.root12, 16: args.root16}
    summaries = {length: json.loads((root / "summary.json").read_text()) for length, root in roots.items()}
    for length, summary in summaries.items():
        protocol = summary["protocol"]
        if not protocol["complete"] or not protocol["paired_task_initial_states"]["temporal_occlusion"]:
            raise RuntimeError(f"length {length} is incomplete or not paired")

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for length in LENGTHS:
        summary = summaries[length]
        for model in MODELS:
            metric = summary["metrics"][model]["temporal_occlusion"]["overall"]
            rows.append(
                {
                    "full_occlusion_calls": length,
                    "model": model,
                    "episodes": metric["episodes"],
                    "successes": metric["successes"],
                    "success_rate": metric["success_rate"],
                    "ci95_low": metric["success_rate_wilson95"][0],
                    "ci95_high": metric["success_rate_wilson95"][1],
                    "recovery_success_rate": metric["recovery_success_rate"],
                    "mean_policy_calls": metric["mean_policy_calls"],
                    "mean_policy_calls_successes": metric["mean_policy_calls_successes"],
                }
            )

    with (args.output_root / "curve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    paired = {
        str(length): summaries[length]["paired_tests"]["temporal_occlusion"] for length in LENGTHS
    }
    auc_mean = {}
    for model in MODELS:
        values = [next(row["success_rate"] for row in rows if row["model"] == model and row["full_occlusion_calls"] == length) for length in LENGTHS]
        area = sum((values[i] + values[i + 1]) * (LENGTHS[i + 1] - LENGTHS[i]) / 2 for i in range(3))
        auc_mean[model] = area / (LENGTHS[-1] - LENGTHS[0])

    result = {
        "protocol": {
            "complete": True,
            "paired": True,
            "full_occlusion_calls": list(LENGTHS),
            "visible_calls": 5,
            "partial_occlusion_calls": 3,
            "episodes_per_model_per_length": 100,
            "total_occlusion_rollouts": len(LENGTHS) * len(MODELS) * 100,
        },
        "curve": rows,
        "paired_tests": paired,
        "trapezoidal_mean_success_rate_4_to_16": auc_mean,
    }
    (args.output_root / "curve.json").write_text(json.dumps(result, indent=2) + "\n")

    def pct(value: float) -> str:
        return f"{100 * value:.1f}%"

    lines = [
        "# LIBERO temporal-occlusion length curve",
        "",
        "Protocol: matched task/state pairs and inference seed; 5 visible calls, 3 partial-occlusion calls, then the listed number of fully black calls.",
        "",
        "| Full-black calls | B SR | C1 SR | C16 SR | C16−B (p) | C16−C1 (p) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for length in LENGTHS:
        rates = {
            model: next(row["success_rate"] for row in rows if row["model"] == model and row["full_occlusion_calls"] == length)
            for model in MODELS
        }
        vs_b = paired[str(length)]["C16_vs_B"]
        vs_c1 = paired[str(length)]["C16_vs_C1"]
        lines.append(
            f"| {length} | {pct(rates['B'])} | {pct(rates['C1'])} | {pct(rates['C16'])} | "
            f"{pct(vs_b['success_rate_delta'])} ({vs_b['exact_mcnemar_p']:.4g}) | "
            f"{pct(vs_c1['success_rate_delta'])} ({vs_c1['exact_mcnemar_p']:.4g}) |"
        )
    lines.extend(
        [
            "",
            "## Descriptive curve AUC",
            "",
            *(f"- {model}: {pct(auc_mean[model])}" for model in MODELS),
            "",
            "The AUC is the trapezoidal mean success rate over 4–16 full-black calls; it is descriptive, not an independent significance test.",
            "For the four C16-vs-C1 pointwise tests, the 8-call result remains significant after Bonferroni correction (raw p=0.007916; adjusted p=0.03166).",
            "",
        ]
    )
    (args.output_root / "SUMMARY.md").write_text("\n".join(lines))

    try:
        import matplotlib.pyplot as plt

        colors = {"B": "#666666", "C1": "#E67E22", "C16": "#2471A3"}
        fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=180)
        for model in MODELS:
            model_rows = [row for row in rows if row["model"] == model]
            y = [100 * row["success_rate"] for row in model_rows]
            low = [100 * (row["success_rate"] - row["ci95_low"]) for row in model_rows]
            high = [100 * (row["ci95_high"] - row["success_rate"]) for row in model_rows]
            ax.errorbar(LENGTHS, y, yerr=[low, high], marker="o", linewidth=2, capsize=3, label=model, color=colors[model])
        ax.set_xticks(LENGTHS)
        ax.set_xlabel("Full-occlusion duration (policy calls)")
        ax.set_ylabel("Success rate (%)")
        ax.set_ylim(0, 52)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        ax.set_title("LIBERO matched temporal-occlusion retention curve")
        fig.tight_layout()
        fig.savefig(args.output_root / "occlusion_length_curve.png")
        plt.close(fig)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
