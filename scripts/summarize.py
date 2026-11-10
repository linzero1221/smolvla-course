#!/usr/bin/env python3
"""Summarize evaluation episodes into tables and figures.

Outputs (under ``--output-dir``):
  summary.csv           success counts and rates per condition and per object
  language_pairs.csv    paired original vs paraphrase outcomes per (task, state, paraphrase)
  summary.md            compact markdown tables for the report
  success_by_condition.png
  training_loss.png     only if a training log CSV is supplied

Usage:
    python scripts/summarize.py --episodes outputs/eval/run/checkpoint/episodes.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CONDITIONS = ("seen_original", "seen_paraphrase", "unseen_original")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def as_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"true", "1", "yes"}


def rate_table(rows: list[dict]) -> list[dict]:
    table = []
    for condition in CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition]
        if not subset:
            continue
        successes = sum(as_bool(row["success"]) for row in subset)
        table.append(
            {
                "condition": condition,
                "successes": successes,
                "episodes": len(subset),
                "success_rate": successes / len(subset),
            }
        )
    return table


def object_table(rows: list[dict]) -> list[dict]:
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(row["condition"], row["object"])].append(row)
    table = []
    for (condition, obj), subset in sorted(buckets.items()):
        successes = sum(as_bool(row["success"]) for row in subset)
        table.append(
            {
                "condition": condition,
                "object": obj,
                "successes": successes,
                "episodes": len(subset),
                "success_rate": successes / len(subset),
            }
        )
    return table


def language_pairs(rows: list[dict]) -> list[dict]:
    original = {
        (row["object"], row["init_state_id"]): as_bool(row["success"])
        for row in rows
        if row["condition"] == "seen_original"
    }
    pairs = []
    for row in rows:
        if row["condition"] != "seen_paraphrase":
            continue
        key = (row["object"], row["init_state_id"])
        if key not in original:
            continue
        before, after = original[key], as_bool(row["success"])
        pairs.append(
            {
                "object": row["object"],
                "init_state_id": row["init_state_id"],
                "prompt_id": row["prompt_id"],
                "original_success": before,
                "paraphrase_success": after,
                "delta": int(after) - int(before),
            }
        )
    return pairs


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_conditions(table: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["condition"] for row in table]
    rates = [row["success_rate"] * 100 for row in table]
    counts = [f"{row['successes']}/{row['episodes']}" for row in table]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, rates, color="#4c72b0")
    for bar, count in zip(bars, counts, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, count, ha="center", va="bottom")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("LIBERO-Object closed-loop success by condition")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def read_training_log(log_path: Path) -> tuple[list[float], list[float]]:
    """Accept either a step,loss CSV or the free-form lerobot-train log.

    lines in the lerobot log look like
    ``step:5K smpl:5K ep:4K epch:0.26 loss:1.234 grdn:... lr:... updt_s:...``
    """
    if log_path.suffix == ".csv":
        steps, losses = [], []
        for row in read_csv(log_path):
            if "loss" not in row or row["loss"] in ("", "nan"):
                continue
            steps.append(float(row["step"]))
            losses.append(float(row["loss"]))
        return steps, losses

    import re

    pattern = re.compile(r"step:(\d+(?:\.\d+)?[KMB]?).*?loss:([0-9.eE+-]+)")
    steps, losses = [], []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        raw_step = match.group(1)
        multiplier = {"K": 1e3, "M": 1e6, "B": 1e9}.get(raw_step[-1], 1)
        step = float(raw_step[:-1]) * multiplier if raw_step[-1] in "KMB" else float(raw_step)
        steps.append(step)
        losses.append(float(match.group(2)))
    return steps, losses


def plot_loss(log_path: Path, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps, losses = read_training_log(log_path)
    if not losses:
        print(f"warning: no loss values found in {log_path}")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, losses, linewidth=1)
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.set_title("SmolVLA LoRA training loss")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--training-log", type=Path, default=None)
    args = parser.parse_args()

    rows = read_csv(args.episodes)
    output_dir = args.output_dir or args.episodes.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = rate_table(rows)
    write_csv(output_dir / "summary.csv", conditions, ["condition", "successes", "episodes", "success_rate"])

    objects = object_table(rows)
    write_csv(output_dir / "summary_by_object.csv", objects, ["condition", "object", "successes", "episodes", "success_rate"])

    pairs = language_pairs(rows)
    write_csv(
        output_dir / "language_pairs.csv",
        pairs,
        ["object", "init_state_id", "prompt_id", "original_success", "paraphrase_success", "delta"],
    )

    lines = ["# Evaluation summary", ""]
    lines.append("| condition | successes | episodes | success rate |")
    lines.append("|---|---:|---:|---:|")
    for row in conditions:
        lines.append(f"| {row['condition']} | {row['successes']} | {row['episodes']} | {row['success_rate']:.1%} |")
    lines.append("")
    lines.append("| condition | object | successes | episodes | success rate |")
    lines.append("|---|---|---:|---:|---:|")
    for row in objects:
        lines.append(
            f"| {row['condition']} | {row['object']} | {row['successes']} | {row['episodes']} | "
            f"{row['success_rate']:.1%} |"
        )
    if pairs:
        improved = sum(1 for pair in pairs if pair["delta"] > 0)
        degraded = sum(1 for pair in pairs if pair["delta"] < 0)
        lines.append("")
        lines.append(f"Paired paraphrase outcomes: {improved} success->failure, {degraded} failure->success, "
                     f"{len(pairs) - improved - degraded} unchanged out of {len(pairs)} pairs.")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    plot_conditions(conditions, output_dir / "success_by_condition.png")
    if args.training_log:
        plot_loss(args.training_log, output_dir / "training_loss.png")

    counts = {row["condition"]: row["episodes"] for row in conditions}
    print(json.dumps(counts, indent=2))
    print(f"wrote summaries to {output_dir}")
    expected = {"seen_original": 30, "seen_paraphrase": 60, "unseen_original": 20}
    for condition, expected_count in expected.items():
        if counts.get(condition, 0) != expected_count:
            print(f"warning: {condition} has {counts.get(condition, 0)} episodes, expected {expected_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
