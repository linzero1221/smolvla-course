#!/usr/bin/env python3
"""Convert a lerobot-train log into ``training_log.csv``.

LeRobot logs one line per ``log_freq`` steps, e.g.::

    step:500 smpl:500 ep:26 epch:0.03 loss:1.234 grdn:2.1 lr:9.50e-04 updt_s:0.62

The script extracts ``step``, ``loss``, ``learning_rate`` and a wall-clock
estimate, and writes the columns required by the project plan.

Usage:
    python scripts/parse_train_log.py outputs/train/pilot_s42/train.log
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

PATTERNS = {
    "step": re.compile(r"\bstep:(\d+(?:\.\d+)?)([KMB]?)\b"),
    "loss": re.compile(r"\bloss:([0-9.eE+-]+)"),
    "learning_rate": re.compile(r"\blr:([0-9.eE+-]+)"),
}


def to_number(raw: str, suffix: str = "") -> float:
    scale = {"K": 1e3, "M": 1e6, "B": 1e9}.get(suffix, 1)
    return float(raw) * scale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or args.log.with_name("training_log.csv")
    rows = []
    previous_step = 0.0
    wall_time = 0.0
    for line in args.log.read_text(encoding="utf-8", errors="ignore").splitlines():
        step_match = PATTERNS["step"].search(line)
        loss_match = PATTERNS["loss"].search(line)
        lr_match = PATTERNS["learning_rate"].search(line)
        if not (step_match and loss_match):
            continue
        step = to_number(step_match.group(1), step_match.group(2))
        delta = step - previous_step
        update_match = re.search(r"\bupdt_s:([0-9.eE+-]+)", line)
        if update_match and delta > 0:
            wall_time += float(update_match.group(1)) * delta
        previous_step = step
        rows.append(
            {
                "step": int(step),
                "loss": float(loss_match.group(1)),
                "learning_rate": float(lr_match.group(1)) if lr_match else "",
                "wall_time_s": round(wall_time, 3),
            }
        )

    if not rows:
        print(f"error: no training rows parsed from {args.log}", file=sys.stderr)
        return 1

    # The console log interleaves tqdm carriage returns, so one INFO line can be
    # captured more than once. Keep the last record per step.
    deduped = {}
    for row in rows:
        deduped[row["step"]] = row
    rows = [deduped[step] for step in sorted(deduped)]

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss", "learning_rate", "wall_time_s"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
