#!/usr/bin/env bash
# One-command reproduction entry points.
#
#   bash reproduce.sh --mode smoke   # small self-training + evaluation loop (~15 min)
#   bash reproduce.sh --mode full    # data prep, 20k training, checkpoint choice, 110 episodes
#
# The smoke mode trains 200 steps from the pinned base weights, then evaluates a
# single seen task with three fixed init states. It exists to prove the whole
# pipeline works; its numbers are not report results.
set -euo pipefail

mode=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) mode="${2:-}"; shift 2 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

case "$mode" in
    smoke|full) ;;
    *) printf 'Usage: %s --mode {smoke|full}\n' "$0" >&2; exit 2 ;;
esac

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

if [[ ! -d data/libero_object_train ]]; then
    python scripts/prepare_data.py --config configs/experiment.json --export-train
fi

if [[ "$mode" == "smoke" ]]; then
    bash scripts/train.sh pilot
    bash scripts/evaluate.py \
        --checkpoint outputs/train/pilot_s42/checkpoints/000200 \
        --conditions seen_original \
        --max-episodes 3 \
        --output-dir outputs/eval/smoke
    python scripts/summarize.py --episodes outputs/eval/smoke/pilot_s42_000200/episodes.csv \
        --output-dir outputs/eval/smoke/pilot_s42_000200
    exit 0
fi

bash scripts/train.sh formal

# Pick the checkpoint with the highest validation success (ties -> earlier step).
best=""
best_rate="-1"
for checkpoint in outputs/train/formal_s42/checkpoints/*/; do
    step="$(basename "$checkpoint")"
    python scripts/evaluate.py --checkpoint "$checkpoint" --validation --no-video \
        --output-dir "outputs/eval/validation"
    rate="$(python - "$checkpoint" <<'PY'
import csv, pathlib, sys
root = pathlib.Path("outputs/eval/validation")
matches = sorted(root.glob(f"formal_s42_{pathlib.Path(sys.argv[1]).name}/episodes.csv"))
rows = list(csv.DictReader(matches[-1].open(encoding="utf-8")))
success = sum(row["success"].lower() == "true" for row in rows)
print(f"{success / len(rows):.6f}")
PY
)"
    printf 'checkpoint %s validation success rate %s\n' "$step" "$rate"
    if python - "$rate" "$best_rate" <<'PY'
import sys
sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)
PY
    then
        best="$checkpoint"
        best_rate="$rate"
    fi
done

printf 'selected checkpoint: %s (validation %s)\n' "$best" "$best_rate"
python scripts/evaluate.py --checkpoint "$best" --output-dir outputs/eval/formal
python scripts/summarize.py --episodes "outputs/eval/formal/$(basename "$(dirname "$best")")_$(basename "$best")/episodes.csv"

printf '%s\n' 'Formal evaluation finished. Review outputs/eval/formal/ and deliverables/evidence/ before reporting.'
