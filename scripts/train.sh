#!/usr/bin/env bash
# Fine-tune SmolVLA on the LIBERO-Object training subset with LoRA.
#
# Usage:
#   bash scripts/train.sh pilot     # 200-step trial run  -> outputs/train/pilot_s42
#   bash scripts/train.sh formal    # 20k-step full run    -> outputs/train/formal_s42
#   bash scripts/train.sh resume <checkpoint_dir> <output_dir>
#                                   # continue a paused run from a checkpoint
#
# The three modes read sizes and hyper-parameters from configs/experiment.json
# (resume reads them from the checkpoint's train_config.json) so the script and
# the experiment config cannot drift. The base weights are resolved from the
# pinned revision recorded in the config.
set -euo pipefail

mode="${1:-}"
case "$mode" in
    pilot|formal|resume) ;;
    *)
        printf 'Usage: %s {pilot|formal|resume <checkpoint_dir> <output_dir>}\n' "$0" >&2
        exit 2
        ;;
esac
if [[ "$mode" == "resume" && $# -ne 3 ]]; then
    printf 'Usage: %s resume <checkpoint_dir> <output_dir>\n' "$0" >&2
    exit 2
fi

if [[ $(uname -s) != Linux ]]; then
    printf '%s\n' 'Run this script on a Linux GPU host or inside WSL2.' >&2
    exit 1
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

config="${CONFIG:-configs/experiment.json}"
dataset_root="${DATASET_ROOT:-data/libero_object_train}"
base_model_dir="${BASE_MODEL_DIR:-data/smolvla_base}"

read_cfg() {
    python3 - "$config" "$1" <<'PY'
import json, sys
from functools import reduce
value = json.load(open(sys.argv[1], encoding="utf-8"))
for key in sys.argv[2].split("."):
    value = value[key]
print(value)
PY
}

repo_id="$(read_cfg model.repo_id)"
revision="$(read_cfg model.revision)"
train_repo_id="course/libero_object_train"
seed="$(read_cfg training.seed)"
rank="$(read_cfg training.rank)"
alpha="$(read_cfg training.alpha)"
batch_size="$(read_cfg training.batch_size)"
log_freq="$(read_cfg training.log_every_steps)"

if [[ "$mode" == pilot ]]; then
    steps="$(read_cfg training.pilot_steps)"
    save_freq="$steps"
    warmup="$((steps / 10))"
    decay_steps="$steps"
    output_dir="outputs/train/pilot_s${seed}"
elif [[ "$mode" == formal ]]; then
    steps="$(read_cfg training.formal_steps)"
    save_freq="$(read_cfg training.save_every_steps)"
    warmup="$(read_cfg training.warmup_steps)"
    decay_steps="$steps"
    output_dir="outputs/train/formal_s${seed}"
else
    # resume: <checkpoint_dir> <output_dir>
    resume_checkpoint="${2%/}"
    output_dir="${3%/}"
    resume_config="$resume_checkpoint/pretrained_model/train_config.json"
    if [[ ! -f "$resume_config" ]]; then
        printf 'No train_config.json under %s/pretrained_model\n' "$resume_checkpoint" >&2
        exit 1
    fi
    # Sizes and hyper-parameters come from the checkpoint itself; only the new
    # output directory is overridden so the paused run is not overwritten.
    steps="$(python3 - "$resume_config" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["steps"])
PY
)"
fi

if [[ -e "$output_dir" ]]; then
    printf 'Output directory already exists: %s\nPick another run name or move it away.\n' "$output_dir" >&2
    exit 1
fi
if [[ ! -d "$dataset_root" ]]; then
    printf 'Training dataset missing: %s\nRun: python scripts/prepare_data.py --export-train\n' "$dataset_root" >&2
    exit 1
fi

# Resolve the base checkpoint at the pinned revision. Prefer an existing local
# copy; otherwise download it into data/smolvla_base.
if [[ ! -f "$base_model_dir/config.json" ]]; then
    mkdir -p "$base_model_dir"
    if command -v hf >/dev/null; then
        hf download "$repo_id" --revision "$revision" --local-dir "$base_model_dir"
    else
        python3 - "$repo_id" "$revision" "$base_model_dir" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])
PY
    fi
fi

# The LoRA adapter and the normalization processors are small; keep the base
# weights out of the output directory to avoid duplicating ~2 GB per run.
# lerobot-train refuses to write into an existing directory, so the training
# directory is created by it; the console log lives next to it.
log_file="${output_dir}.log"

if [[ "$mode" == resume ]]; then
    # The checkpoint's train_config.json is authoritative for step count,
    # optimizer/scheduler state and data order; pass it via --config_path.
    lerobot-train \
        --config_path="$resume_config" \
        --resume=true \
        --output_dir="$output_dir" 2>&1 | tee "$log_file"
else
    lerobot-train \
        --policy.path="$base_model_dir" \
        --policy.device=cuda \
        --policy.push_to_hub=false \
        --policy.input_features=null \
        --policy.output_features=null \
        --policy.optimizer_lr="$(read_cfg training.learning_rate)" \
        --policy.scheduler_decay_lr="$(read_cfg training.final_learning_rate)" \
        --policy.scheduler_warmup_steps="$warmup" \
        --policy.scheduler_decay_steps="$decay_steps" \
        --dataset.repo_id="$train_repo_id" \
        --dataset.root="$dataset_root" \
        --dataset.video_backend="$(read_cfg dataset.video_backend)" \
        --dataset.use_imagenet_stats=false \
        --peft.method_type="$(read_cfg training.method)" \
        --peft.r="$rank" \
        --peft.lora_alpha="$alpha" \
        --steps="$steps" \
        --batch_size="$batch_size" \
        --seed="$seed" \
        --env_eval_freq=0 \
        --save_freq="$save_freq" \
        --log_freq="$log_freq" \
        --num_workers=2 \
        --wandb.enable=false \
        --output_dir="$output_dir" 2>&1 | tee "$log_file"
fi

printf '\nRun finished: %s\n' "$output_dir"
