#!/usr/bin/env bash
set -euo pipefail

if [[ $(uname -s) != Linux ]]; then
    printf '%s\n' 'Run this script in a Linux GPU environment or WSL2.' >&2
    exit 1
fi

if ! command -v lerobot-eval >/dev/null; then
    printf '%s\n' 'Install the LeRobot dependencies from README.md and activate the environment.' >&2
    exit 1
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p outputs/smoke
output_dir="outputs/smoke/$(date -u +%Y%m%dT%H%M%SZ)-$$"

lerobot-eval \
    --policy.path=lerobot/smolvla_libero \
    --policy.device=cuda \
    --policy.n_action_steps=10 \
    --env.type=libero \
    --env.task=libero_object \
    --env.task_ids='[0]' \
    --env.observation_width=256 \
    --env.observation_height=256 \
    --env.control_mode=relative \
    --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
    --eval.batch_size=1 \
    --eval.n_episodes=1 \
    --seed=42 \
    --output_dir="$output_dir" 2>&1 | tee "$output_dir.log"
