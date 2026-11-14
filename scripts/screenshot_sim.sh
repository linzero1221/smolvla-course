#!/usr/bin/env bash
cd /mnt/d/AiCoding/VLA/.claude/worktrees/desktop-codex-lin-wsl-setup-ba03f133
source "$HOME/.venvs/smolvla-course/bin/activate"
export HF_ENDPOINT=https://hf-mirror.com
export MUJOCO_GL=egl

clear
echo "=============================================================="
echo " LIBERO-Object closed-loop evaluation - live run"
echo " $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo " checkpoint: formal_s42_resume/checkpoints/020000 (LoRA adapter)"
echo " suite     : libero_object   task: alphabet_soup   init state: 5"
echo " protocol  : 256x256, relative control, 10 action steps/chunk"
echo "=============================================================="
echo
python scripts/evaluate.py \
    --checkpoint outputs/train/formal_s42_resume/checkpoints/020000 \
    --conditions seen_original \
    --max-episodes 1 \
    --output-dir outputs/eval/screenshot_demo
echo
echo "=== episode finished; CSV written under outputs/eval/screenshot_demo ==="
sleep 900
