#!/usr/bin/env bash
cd /mnt/d/AiCoding/VLA/.claude/worktrees/desktop-codex-lin-wsl-setup-ba03f133
source "$HOME/.venvs/smolvla-course/bin/activate"
export HF_ENDPOINT=https://hf-mirror.com

clear
echo "=============================================================="
echo " SmolVLA LoRA Training - live demo run"
echo " $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo " base  : lerobot/smolvla_base @ 5e8d12a6"
echo " data  : data/libero_object_train (136 episodes / 19445 frames)"
echo " LoRA  : r=32 alpha=64  batch=1  lr 1e-3 -> 1e-4  seed=42"
echo " this demo runs 300 steps to show the live progress bar"
echo "=============================================================="
echo
echo "[GPU before start]"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo

rm -rf outputs/train/screenshot_demo_s42
lerobot-train \
    --policy.path=data/smolvla_base \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.input_features=null \
    --policy.output_features=null \
    --policy.optimizer_lr=0.001 \
    --policy.scheduler_decay_lr=0.0001 \
    --policy.scheduler_warmup_steps=30 \
    --policy.scheduler_decay_steps=300 \
    --dataset.repo_id=course/libero_object_train \
    --dataset.root=data/libero_object_train \
    --dataset.video_backend=pyav \
    --dataset.use_imagenet_stats=false \
    --peft.method_type=LORA --peft.r=32 --peft.lora_alpha=64 \
    --steps=300 --batch_size=1 --seed=42 --env_eval_freq=0 \
    --save_freq=300 --log_freq=50 --num_workers=2 --wandb.enable=false \
    --output_dir=outputs/train/screenshot_demo_s42 2>&1 | tail -n 30

echo
echo "=== demo finished, GPU after training ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
sleep 900
