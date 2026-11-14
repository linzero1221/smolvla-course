#!/usr/bin/env bash
cd /mnt/d/AiCoding/VLA
source "$HOME/.venvs/smolvla-course/bin/activate"
clear
echo "=============================================================="
echo " SmolVLA Course Project - Environment Verification"
echo " $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "=============================================================="
echo
echo "-------------------- nvidia-smi ------------------------------"
nvidia-smi | head -12
echo
echo "-------------------- CUDA / PyTorch --------------------------"
python -c "
import torch, lerobot
print(f'PyTorch     : {torch.__version__}')
print(f'CUDA build  : {torch.version.cuda}')
print(f'CUDA avail  : {torch.cuda.is_available()}')
print(f'GPU         : {torch.cuda.get_device_name(0)}')
print(f'GPU memory  : {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB')
print(f'LeRobot     : {lerobot.__version__}')
x = torch.ones(2, 2, device='cuda')
print(f'Tensor test : sum={x.sum().item():.1f} on {x.device}')
"
echo
echo "-------------------- Installed components --------------------"
python -c "
import importlib.metadata as m
for p in ['lerobot','torch','mujoco','robosuite','hf-libero','bddl','robomimic']:
    try: print(f'{p:<12} {m.version(p)}')
    except Exception: print(f'{p:<12} n/a')
"
echo
sleep 900
