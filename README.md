# 轻量 SmolVLA 微调与泛化评估

用 LeRobot 开源数据微调约 450M 参数的 SmolVLA，在 LIBERO 仿真中分别测试英文指令改写和未见物体。主实验使用 3 个训练任务、2 个留出任务和 110 个正式测试回合。

- [项目计划](docs/PROJECT_PLAN.md)记录排期、数据划分、训练与评测流程、验收条件和课程交付物。
- [组员与 AI 协作约定](AGENTS.md)记录分工、Git 工作流、实验协议与交接要求。
- [实验配置](configs/experiment.json)保存物体、改写指令、初始状态划分和首轮训练参数。
- [仿真检查脚本](scripts/smoke_eval.sh)使用公开的 LIBERO 权重执行一个闭环仿真回合，并保存日志与评测输出。
- [数据准备脚本](scripts/prepare_data.py)生成任务映射、物体审计和只含三个已见任务的训练子集。
- [训练脚本](scripts/train.sh)、[评测脚本](scripts/evaluate.py)和[汇总脚本](scripts/summarize.py)覆盖 200 步试跑、20k 正式训练、checkpoint 选择与 110 回合正式评测。

本机已安装 WSL2 和 Ubuntu 24.04，并在 Ubuntu 中通过 RTX 5060 的 CUDA 驱动检查。首个 LIBERO 单回合仿真已完成：`outputs/smoke/20261110T080108Z-355/` 下的 `eval_info.json` 与 280 帧回放，环境记录见 `outputs/environment/smoke-eval-2026-11-10.txt`。该回合只用于环境验收，任务得分不进入正式结果。

## 第一次运行

在装有 NVIDIA 驱动的 Linux GPU 主机或支持 GPU 的 WSL2 发行版中执行。以下以 Python 3.12 和 LeRobot 0.6.1 为起始版本。准备约 30 GB 工作空间。若使用 WSL2，建议把项目和模型缓存放在 Linux 文件系统。

Windows 用户使用 WSL2 时，按专门的步骤安装 Ubuntu、配置 CUDA 版 PyTorch 并执行首个回合。D 盘安装位置和作业完成后的清理步骤也记录在同一份文档中。[WSL2 安装与卸载说明](docs/WSL_SETUP.md)

以下是已有 Linux GPU 主机的 Conda 入口。

```bash
conda create -y -n smolvla-course python=3.12
conda activate smolvla-course
python -m pip install 'lerobot[training,evaluation,smolvla,libero,peft]==0.6.1'
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda"))'
bash scripts/smoke_eval.sh
```

以终端日志、`outputs/smoke/` 下的 `eval_info.json` 和回放为首日验收依据。仿真完成一回合即通过环境检查，任务得分用于辅助诊断。若 CUDA 或渲染初始化失败，先按计划中的环境故障处理排查。 [LeRobot 0.6.1 安装指南](https://huggingface.co/docs/lerobot/v0.6.1/en/installation) · [LIBERO 仿真指南](https://huggingface.co/docs/lerobot/v0.6.1/en/libero)

首次运行 LIBERO 时会询问是否自定义数据集目录。输入 `n` 使用默认位置即可；本项目已在这台机器上生成 `~/.libero/config.yaml`。若 HuggingFace 直连不稳定，可设置 `HF_ENDPOINT=https://hf-mirror.com` 后重试。

## 数据、训练与评测

```bash
# 1. 任务映射、物体审计、训练子集（136 episodes / 19445 frames / 3 tasks）
python scripts/prepare_data.py --config configs/experiment.json --export-train

# 2. 200 步试跑，检查 loss、显存和 checkpoint 重载
bash scripts/train.sh pilot

# 3. 20k 步正式训练（每 5k 保存）
bash scripts/train.sh formal

# 4. 用验证初始状态 0–4 选 checkpoint
python scripts/evaluate.py --checkpoint outputs/train/formal_s42/checkpoints/020000 --validation --no-video

# 5. 正式评测（30 + 60 + 20 = 110 回合）
python scripts/evaluate.py --checkpoint outputs/train/formal_s42/checkpoints/020000 --output-dir outputs/eval/formal

# 6. 汇总与出图
python scripts/summarize.py --episodes outputs/eval/formal/formal_s42_020000/episodes.csv
```

`bash scripts/reproduce.sh --mode smoke` 串联一次 200 步自训与 3 回合评测；`--mode full` 串联数据准备、正式训练、checkpoint 选择、110 回合评测与汇总。

## 工程目录

```text
VLA/
├── AGENTS.md
├── README.md
├── configs/
│   └── experiment.json
├── docs/
│   ├── PROJECT_PLAN.md
│   └── WSL_SETUP.md
├── scripts/
│   ├── smoke_eval.sh       # 环境检查入口
│   ├── prepare_data.py     # 任务映射、物体审计、训练子集导出
│   ├── train.sh            # pilot / formal 训练封装
│   ├── evaluate.py         # 固定初始状态与指令替换的闭环评测
│   ├── summarize.py        # 汇总表、语言配对与图表
│   └── reproduce.sh        # smoke / full 一键复现
├── data/                  # 本地数据、物体审计与数据划分清单
├── outputs/               # 环境记录、训练、checkpoint、评估和图表
└── deliverables/
    ├── report/            # 课程报告
    ├── slides/            # 汇报 PPTX 或 PDF
    └── evidence/          # 运行截图、曲线、结果表与回放
```
