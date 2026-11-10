# 轻量 SmolVLA 微调与泛化评估

用 LeRobot 开源数据微调约 450M 参数的 SmolVLA，在 LIBERO 仿真中分别测试英文指令改写和未见物体。主实验使用 3 个训练任务、2 个留出任务和 110 个正式测试回合。

- [项目计划](docs/PROJECT_PLAN.md)记录排期、数据划分、训练与评测流程、验收条件和课程交付物。
- [组员与 AI 协作约定](AGENTS.md)记录分工、Git 工作流、实验协议与交接要求。
- [实验配置](configs/experiment.json)保存物体、改写指令、初始状态划分和首轮训练参数。
- [仿真检查脚本](scripts/smoke_eval.sh)使用公开的 LIBERO 权重执行一个闭环仿真回合，并保存日志与评测输出。

当前是项目初始化阶段。正式数据准备、训练、评估与报告按计划实施。本机检测到 RTX 5060 8 GB，仿真脚本需要 Linux GPU 环境，尚未在本机跑过闭环。

## 第一次运行

在装有 NVIDIA 驱动的 Linux GPU 主机或支持 GPU 的 WSL2 发行版中执行。以下以 Python 3.12 和 LeRobot 0.6.1 为起始版本。准备约 30 GB 工作空间。若使用 WSL2，建议把项目和模型缓存放在 Linux 文件系统。

```bash
conda create -y -n smolvla-course python=3.12
conda activate smolvla-course
python -m pip install 'lerobot[training,evaluation,smolvla,libero,peft]==0.6.1'
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda"))'
bash scripts/smoke_eval.sh
```

以终端日志、`outputs/smoke/` 下的 `eval_info.json` 和回放为首日验收依据。仿真完成一回合即通过环境检查，任务得分用于辅助诊断。若 CUDA 或渲染初始化失败，先按计划中的环境故障处理排查。 [LeRobot 0.6.1 安装指南](https://huggingface.co/docs/lerobot/v0.6.1/en/installation) · [LIBERO 仿真指南](https://huggingface.co/docs/lerobot/v0.6.1/en/libero)

## 工程目录

```text
VLA/
├── AGENTS.md
├── README.md
├── configs/
│   └── experiment.json
├── docs/
│   └── PROJECT_PLAN.md
├── scripts/
│   └── smoke_eval.sh       # 已提供的环境检查入口
├── data/                  # 本地数据、物体审计与数据划分清单
├── outputs/               # 环境记录、训练、checkpoint、评估和图表
└── deliverables/
    ├── report/            # 课程报告
    ├── slides/            # 汇报 PPTX 或 PDF
    └── evidence/          # 运行截图、曲线、结果表与回放
```
