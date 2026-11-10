# 轻量 SmolVLA 微调课程项目执行计划

**拟定周期** 9 个工作日。**实验环境** Linux 单卡 GPU，优先使用已有环境。**主交付** 10–20 页课程报告、15 分钟汇报、可复现代码包和运行记录。

## 1. 研究问题与实验范围

用 LeRobot 的 LIBERO 示范数据对约 450M 参数的 `lerobot/smolvla_base` 做 LoRA 微调。限定机械臂执行抓取并放入篮子的操作。分别测量相同任务换一种英文表达时的性能变化，以及微调数据里从未出现过的物体在对应仿真场景中的任务成功率。单卡约束下先完成一个正式训练种子和三组正式评测。 [SmolVLA 论文](https://arxiv.org/abs/2506.01844) · [LeRobot 0.6.1 PEFT 文档](https://huggingface.co/docs/lerobot/v0.6.1/en/peft_training)

| 外部项目 | 本项目如何使用 |
|---|---|
| [Hugging Face LeRobot](https://github.com/huggingface/lerobot/tree/v0.6.1) | 复用模型、数据集、训练器、环境和动作处理器，固定起始版本 `v0.6.1`。 |
| [原版 LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/master/libero/libero/bddl_files/libero_object) | 检查 BDDL 中目标、干扰物和成功谓词，构造可审计的物体划分。 |
| [felixzheng02/smolvla](https://github.com/felixzheng02/smolvla) | 参考推理时替换指令、组织回合结果及生成对比图的方法。 |
| [wycliffeoleti/smolVLA](https://github.com/wycliffeoleti/smolVLA) | 参考 8 GB 显存上的 LoRA 配置和日志。仓库结果使用另一套数据与评测协议。 |
| [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) · [LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO) | 用于讨论指令和物体扰动的相关工作。 |

## 2. 环境与首日验收

1. 创建 Python 3.12 环境，验证 `nvidia-smi`、CUDA 张量和 LeRobot CLI。本机的 Ubuntu 24.04 已在 WSL2 中通过 CUDA 驱动检查。按所用环境选择步骤。[Linux GPU 安装入口](../README.md#第一次运行) · [WSL2 安装与验收](WSL_SETUP.md)
2. 执行 `bash scripts/smoke_eval.sh`。它用公开的 `lerobot/smolvla_libero` 权重跑一个原版 LIBERO-Object 回合。检查两路相机画面、机械臂运动、评测 JSON 和 MP4。在 `deliverables/evidence/` 保存一张包含环境与成功执行回合的截图。
3. 将 `nvidia-smi`、`python --version`、`python -m pip freeze`、安装命令和运行时间保存到 `outputs/environment/`。记录 LeRobot、MuJoCo、robosuite、`hf-libero` 版本。解析基础模型、数据集和环境资源的 revision，填入实验配置或环境记录。
4. 若 CUDA 张量失败，按固定版本安装文档检查 Blackwell 对应的 PyTorch CUDA 构建。若 CUDA 正常而 EGL 初始化失败，在 Linux 上测试 `MUJOCO_GL=osmesa bash scripts/smoke_eval.sh`，需已有可用的 OSMesa 库。若半天仍无法生成回放，切到可用的 Linux GPU 主机，保留失败日志。 [LeRobot 安装说明](https://huggingface.co/docs/lerobot/v0.6.1/en/installation) · [LIBERO 环境说明](https://huggingface.co/docs/lerobot/v0.6.1/en/libero)

## 3. 数据准备与物体划分

主实验使用 `lerobot/libero` 视频版，只读取原版 LIBERO-Object 中匹配的任务。实验配置已记录数据集和基础模型的 Hub commit SHA，复现时按配置中的版本下载。需要核对这些版本是否仍可访问时，运行以下命令。[实验配置](../configs/experiment.json) · [LIBERO 数据格式与版本说明](https://huggingface.co/docs/lerobot/v0.6.1/en/libero)

```bash
python - <<'PY'
import json
from pathlib import Path

from huggingface_hub import HfApi

config = json.loads(Path("configs/experiment.json").read_text(encoding="utf-8"))
api = HfApi()
for name, lookup in (("dataset", api.dataset_info), ("model", api.model_info)):
    entry = config[name]
    info = lookup(entry["repo_id"], revision=entry["revision"])
    assert info.sha == entry["revision"], (name, info.sha)
    print(name, info.sha)
PY
```

| 集合 | 目标物体 | 用法 |
|---|---|---|
| 已见物体 | `alphabet_soup`、`cream_cheese`、`salad_dressing` | 选取三项任务的全部合格示范轨迹微调。仿真中另外指定初始状态用于验证和测试。 |
| 未见物体 | `bbq_sauce`、`chocolate_pudding` | 这两项任务的轨迹、目标和干扰物身份均参与划分审计，两个留出物体都不能出现在训练场景中。用原版仿真场景做正式评估。 |

已实现 `scripts/prepare_data.py --config configs/experiment.json`（2026-11-10 跑通），步骤如下。

1. 用 `bddl_stem` 匹配当前仿真器里的 `task_id` 和原始英文指令。再用原始指令定位数据集任务，对照任务文本、轨迹样例和 BDDL。仿真 `task_id` 和数据集 `task_index` 分别记录。若匹配存在歧义，先查清再生成划分。
2. 导出 `data/task_map.csv` 和 `data/object_audit.csv`。物体审计以 BDDL 中的 `:objects` 为主，核对已见场景里两个留出物体均未出现。每个训练任务抽看至少三条轨迹的首帧和抓取段，确认目标和干扰物与任务定义相符。
3. 按完整 episode 导出 `data/train_episodes.csv`，至少记录 `source_episode_index`、`source_task_index`、`bddl_stem` 和 `frames`。使用 LeRobot 数据集工具导出只含训练轨迹的 `data/libero_object_train/`，检查筛选后本地数据集的 `meta/stats.json` 只由保留轨迹重新聚合或计算。 [LeRobot 数据集工具](https://huggingface.co/docs/lerobot/v0.6.1/en/using_dataset_tools)
4. 抽取一批训练数据核对双路 RGB 的顺序、8 维状态、7 维相对动作、英文任务文本、帧率和动作时序。把实际数据列与评估环境列的对应关系写入运行记录，确认手腕相机未与主相机互换。

**验收。**三项训练任务都有可读轨迹，导出数据只有这三项任务，留出物体与训练场景清单的交集为空。每项正式评测任务的环境初始状态库至少有 15 条可用状态。

## 4. 训练与选模型

**起始参数。**以 [实验配置](../configs/experiment.json) 为准。训练种子 42，LoRA `r=32`、`alpha=64`，batch size 1，学习率 `1e-3` 降到 `1e-4`。试跑 200 步，首轮正式训练 20k 步，每 5k 步保存一次。记录实际更新的模块和可训练参数量。 [官方 SmolVLA LoRA 示例](https://huggingface.co/docs/lerobot/v0.6.1/en/peft_training)

已实现 `scripts/train.sh`（pilot / formal 两种模式，参数从实验配置读取）。下列是供该脚本封装的上游 CLI 命令模板。先完成第 3 节的数据导出，并令 `MODEL_REVISION` 等于已记录的基础权重 SHA。正式 run 的输出路径应是尚未使用的新目录。

```bash
BASE_MODEL_DIR="$PWD/data/smolvla_base"
hf download lerobot/smolvla_base --revision "$MODEL_REVISION" --local-dir "$BASE_MODEL_DIR"

lerobot-train \
    --policy.path="$BASE_MODEL_DIR" \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.input_features=null \
    --policy.output_features=null \
    --policy.optimizer_lr=0.001 \
    --policy.scheduler_decay_lr=0.0001 \
    --policy.scheduler_warmup_steps=500 \
    --policy.scheduler_decay_steps=20000 \
    --dataset.repo_id=course/libero_object_train \
    --dataset.root=data/libero_object_train \
    --dataset.video_backend=pyav \
    --peft.method_type=LORA --peft.r=32 --peft.lora_alpha=64 \
    --steps=20000 --batch_size=1 --seed=42 --env_eval_freq=0 \
    --save_freq=5000 --log_freq=50 --wandb.enable=false \
    --output_dir=outputs/train/formal_s42
```

1. **试跑。**把步数与保存频率设为 200，把 warmup 设为 20，衰减步数设为 200。用独立目录 `outputs/train/pilot_s42`。验证 loss 有限、可保存 checkpoint，并从 checkpoint 重载 adapter 和归一化处理器完成至少一次已见任务仿真。记录实际步/秒、峰值显存、预计的 20k 步训练时长和评估单回合时长。
2. **正式训练。**从相同基础权重重新开始，保存环境、完整参数和日志。记录 5k、10k、15k、20k checkpoint。以逐步日志输出 `outputs/train/training_log.csv`，至少含 `step,loss,learning_rate,wall_time_s`。在训练配置中记下显存统计方式，避免混用 PyTorch 分配峰值和 `nvidia-smi` 的显卡总占用。
3. **选 checkpoint。**每个候选在三个已见物体任务上各跑 5 个固定验证初始状态，使用原始指令。选成功率最高的 checkpoint，平局取更早的 checkpoint。验证状态索引为 `0–4`，正式测试使用 `5–14`。冻结 checkpoint 路径和评测配置后再开跑正式测试。
4. **遇到问题。**显存不足时先保持 batch size 1、停止其它显存占用进程，训练和仿真分开执行。若已见任务验证全失败，优先核查相机名和旋转、状态与动作维度、归一化及 checkpoint 是否正确加载。链路正确且有剩余算力时，从原配置续训到 40k 步并记录变更。若仍低于可比较的成功水平，保留实测，分析哪些能力无法凭这次实验判定。

## 5. 正式评测、记录与分析

已实现 `scripts/evaluate.py --config configs/experiment.json --checkpoint <checkpoint>`（含 `--validation` 选模模式）。复用 LeRobot 的策略和环境处理器。对每个回合显式加载对应的固定初始状态并重置策略动作队列。在同义改写组只覆盖传入策略的任务文本，原任务、环境成功谓词和动作上限沿用原始任务。沿用官方仿真器终止设置，观测尺寸固定为 256×256，控制模式为相对动作，每次推理执行 10 个动作步。 [LIBERO 环境与评估配置](https://huggingface.co/docs/lerobot/v0.6.1/en/libero) · [LeRobot 环境源代码](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/envs/libero.py)

| 条件 ID | 测试内容 | 回合数 |
|---|---|---:|
| `seen_original` | 3 个已见物体 × 原句 × 10 个初始状态 | 30 |
| `seen_paraphrase` | 3 个已见物体 × 两条改写 × 10 个相同初始状态 | 60 |
| `unseen_original` | 2 个留出物体 × 原句 × 10 个初始状态 | 20 |

原句与改写按 `(task_id, init_state_id)` 配对，推理种子一致。逐回合输出 `outputs/eval/episodes.csv`，记录 `run_id,checkpoint,condition,task_id,object,init_state_id,prompt_id,prompt,seed,status,success,steps,video_path`。环境或推理异常单列记录，排查后对同一个回合键补跑。任务超时按任务失败计入结果。最终检查三个条件的完整回合数为 30、60、20，回合键唯一且无未处理异常。

已实现 `scripts/summarize.py` 输出 `summary.csv`、训练 loss 曲线、三组成功率柱状图和语言配对结果表。成功率按 `成功回合数 / 有效回合数` 计算，结果表同时给出分子和分母。改写效应报告每种改写相对原句的百分点变化，以及配对中由成功变失败和由失败变成功的次数。未见物体逐项报告，讨论目标类型和干扰物布局同时变化带来的影响。未见仅指本项目微调数据。任务目标区位置较固定，改写成功也可能来自对视觉布局的利用。

从视频中选 1 个成功和 2 个代表性失败，标注抓错物体、抓取滑落、未正确入篮或超时等现象。对同一个 checkpoint 的性能与显存结果做汇总。只有一个训练种子时，讨论结论限于该次训练和已选任务。

## 6. 九天排期与验收

| 时间 | 工作和负责人角色 | 可验收产物 |
|---|---|---|
| 第 1 天 | 环境负责人安装依赖，跑通官方 checkpoint 的单回合评估 | 环境记录、评估 JSON 和录像 |
| 第 2 天 | 数据负责人锁定版本、任务映射和 BDDL 审计，导出训练轨迹 | 任务映射、物体审计、episode 清单、训练子集统计 |
| 第 3 天 | 训练负责人试跑 200 步。评测负责人同时实现固定状态与指令替换 | 可重载的试跑权重、loss 日志、推理耗时和峰值显存 |
| 第 4 天 | 训练负责人启动 20k 步正式训练。评测负责人核对每回合记录及语言配对 | 正式配置、训练曲线原始数据和候选 checkpoint |
| 第 5 天 | 评测负责人按验证状态选 checkpoint，固定测试清单 | 验证结果、权重路径与正式测试配置 |
| 第 6 天 | 跑完 110 回合，汇总计数并挑选视频 | 逐回合 CSV、主结果表、图和典型回放 |
| 第 7–8 天 | 全组撰写报告、制作 PPT、清理复现流程，补齐截图 | 课程报告、PPT 初稿、代码包与运行证据 |
| 第 9 天 | 在干净环境检查复现入口，做一次计时彩排 | 可交版本和 15 分钟讲稿 |

已实现 `scripts/reproduce.sh --mode smoke` 和 `--mode full`。前者运行小规模自训与评估闭环，后者串联数据准备、正式训练、checkpoint 选择、测试与绘图。环境检查入口为 `bash scripts/smoke_eval.sh`，单回合验收产物在 `outputs/smoke/20261110T080108Z-355/`。

## 7. 课程交付与汇报要求

**报告约 14 页。**摘要 1 页、相关工作 2 页、方法 3 页、实验设置与结果 4 页、分析讨论 3 页、结论 1 页。正文描述数据划分、模型初始化、LoRA 设置、闭环成功指标和失败案例，表图使用同一份逐回合记录生成。

**代码包与运行证据。**README 含运行环境、安装步骤、数据与权重版本、复现命令和产物位置。打包训练、评估、统计脚本和配置。附环境运行截图、训练日志或曲线截图、仿真截图、训练曲线和结果表。大体积数据与公开权重通过固定版本的下载命令获取。

**汇报 12 页，共 15 分钟。**提交 PPTX 或 PDF。按实际成员填写职责和产物。

| 页码 | 内容 | 时间 |
|---|---|---:|
| 1–2 | 问题与背景，用仿真画面对比原句、同义改写和未见目标 | 2 分钟 |
| 3–5 | 方法与方案，讲 SmolVLA、LoRA、物体划分和实验控制 | 4 分钟 |
| 6–8 | 实验结果或分析，展示训练曲线、资源开销、成功率与回放 | 5 分钟 |
| 9 | 结论，逐一回答语言和物体两个研究问题 | 1 分钟 |
| 10 | 组内分工说明，列出姓名、工作与产物 | 1 分钟 |
| 11 | **主要贡献声明**，说明实际完成的工作并标明开源复用部分 | 1 分钟 |
| 12 | **局限与展望**，讨论样本量、单种子、场景差异和后续验证 | 1 分钟 |
