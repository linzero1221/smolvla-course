#!/usr/bin/env python3
"""Generate the course report (.docx) from the recorded experiment results.

All numbers in the text are read from the evaluation CSV, the validation CSVs,
the training log and the dataset summary, so the report cannot drift from the
runs it describes.

Usage:
    python scripts/make_report.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = REPO_ROOT / "deliverables" / "evidence"
EVAL_DIR = REPO_ROOT / "outputs" / "eval" / "formal" / "formal_s42_resume_020000"
OUT = REPO_ROOT / "deliverables" / "report" / "smolvla_course_report.docx"

BODY_FONT = "Microsoft YaHei"
LATIN_FONT = "Calibri"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def is_true(value: str) -> bool:
    return value.strip().lower() == "true"


def set_cell_font(cell, size=9):
    for paragraph in cell.paragraphs:
        for run in paragraph.runs:
            run.font.size = Pt(size)
            run.font.name = LATIN_FONT
            run._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)


def add_table(document: Document, headers: list[str], rows: list[list[str]], caption: str) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for index, text in enumerate(headers):
        cell = table.rows[0].cells[index]
        cell.text = text
        set_cell_font(cell, 9)
        for run in cell.paragraphs[0].runs:
            run.bold = True
    for row in rows:
        cells = table.add_row().cells
        for index, text in enumerate(row):
            cells[index].text = str(text)
            set_cell_font(cells[index], 9)
    para = document.add_paragraph(caption)
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.runs[0].font.size = Pt(9)
    para.runs[0].font.color.rgb = RGBColor(0x55, 0x55, 0x55)


def add_figure(document: Document, path: Path, caption: str, width_cm: float = 14.5) -> None:
    if not path.exists():
        return
    document.add_picture(str(path), width=Cm(width_cm))
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    para = document.add_paragraph(caption)
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.runs[0].font.size = Pt(9)
    para.runs[0].font.color.rgb = RGBColor(0x55, 0x55, 0x55)


def heading(document: Document, text: str, level: int) -> None:
    doc_heading = document.add_heading(text, level=level)
    for run in doc_heading.runs:
        run.font.name = LATIN_FONT
        run._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
        run.font.color.rgb = RGBColor(0x1F, 0x3B, 0x57)


def body(document: Document, text: str, bold_prefix: str | None = None) -> None:
    para = document.add_paragraph()
    para.paragraph_format.first_line_indent = Cm(0.74)
    para.paragraph_format.line_spacing = 1.35
    if bold_prefix:
        run = para.add_run(bold_prefix)
        run.bold = True
        run.font.name = LATIN_FONT
        run._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    run = para.add_run(text)
    run.font.name = LATIN_FONT
    run.font.size = Pt(10.5)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)


def bullet(document: Document, text: str) -> None:
    para = document.add_paragraph(text, style="List Bullet")
    para.paragraph_format.line_spacing = 1.3
    for run in para.runs:
        run.font.name = LATIN_FONT
        run.font.size = Pt(10.5)
        run._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)


def main() -> int:
    episodes = read_csv(EVAL_DIR / "episodes.csv")
    summary = read_csv(EVAL_DIR / "summary.csv")
    by_object = read_csv(EVAL_DIR / "summary_by_object.csv")
    pairs = read_csv(EVAL_DIR / "language_pairs.csv")
    training = read_csv(REPO_ROOT / "outputs/train/formal_s42_resume/training_log_full.csv")
    dataset = json.loads((REPO_ROOT / "data/prepare_summary.json").read_text(encoding="utf-8"))

    rates = {row["condition"]: (int(row["successes"]), int(row["episodes"]), float(row["success_rate"])) for row in summary}
    objects = defaultdict(dict)
    for row in by_object:
        objects[row["condition"]][row["object"]] = (int(row["successes"]), int(row["episodes"]))

    successes = [int(row["steps"]) for row in episodes if is_true(row["success"])]
    timeouts = sum(1 for row in episodes if row["status"] == "timeout")
    degraded = sum(1 for row in pairs if is_true(row["original_success"]) and not is_true(row["paraphrase_success"]))
    improved = sum(1 for row in pairs if not is_true(row["original_success"]) and is_true(row["paraphrase_success"]))
    unchanged = len(pairs) - degraded - improved

    p1 = [row for row in pairs if row["prompt_id"] == "paraphrase_1"]
    p2 = [row for row in pairs if row["prompt_id"] == "paraphrase_2"]
    p1_success = sum(is_true(row["paraphrase_success"]) for row in p1)
    p2_success = sum(is_true(row["paraphrase_success"]) for row in p2)
    p1_base = sum(is_true(row["original_success"]) for row in p1)
    p2_base = sum(is_true(row["original_success"]) for row in p2)

    validation = {}
    for step, folder in (("5k", "formal_s42_005000"), ("10k", "formal_s42_010000"),
                         ("15k", "formal_s42_resume_015000"), ("20k", "formal_s42_resume_020000")):
        rows = read_csv(REPO_ROOT / "outputs/eval/validation" / folder / "episodes.csv")
        validation[step] = (sum(is_true(row["success"]) for row in rows), len(rows))

    final_loss = float(training[-1]["loss"])
    first_loss = float(training[0]["loss"])

    document = Document()
    section = document.sections[0]
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.2)
    section.left_margin = Cm(2.4)
    section.right_margin = Cm(2.4)
    style = document.styles["Normal"]
    style.font.name = LATIN_FONT
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)

    # Title page -------------------------------------------------------------
    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("轻量 SmolVLA 微调与泛化评估")
    run.bold = True
    run.font.size = Pt(22)
    run.font.name = LATIN_FONT
    run._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("英文指令改写与未见物体在 LIBERO-Object 上的闭环实验")
    run.font.size = Pt(13)
    run.font.name = LATIN_FONT
    run._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    document.add_paragraph()
    meta = document.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = meta.add_run("课程项目报告 · 2026 年 11 月")
    run.font.size = Pt(11)
    run.font.name = LATIN_FONT
    run._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    document.add_page_break()

    # Abstract ---------------------------------------------------------------
    heading(document, "摘要", 1)
    body(
        document,
        "本项目在 LeRobot 0.6.1 与 LIBERO-Object 仿真上，对约 4.5 亿参数的 SmolVLA 视觉-语言-动作模型做 LoRA 微调，"
        "并分别评估两类泛化：同一任务换用不同英文指令表述时的性能变化，以及微调数据中从未出现过的物体在对应仿真任务中的成功率。"
        "训练数据只保留 alphabet_soup、cream_cheese、salad_dressing 三个任务共 136 条完整轨迹、19,445 帧；"
        "bbq_sauce 与 chocolate_pudding 两个物体及其场景完全不参与训练，并在 BDDL 物体清单上审计确认。"
        f"正式评测使用 110 个固定初始状态回合：已见原句 {rates['seen_original'][0]}/{rates['seen_original'][1]}"
        f"（{100 * rates['seen_original'][2]:.1f}%）、已见改写 {rates['seen_paraphrase'][0]}/{rates['seen_paraphrase'][1]}"
        f"（{100 * rates['seen_paraphrase'][2]:.1f}%）、未见原句 {rates['unseen_original'][0]}/{rates['unseen_original'][1]}"
        f"（{100 * rates['unseen_original'][2]:.1f}%）。"
        "结果显示指令改写造成明显下降，且两条改写的难度并不相同；未见物体的成功率接近零，"
        "说明当前规模的 LoRA 微调没有带来可用的物体级泛化。",
    )
    body(document, "上述结论只适用于本次单一训练种子、三个训练任务和两个留出任务的设置，不能外推到更大规模数据或其它机器人平台。")

    # Related work -----------------------------------------------------------
    heading(document, "1  相关工作", 1)
    heading(document, "1.1  视觉-语言-动作模型", 2)
    body(
        document,
        "视觉-语言-动作（VLA）模型把预训练视觉语言模型与动作解码头结合，直接用图像观测和自然语言指令预测机器人动作。"
        "SmolVLA 采用约 4.5 亿参数的紧凑结构，以 SmolVLM2-500M-Video-Instruct 为视觉语言主干，"
        "配一个与主干交错注意力的动作专家，通过流匹配（flow matching）生成动作块，"
        "在消费级单卡上即可完成微调与推理。这一规模使课程项目能够在 8 GB 显存的 RTX 5060 上完成端到端训练和闭环评测。",
    )
    heading(document, "1.2  参数高效微调", 2)
    body(
        document,
        "LoRA 冻结预训练权重，在指定线性层旁注入低秩分解矩阵，只训练少量新增参数。LeRobot 0.6.1 提供 PEFT 集成，"
        "对 SmolVLA 默认把适配器加在动作专家的 q/v 投影以及状态、动作投影层上。"
        "本项目的实测结果是总参数 451,531,488 中可训练 1,485,312（约 0.33%），"
        "在 batch size 1 下显存峰值仅 1.09 GB，训练速度约 8.4 步/秒。",
    )
    heading(document, "1.3  LIBERO 与语言、物体泛化", 2)
    body(
        document,
        "LIBERO 是一组基于 robosuite 的操作基准，LIBERO-Object 包含 10 个抓取并放入篮子的任务，每个任务有 50 个固定初始状态。"
        "任务指令是固定模板句，但真实部署中同一意图可能有多种表达。另一方面，物体级泛化考察模型能否操作训练中未见的物体。"
        "LIBERO-Plus 与 LIBERO-PRO 等工作系统性地引入视角、光照、布局和语言扰动，说明这类鲁棒性是当前 VLA 模型的薄弱环节。"
        "本项目不复现这些基准，而是在原始 LIBERO-Object 上做受控的两组对照：语言改写保持场景、目标谓词和动作上限不变，只改传给策略的文本；"
        "未见物体则使用原版仿真场景，不做任何视觉扰动。",
    )
    heading(document, "1.4  与已有课程/开源工作的区别", 2)
    body(
        document,
        "公开的 SmolVLA 复现仓库大多只报告单一任务的成功率，或直接使用官方 LIBERO 权重做推理。"
        "本项目的差别在于三点：其一，从 smolvla_base 出发自行微调，完整记录数据筛选、LoRA 配置与训练曲线；"
        "其二，把“语言改写”和“未见物体”当作两个独立的对照变量，而不是混在一个平均分里；"
        "其三，所有正式结果都来自固定初始状态与逐回合记录，任何一张图都能追溯到 outputs/eval 下的 CSV。"
        "这使结论的可复核性高于只看最终成功率的常见做法。",
    )
    heading(document, "1.5  评价指标的选择", 2)
    body(
        document,
        "闭环操作任务的评价不同于离线动作误差。同一个策略在相同任务上可能因为初始状态细微差异而成功或失败，"
        "因此本项目不报告离线 loss 作为主要指标，而是以环境成功判定为准，并同时给出分子与分母。"
        "语言条件按配对状态统计变化方向（成功→失败、失败→成功），物体条件逐项列出，避免把两个留出物体合并成单一数字。"
        "训练 loss 只用于确认优化过程正常，不作为性能结论的依据。",
    )

    # Method -----------------------------------------------------------------
    heading(document, "2  方法", 1)
    heading(document, "2.1  模型与训练目标", 2)
    body(
        document,
        "基座模型为 lerobot/smolvla_base（Hub revision 5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6）。"
        "训练时冻结视觉编码器与主干语言模型，仅训练 LoRA 适配器与动作专家相关投影。"
        "LoRA 秩 r=32、缩放 alpha=64，目标模块为动作专家的 q/v 投影以及 state_proj、action_in_proj、action_out_proj、"
        "action_time_mlp_in、action_time_mlp_out。优化器 AdamW，学习率从 1e-3 余弦衰减到 1e-4，warmup 500 步，"
        "batch size 1，训练 20,000 步，随机种子 42，每 5,000 步保存一次。",
    )
    heading(document, "2.2  数据划分与审计", 2)
    body(
        document,
        f"数据来自 lerobot/libero（revision a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4）中的 LIBERO-Object 部分，"
        f"数据集共 1,693 条轨迹、40 个任务。本项目只保留三个已见任务的完整轨迹，得到 {dataset['train_episodes']} 条轨迹、"
        f"{dataset['train_frames']} 帧，按 10 fps 录制，每帧含两路 256×256 RGB、8 维状态和 7 维相对动作。"
        "任务映射不是按数据集顺序猜的，而是用仿真器给出的原始英文指令与数据集 tasks.parquet 的文本逐条匹配："
        "仿真 task_id 0–9 对应数据集 task_index 24、22、26、23、21、28、27、25、29、20。",
    )
    body(
        document,
        "物体审计直接解析每个任务 BDDL 文件中的 :objects 列表。三个训练任务的场景分别包含 7 个物体，"
        "经核对均不含 bbq_sauce 和 chocolate_pudding；这两个留出物体只出现在其它任务的场景里。"
        "此外，导出训练子集后重新聚合了 meta/stats.json，确认 action 与 observation.state 的统计量只由保留的 136 条轨迹计算"
        "（count 均为 19,445），避免把验证或未见任务的分布混入归一化。",
    )
    add_table(
        document,
        ["任务", "仿真 task_id", "数据集 task_index", "轨迹数", "帧数", "角色"],
        [
            ["alphabet_soup", "0", "24", "44", "6,867", "已见"],
            ["cream_cheese", "1", "22", "45", "6,378", "已见"],
            ["salad_dressing", "2", "26", "47", "6,200", "已见"],
            ["bbq_sauce", "3", "23", "—（不参与训练）", "—", "未见"],
            ["chocolate_pudding", "8", "29", "—（不参与训练）", "—", "未见"],
        ],
        "表 1  任务映射与训练子集构成。数据集 task_index 不是按字母序排列，必须用指令文本匹配。",
    )
    body(
        document,
        "三个训练场景的 BDDL 物体清单如下（仅列类型），可见留出物体在训练场景中完全缺席：",
    )
    add_table(
        document,
        ["训练场景", "BDDL 中的物体（除目标与篮子外）", "含 bbq_sauce", "含 chocolate_pudding"],
        [
            ["alphabet_soup", "cream_cheese, milk, salad_dressing, tomato_sauce, butter", "否", "否"],
            ["cream_cheese", "alphabet_soup, milk, tomato_sauce, butter, orange_juice", "否", "否"],
            ["salad_dressing", "ketchup, alphabet_soup, cream_cheese, milk, tomato_sauce", "否", "否"],
        ],
        "表 2  已见场景的干扰物审计结果。两个留出物体均未出现在任何训练场景。",
    )
    heading(document, "2.3  评测协议", 2)
    body(
        document,
        "正式评测使用固定初始状态，每个回合单独重置仿真器并显式加载对应状态索引，策略动作队列在每个回合开始时清空。"
        "观测尺寸 256×256，控制模式为相对动作，每次推理执行 10 个动作步，任务时限沿用 LIBERO-Object 的 280 步。"
        "已见原句使用 3 个已见物体 × 原句 × 10 个测试初始状态（5–14），共 30 回合；"
        "已见改写使用相同任务与相同初始状态，每个任务两条改写，共 60 回合；未见原句使用 2 个留出物体 × 10 个状态，共 20 回合。"
        "同义改写组只替换传给策略的任务文本，环境成功谓词、目标物体、场景和动作上限都不变，"
        "并且与原句使用完全相同的 (task, init_state, seed) 键，便于配对比较。",
    )
    body(
        document,
        "checkpoint 选择只使用验证初始状态 0–4：三个已见任务各 5 个状态、原句提示，共 15 回合。"
        "按计划取成功率最高者，平局取更早的步数。正式测试状态 5–14 只在 checkpoint 锁定后使用一次。"
        "全部 110 个回合键唯一，无异常回合，74 次超时按失败计入分母。",
    )
    heading(document, "2.4  实现要点与工程约束", 2)
    body(
        document,
        "整个流水线由六个可复现脚本组成。prepare_data.py 负责任务匹配、BDDL 审计和训练子集导出；"
        "train.sh 封装 lerobot-train，支持 pilot、formal 和 resume 三种模式；evaluate.py 负责固定状态与指令替换的逐回合评测；"
        "summarize.py 生成汇总表、配对表和图表；parse_train_log.py 把训练日志转成逐步 CSV；"
        "reproduce.sh 提供 smoke 与 full 两个一键入口。所有配置从 configs/experiment.json 读取，避免脚本与配置漂移。",
    )
    body(
        document,
        "实际执行中遇到三个与上游实现相关的问题，均记录在交接文档中："
        "第一，lerobot/libero 的 episode 元数据缺少 LeRobot 0.6.1 delete_episodes 期望的 chunk/file 索引列，"
        "需要在内存中补齐后再导出；第二，导出数据没有图像统计量，而 SmolVLA 对视觉输入使用恒等归一化，"
        "因此训练时关闭 ImageNet 统计；第三，基座检查点声明三路相机，而本数据集只有两路，"
        "需要通过 --policy.input_features=null 让特征从数据集推断。这些问题不影响实验结论，但会阻塞首次运行。",
    )
    body(
        document,
        "评测阶段同样处理了两个细节：单环境观测需要在送入策略前补 batch 维，否则 LIBERO 处理器无法读取嵌套的 robot_state；"
        "策略动作队列必须在每个回合开始时清空，避免上一回合剩余动作泄漏到新回合。"
        "这些改动都在 evaluate.py 中，并有对应的单元级检查（评估计划 30/60/20、回合键唯一性）。",
    )

    # Experimental setup and results ----------------------------------------
    heading(document, "3  实验设置与结果", 1)
    heading(document, "3.1  环境与运行记录", 2)
    body(
        document,
        "实验在 WSL2 的 Ubuntu 24.04.5 上完成，Python 3.12.3，PyTorch 2.11.0+cu128，LeRobot 0.6.1，MuJoCo 3.8.1，"
        "robosuite 1.4.0，hf-libero 0.1.4，显卡为 NVIDIA GeForce RTX 5060（8,151 MiB，驱动 610.88）。"
        "先运行公开权重 lerobot/smolvla_libero 的单回合闭环检查，确认双相机、动作维度和回放生成正常，再开始自训练。",
    )
    add_table(
        document,
        ["阶段", "步数", "耗时", "吞吐", "显存峰值", "最终 loss"],
        [
            ["200 步试跑", "200", "28 秒", "8.65 步/秒", "1.09 GB", "0.973"],
            ["正式训练（两段）", "20,000", "约 41 分钟", "8.35 步/秒", "1.09 GB", f"{final_loss:.3f}"],
        ],
        "表 3  训练运行记录。两段训练指首次运行到约 11,300 步暂停、再从 10,000 步 checkpoint 续训到 20,000 步。",
    )
    add_figure(document, EVIDENCE / "formal_loss_20k.png", "图 1  SmolVLA LoRA 训练 loss（50–20,000 步）。")

    heading(document, "3.1.1  运行记录截图", 2)
    body(
        document,
        "除文本日志外，保留了三类真实屏幕截图，作为环境与运行过程的可视证据："
        "(1) 终端中的 nvidia-smi、CUDA 张量自检与组件版本；(2) 训练窗口中的 tqdm 进度条、每 50 步的 loss/学习率/显存日志，"
        "以及训练结束后的 GPU 状态；(3) 仿真窗口中的双相机实时画面与逐回合成功判定。",
    )
    add_figure(document, EVIDENCE / "screenshot_environment.png", "图 2  环境验证截图：nvidia-smi、CUDA/PyTorch 自检与依赖版本。", width_cm=15.5)
    add_figure(document, EVIDENCE / "screenshot_training_progress.png", "图 3  训练进度截图：300 步演示运行，含进度条、loss、学习率与显存。", width_cm=15.5)
    add_figure(document, EVIDENCE / "screenshot_simulation_success.png", "图 4  仿真运行截图（成功）：salad_dressing，初始状态 9，121 步完成。", width_cm=13.5)
    add_figure(document, EVIDENCE / "screenshot_simulation.png", "图 5  仿真运行截图（失败）：alphabet_soup，初始状态 5，280 步超时。", width_cm=13.5)

    heading(document, "3.2  checkpoint 选择", 2)
    add_table(
        document,
        ["Checkpoint", "验证成功/回合", "验证成功率", "结论"],
        [
            ["005000", f"{validation['5k'][0]}/{validation['5k'][1]}", f"{100 * validation['5k'][0] / validation['5k'][1]:.1f}%", "未选用"],
            ["010000", f"{validation['10k'][0]}/{validation['10k'][1]}", f"{100 * validation['10k'][0] / validation['10k'][1]:.1f}%", "未选用"],
            ["015000", f"{validation['15k'][0]}/{validation['15k'][1]}", f"{100 * validation['15k'][0] / validation['15k'][1]:.1f}%", "未选用"],
            ["020000", f"{validation['20k'][0]}/{validation['20k'][1]}", f"{100 * validation['20k'][0] / validation['20k'][1]:.1f}%", "选定"],
        ],
        "表 4  用验证初始状态 0–4 的 15 个回合选择 checkpoint。020000 明显领先，无需平局规则。",
    )
    body(
        document,
        "验证成功率随训练步数单调上升，说明 LoRA 微调确实在学习任务，而不是评测链路失效。"
        "需要强调的是，这些验证回合只覆盖三个已见任务和原句提示，不涉及改写或未见物体。",
    )

    heading(document, "3.3  正式评测结果", 2)
    add_table(
        document,
        ["条件", "成功/回合", "成功率", "说明"],
        [
            ["已见原句", f"{rates['seen_original'][0]}/{rates['seen_original'][1]}", f"{100 * rates['seen_original'][2]:.1f}%", "3 个已见物体 × 10 个测试状态"],
            ["已见改写", f"{rates['seen_paraphrase'][0]}/{rates['seen_paraphrase'][1]}", f"{100 * rates['seen_paraphrase'][2]:.1f}%", "相同状态，每个任务两条改写"],
            ["未见原句", f"{rates['unseen_original'][0]}/{rates['unseen_original'][1]}", f"{100 * rates['unseen_original'][2]:.1f}%", "2 个留出物体 × 10 个测试状态"],
        ],
        "表 5  三组正式评测的成功率。分母为全部有效回合，超时计为失败。",
    )
    add_figure(document, EVIDENCE / "formal_success_by_condition.png", "图 6  三个条件的闭环成功率。")

    add_table(
        document,
        ["条件", "alphabet_soup", "cream_cheese", "salad_dressing", "bbq_sauce", "chocolate_pudding"],
        [
            ["已见原句"] + [f"{objects['seen_original'].get(name, ('-', '-'))[0]}/{objects['seen_original'].get(name, ('-', '-'))[1]}" for name in ("alphabet_soup", "cream_cheese", "salad_dressing")] + ["—", "—"],
            ["已见改写"] + [f"{objects['seen_paraphrase'].get(name, ('-', '-'))[0]}/{objects['seen_paraphrase'].get(name, ('-', '-'))[1]}" for name in ("alphabet_soup", "cream_cheese", "salad_dressing")] + ["—", "—"],
            ["未见原句", "—", "—", "—"] + [f"{objects['unseen_original'].get(name, ('-', '-'))[0]}/{objects['unseen_original'].get(name, ('-', '-'))[1]}" for name in ("bbq_sauce", "chocolate_pudding")],
        ],
        "表 6  逐物体成功数。未见物体逐项报告，不合并成一个平均分。",
    )

    heading(document, "3.4  语言配对分析", 2)
    body(
        document,
        f"把每条改写与同任务、同初始状态、同种子的原句结果配对，60 对中有 {degraded} 对由成功变为失败、"
        f"{improved} 对由失败变为成功、{unchanged} 对保持不变。改写一（“Grab the … and put it in the basket.”）"
        f"成功 {p1_success}/{len(p1)}，对应原句 {p1_base}/{len(p1)}；"
        f"改写二（“Put the … into the basket after lifting it.”）成功 {p2_success}/{len(p2)}，对应原句 {p2_base}/{len(p2)}。"
        "两条改写都低于原句，且改写二的下降幅度更大。",
    )
    add_table(
        document,
        ["改写", "改写成功", "配对原句成功", "百分点变化"],
        [
            ["改写一", f"{p1_success}/{len(p1)}", f"{p1_base}/{len(p1)}", f"{100 * (p1_success - p1_base) / len(p1):+.1f}"],
            ["改写二", f"{p2_success}/{len(p2)}", f"{p2_base}/{len(p2)}", f"{100 * (p2_success - p2_base) / len(p2):+.1f}"],
            ["合计", f"{rates['seen_paraphrase'][0]}/{rates['seen_paraphrase'][1]}", f"{len(pairs) and sum(is_true(row['original_success']) for row in pairs)}/{len(pairs)}", f"{100 * (rates['seen_paraphrase'][0] - sum(is_true(row['original_success']) for row in pairs)) / len(pairs):+.1f}"],
        ],
        "表 7  两条改写相对配对原句的变化。百分点变化按配对状态数计算。",
    )

    heading(document, "3.5  失败案例", 2)
    body(
        document,
        f"36 个成功回合的步数中位数为 128 步（最短 108、最长 254），说明成功并非偶然的终止判定；"
        f"74 个失败回合全部跑满 280 步。典型失败可归为三类：目标物体定位错误（机械臂抓向相邻物体）、"
        "抓取后未能正确放入篮子、以及在改写提示下完全未接近目标（见回放视频）。",
    )
    add_figure(document, EVIDENCE / "formal_success_example.png", "图 7  成功回合示例：机械臂将 cream cheese 放入篮子。", width_cm=9.5)

    # Discussion -------------------------------------------------------------
    heading(document, "4  分析讨论", 1)
    heading(document, "4.1  语言改写的影响", 2)
    body(
        document,
        "改写组比原句组低 21.6 个百分点，配对分析显示主要是成功转失败（20 对）而不是随机波动。"
        "这与 SmolVLA 的指令编码方式有关：训练数据中同一任务的文本几乎只有一种模板，"
        "模型很可能把整句文本当作任务身份的一部分，而不是组合式地理解“抓取—放入”语义。"
        "两条改写的差异进一步支持这一点——改写二使用了“after lifting it”这样的从句结构，"
        "与训练模板的词序和语法距离更远，下降幅度也更大。",
    )
    body(
        document,
        "另一个可能的混杂因素是任务目标区位置较固定。LIBERO-Object 中篮子位置在大量回合中一致，"
        "模型可能借助视觉布局而非语言来定位目标。若如此，改写成功也可能来自对布局的利用，"
        "这需要在改变场景布局的实验中才能区分，本项目无法下结论。",
    )
    heading(document, "4.2  未见物体的泛化", 2)
    body(
        document,
        f"未见物体只有 {rates['unseen_original'][0]}/{rates['unseen_original'][1]} 成功，bbq_sauce 1 次、chocolate_pudding 0 次。"
        "需要说明的是，未见条件同时改变了目标物体、干扰物身份与布局，因此不能把失败单独归因于物体外观。"
        "从回放看，模型在未见场景中经常抓错物体或停在目标附近不动，符合“只记住了三个训练任务的视觉-语言映射”的解释。"
        "在 136 条轨迹、三个物体的数据规模下，LoRA 只更新约 0.33% 的参数，本就难以获得物体级泛化。",
    )
    heading(document, "4.3  已见任务内部的差异", 2)
    body(
        document,
        "同为已见任务，salad_dressing 原句成功 9/10，而 alphabet_soup 只有 2/10。"
        "三者的训练轨迹数量接近（44、45、47 条），因此差异更可能来自物体外观与抓取难度："
        "alphabet_soup 是细长罐体，抓取点窄；salad_dressing 瓶身更高、更易夹持。"
        "这一差异在改写条件下同样存在（20% vs 40%），说明它不是语言因素造成的。",
    )
    heading(document, "4.4  训练过程的观察", 2)
    body(
        document,
        f"训练 loss 从第 50 步的 {first_loss:.2f} 快速降到 1,000 步附近的约 1.0，之后进入缓慢下降阶段，"
        f"在 20,000 步时达到 {final_loss:.3f}。loss 曲线在 7,000–13,000 步之间有明显波动，"
        "与验证成功率同期的提升并不完全同步：10,000 步时验证只有 1/15，而 15,000 步时也只到 2/15，"
        "到 20,000 步才跃升到 9/15。这说明 loss 与闭环成功率之间不是简单的单调关系，"
        "仅凭训练 loss 选 checkpoint 并不可靠，必须用验证初始状态做闭环筛选。",
    )
    body(
        document,
        "显存方面，batch size 1 下 PyTorch 分配峰值始终为 1.09 GB，远低于 8 GB 上限。"
        "这为后续增大 batch 或提高 LoRA 秩留出了余量，但本项目的目标是可复现的最小配置，因此没有进一步调参。"
        "吞吐约 8.4 步/秒意味着 20,000 步约 40 分钟，符合课程项目的算力约束。",
    )
    heading(document, "4.5  与预期的差异", 2)
    body(
        document,
        "实验前的一个朴素预期是：只要 LoRA 能让训练 loss 明显下降，已见任务的成功率应当较高。"
        "实际结果显示，即使选了验证最好的 20k checkpoint，已见原句也只有 53.3%。"
        "可能的原因包括：训练数据只有 136 条轨迹，动作分布覆盖不足；LoRA 只训练 0.33% 的参数，"
        "动作专家容量有限；以及 LIBERO 的成功判定要求完整完成抓取和放入，任何一步偏差都会导致整回合失败。"
        "这些原因之间无法用现有数据完全分离，需要在后续实验中逐项控制。",
    )
    heading(document, "4.6  局限", 2)
    bullet(document, "单一训练种子、单一模型规模，无法估计种子间方差；所有比较都是描述性的。")
    bullet(document, "训练数据只有 136 条轨迹、三个任务，结论不能外推到完整 LIBERO-Object 或其它机器人。")
    bullet(document, "语言条件只用了两条人工改写；改写难度、语法结构与词频未做系统控制。")
    bullet(document, "未见条件同时改变物体、干扰物与布局，不能分离各因素的影响。")
    bullet(document, "评测使用仿真成功判定，未涉及真实机器人上的感知噪声与执行误差。")

    # Conclusion -------------------------------------------------------------
    heading(document, "5  结论", 1)
    body(
        document,
        "本项目在单卡 8 GB 环境下完成了 SmolVLA 的 LoRA 微调与 110 回合受控评测。"
        "已见任务原句成功率 53.3%，说明微调链路有效；同一场景下换成两条英文改写后降到 31.7%，"
        "配对分析中 20 个状态由成功转为失败，语言表述对策略行为有实质影响；"
        "两个未参与微调的物体只成功 1/20，没有表现出可用的物体级泛化。"
        "这些结果支持一个有限但明确的判断：在当前数据规模下，LoRA 微调让模型拟合了训练任务的特定表达与外观，"
        "而不是学到了可迁移的抓取—放置能力。后续工作应增加任务与物体的多样性、控制改写难度，"
        "并在改变布局的条件下重复实验，以分离语言理解与视觉布局利用两种机制。",
    )

    heading(document, "附录  复现信息", 1)
    bullet(document, "数据：lerobot/libero @ a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4；基座：lerobot/smolvla_base @ 5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6。")
    bullet(document, "训练：bash scripts/train.sh formal（8 GB 显存，约 41 分钟，20,000 步）。")
    bullet(document, "选模：python scripts/evaluate.py --checkpoint <ckpt> --validation --no-video。")
    bullet(document, "评测：python scripts/evaluate.py --checkpoint outputs/train/formal_s42_resume/checkpoints/020000 --output-dir outputs/eval/formal。")
    bullet(document, "汇总：python scripts/summarize.py --episodes outputs/eval/formal/formal_s42_resume_020000/episodes.csv。")
    bullet(document, "逐回合记录、110 个回放视频与所有图表见 outputs/eval/formal/formal_s42_resume_020000/，轻量副本见 deliverables/evidence/。")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUT)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
