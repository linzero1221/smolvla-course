#!/usr/bin/env python3
"""Generate the 12-slide, 15-minute course presentation (.pptx).

Layout follows docs/PROJECT_PLAN.md section 7: problem/background, method,
results/analysis, conclusion, division of work, contribution statement and
limitations. Numbers are read from the recorded evaluation outputs.

Usage:
    python scripts/make_slides.py
"""

from __future__ import annotations

import csv
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Cm, Pt

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = REPO_ROOT / "deliverables" / "evidence"
OUT = REPO_ROOT / "deliverables" / "slides" / "smolvla_course_slides.pptx"

NAVY = RGBColor(0x1F, 0x3B, 0x57)
ACCENT = RGBColor(0x4C, 0x72, 0xB0)
GREY = RGBColor(0x55, 0x55, 0x55)
LIGHT = RGBColor(0xF2, 0xF5, 0xF9)

FONT = "Microsoft YaHei"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def true(value: str) -> bool:
    return value.strip().lower() == "true"


def new_deck() -> Presentation:
    deck = Presentation()
    deck.slide_width = Cm(33.87)
    deck.slide_height = Cm(19.05)
    return deck


def textbox(slide, left, top, width, height, text, size=18, bold=False,
            color=NAVY, align=PP_ALIGN.LEFT, line_spacing=1.15):
    box = slide.shapes.add_textbox(Cm(left), Cm(top), Cm(width), Cm(height))
    frame = box.text_frame
    frame.word_wrap = True
    lines = text.split("\n")
    for index, line in enumerate(lines):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.alignment = align
        paragraph.line_spacing = line_spacing
        run = paragraph.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        run.font.name = FONT
    return box


def bullets(slide, left, top, width, height, items, size=17, color=NAVY):
    box = slide.shapes.add_textbox(Cm(left), Cm(top), Cm(width), Cm(height))
    frame = box.text_frame
    frame.word_wrap = True
    for index, item in enumerate(items):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.level = 0
        paragraph.space_after = Pt(8)
        run = paragraph.add_run()
        run.text = f"•  {item}"
        run.font.size = Pt(size)
        run.font.color.rgb = color
        run.font.name = FONT
    return box


def header(slide, title, subtitle=""):
    bar = slide.shapes.add_shape(1, Cm(0), Cm(0), Cm(33.87), Cm(2.1))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    textbox(slide, 1.2, 0.45, 31, 1.2, title, 26, True, RGBColor(0xFF, 0xFF, 0xFF))
    if subtitle:
        textbox(slide, 1.3, 2.5, 31, 1.0, subtitle, 13, False, GREY)


def footer(slide, number, total=13):
    textbox(slide, 28.5, 17.8, 5, 0.8, f"{number} / {total}", 11, False, GREY, PP_ALIGN.RIGHT)


def picture(slide, path: Path, left, top, width):
    if path.exists():
        slide.shapes.add_picture(str(path), Cm(left), Cm(top), width=Cm(width))


def table(slide, headers, rows, left, top, width, height, font_size=12):
    shape = slide.shapes.add_table(len(rows) + 1, len(headers), Cm(left), Cm(top), Cm(width), Cm(height))
    tbl = shape.table
    for index, text in enumerate(headers):
        cell = tbl.cell(0, index)
        cell.text = text
        for paragraph in cell.text_frame.paragraphs:
            for run in paragraph.runs:
                run.font.size = Pt(font_size)
                run.font.bold = True
                run.font.name = FONT
    for r, row in enumerate(rows, start=1):
        for c, value in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = str(value)
            for paragraph in cell.text_frame.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(font_size)
                    run.font.name = FONT
    return shape


def main() -> int:
    episodes = read_csv(REPO_ROOT / "outputs/eval/formal/formal_s42_resume_020000/episodes.csv")
    summary = {row["condition"]: row for row in read_csv(REPO_ROOT / "outputs/eval/formal/formal_s42_resume_020000/summary.csv")}
    pairs = read_csv(REPO_ROOT / "outputs/eval/formal/formal_s42_resume_020000/language_pairs.csv")
    degraded = sum(1 for row in pairs if true(row["original_success"]) and not true(row["paraphrase_success"]))
    improved = sum(1 for row in pairs if not true(row["original_success"]) and true(row["paraphrase_success"]))
    unchanged = len(pairs) - degraded - improved

    deck = new_deck()
    blank = deck.slide_layouts[6]

    # 1 Title -------------------------------------------------------------
    slide = deck.slides.add_slide(blank)
    band = slide.shapes.add_shape(1, Cm(0), Cm(6.3), Cm(33.87), Cm(6.4))
    band.fill.solid()
    band.fill.fore_color.rgb = NAVY
    band.line.fill.background()
    textbox(slide, 2, 7.3, 30, 2, "轻量 SmolVLA 微调与泛化评估", 40, True, RGBColor(0xFF, 0xFF, 0xFF), PP_ALIGN.CENTER)
    textbox(slide, 2, 9.9, 30, 1.6,
            "英文指令改写与未见物体在 LIBERO-Object 上的闭环实验", 20, False,
            RGBColor(0xD6, 0xE2, 0xF0), PP_ALIGN.CENTER)
    textbox(slide, 2, 15.2, 30, 1, "课程项目汇报 · 2026 年 11 月", 15, False, GREY, PP_ALIGN.CENTER)

    # 2 Problem ------------------------------------------------------------
    slide = deck.slides.add_slide(blank)
    header(slide, "问题与背景", "微调后的 VLA 能否听懂换一种说法，能否操作没见过的物体？")
    bullets(slide, 1.6, 3.8, 15.2, 12, [
        "SmolVLA：约 4.5 亿参数，单卡可微调",
        "LIBERO-Object：10 个抓放任务",
        "两问分开测：语言改写 / 未见物体",
        "只微调 3 个物体，留出 2 个",
    ], 16)
    picture(slide, EVIDENCE / "smoke_frame_start.png", 18.2, 4.2, 13.5)
    textbox(slide, 18.2, 12.2, 13.5, 2.5,
            "LIBERO-Object 原始场景：目标、干扰物与篮子均由仿真器固定。\n"
            "本项目的所有对照都在这个场景里完成。", 12, False, GREY)

    # 3 Method -------------------------------------------------------------
    slide = deck.slides.add_slide(blank)
    header(slide, "方法与方案", "LoRA 微调 + 固定初始状态的受控评测")
    bullets(slide, 1.6, 3.6, 15.4, 13, [
        "基座：lerobot/smolvla_base（固定 revision）",
        "LoRA：r=32，α=64，可训练参数 0.33%",
        "训练：batch 1，20,000 步，种子 42",
        "数据：136 条轨迹 / 19,445 帧",
        "审计：留出物体不在训练场景",
        "评测：固定初始状态 + 配对改写",
    ], 15)
    table(slide, ["集合", "物体", "用途"], [
        ["已见", "alphabet_soup / cream_cheese / salad_dressing", "训练与验证/测试"],
        ["未见", "bbq_sauce / chocolate_pudding", "仅正式测试"],
    ], 17.6, 4.2, 14.8, 4.4, 12)
    textbox(slide, 17.6, 9.2, 14.8, 6,
            "同一任务、同一初始状态、同一推理种子下，\n"
            "只替换传给策略的英文指令；\n"
            "环境成功谓词、目标物体与动作上限保持不变。", 14, False, GREY)

    # 4 Data and audit -----------------------------------------------------
    slide = deck.slides.add_slide(blank)
    header(slide, "数据划分与物体审计", "训练场景里确实没有留出物体")
    table(slide, ["任务", "轨迹", "帧数", "角色"], [
        ["alphabet_soup", "44", "6,867", "已见"],
        ["cream_cheese", "45", "6,378", "已见"],
        ["salad_dressing", "47", "6,200", "已见"],
        ["bbq_sauce", "—", "—", "未见"],
        ["chocolate_pudding", "—", "—", "未见"],
    ], 1.8, 4.0, 14.5, 8.4, 13)
    bullets(slide, 17.2, 4.2, 15, 10, [
        "任务映射由原始英文指令逐条匹配，而非按数据集顺序猜",
        "数据集 task_index：24 / 22 / 26 / 23 / 29",
        "BDDL 审计：三个训练场景均不含 bbq_sauce、chocolate_pudding",
        "训练子集 stats.json 只由 136 条轨迹重算（count=19,445）",
        "每个任务有 50 个初始状态，满足至少 15 个可用状态的要求",
    ], 15)

    # 5 Training -----------------------------------------------------------
    slide = deck.slides.add_slide(blank)
    header(slide, "训练过程", "20,000 步，约 41 分钟，8.4 步/秒，显存 1.09 GB")
    picture(slide, EVIDENCE / "formal_loss_20k.png", 1.6, 3.6, 17.5)
    bullets(slide, 20.2, 4.2, 12, 10, [
        "200 步试跑：28 秒，loss 2.25→0.97",
        "正式：loss 2.96→0.503（20k 步）",
        "checkpoint：5k / 10k / 15k / 20k",
        "两段训练：首次到 ~11.3k 暂停，从 10k 续训",
        "loss 与闭环成功率不同步，必须用验证状态选模",
    ], 14)

    # 6 Checkpoint selection ----------------------------------------------
    slide = deck.slides.add_slide(blank)
    header(slide, "checkpoint 选择", "验证初始状态 0–4，每档 15 回合，只用原句")
    table(slide, ["Checkpoint", "验证成功", "成功率"], [
        ["005000", "0/15", "0.0%"],
        ["010000", "1/15", "6.7%"],
        ["015000", "2/15", "13.3%"],
        ["020000", "9/15", "60.0%"],
    ], 3.0, 4.3, 13.5, 7.2, 15)
    bullets(slide, 18.4, 4.3, 14, 10, [
        "成功率随步数单调上升 → 微调链路有效",
        "020000 明显领先，无需平局规则",
        "测试状态 5–14 只在锁定 020000 后使用一次",
        "验证只看已见任务与原句，不涉及改写/未见",
    ], 15)

    # 7 Main results -------------------------------------------------------
    slide = deck.slides.add_slide(blank)
    header(slide, "实验结果", "110 个固定初始状态回合")
    table(slide, ["条件", "成功/回合", "成功率"], [
        ["已见原句", f"{summary['seen_original']['successes']}/{summary['seen_original']['episodes']}", "53.3%"],
        ["已见改写", f"{summary['seen_paraphrase']['successes']}/{summary['seen_paraphrase']['episodes']}", "31.7%"],
        ["未见原句", f"{summary['unseen_original']['successes']}/{summary['unseen_original']['episodes']}", "5.0%"],
    ], 1.8, 4.2, 13, 6.4, 16)
    picture(slide, EVIDENCE / "formal_success_by_condition.png", 16.2, 3.6, 15.8)
    footer(slide, 7, 13)

    # 8 Per-object ---------------------------------------------------------
    slide = deck.slides.add_slide(blank)
    header(slide, "逐物体结果", "已见任务内部也有明显差异")
    table(slide, ["条件", "alphabet_soup", "cream_cheese", "salad_dressing"],
          [["已见原句", "2/10 (20%)", "5/10 (50%)", "9/10 (90%)"],
           ["已见改写", "4/20 (20%)", "7/20 (35%)", "8/20 (40%)"]],
          2.0, 4.3, 29.5, 4.2, 15)
    table(slide, ["条件", "bbq_sauce", "chocolate_pudding"],
          [["未见原句", "1/10 (10%)", "0/10 (0%)"]],
          2.0, 9.6, 20, 3.0, 15)
    textbox(slide, 2.0, 13.4, 29, 4,
            "salad_dressing 原句 9/10，alphabet_soup 只有 2/10；三者轨迹数接近（44/45/47），\n"
            "差异更可能来自物体外观与抓取难度，而非数据量。", 14, False, GREY)
    footer(slide, 8, 13)

    # 9 Language pairing ---------------------------------------------------
    slide = deck.slides.add_slide(blank)
    header(slide, "语言配对分析", "同任务、同初始状态、同种子，60 对")
    textbox(slide, 2.0, 3.8, 29, 2,
            f"{degraded} 对 成功 → 失败　·　{improved} 对 失败 → 成功　·　{unchanged} 对 保持不变",
            22, True, NAVY, PP_ALIGN.CENTER)
    table(slide, ["改写", "改写成功", "配对原句成功", "百分点变化"], [
        ["改写一（Grab the …）", "12/30", "16/30", "−13.3"],
        ["改写二（… after lifting it）", "7/30", "16/30", "−30.0"],
    ], 4.2, 7.2, 25.5, 4.0, 14)
    bullets(slide, 4.0, 12.4, 26, 4.5, [
        "两条改写都低于原句，改写二下降更大 → 与训练模板的语法距离可能是关键",
        "下降主要来自成功转失败（20 对），不是随机波动",
    ], 14)
    footer(slide, 9, 13)

    # 10 Conclusion --------------------------------------------------------
    slide = deck.slides.add_slide(blank)
    header(slide, "结论", "逐一回答两个研究问题")
    textbox(slide, 1.8, 3.8, 14.5, 2, "语言改写", 24, True, ACCENT)
    bullets(slide, 1.8, 5.4, 14.5, 10, [
        "原句 53.3% → 改写 31.7%",
        "60 对中 20 对由成功变失败",
        "改写二比改写一下降更大",
        "结论：模型依赖训练时的固定表述",
    ], 14)
    textbox(slide, 17.8, 3.8, 14.5, 2, "未见物体", 24, True, ACCENT)
    bullets(slide, 17.8, 5.4, 14.5, 10, [
        "留出物体合计 1/20（5.0%）",
        "bbq_sauce 1/10，chocolate_pudding 0/10",
        "失败常伴随抓错物体或停在目标附近",
        "结论：当前规模下无可用物体级泛化",
    ], 14)
    footer(slide, 10, 13)

    # 11 Division of work --------------------------------------------------
    slide = deck.slides.add_slide(blank)
    header(slide, "组内分工说明", "按工作包列出实际产物")
    table(slide, ["工作包", "主要工作", "产物"], [
        ["环境与数据", "WSL2/CUDA 环境、单回合验收、任务映射与物体审计", "环境记录、smoke CSV/MP4、task_map、object_audit"],
        ["训练与复现", "train.sh、200 步试跑、20k 正式训练与续训", "训练日志、4 个 checkpoint、loss 曲线"],
        ["评测与分析", "evaluate.py、固定状态与配对改写、110 回合", "episodes.csv、summary、language_pairs、110 个回放"],
        ["报告与汇报", "报告、PPT、证据整理与复现入口", "report.docx、slides.pptx、deliverables/evidence"],
    ], 1.8, 4.2, 30, 8.5, 12)
    textbox(slide, 1.8, 13.6, 30, 2, "实际执行由课程小组与 AI 助手协作完成；具体姓名与贡献按课程要求补充。", 13, False, GREY)
    footer(slide, 11, 13)

    # 12 Contribution ------------------------------------------------------
    slide = deck.slides.add_slide(blank)
    header(slide, "主要贡献声明", "以实际产物为准，注明开源复用")
    bullets(slide, 1.6, 3.8, 30, 9, [
        "在单卡 8 GB 环境完成 SmolVLA LoRA 微调与 110 回合受控评测，全部结果可追溯到逐回合 CSV 与回放",
        "建立可审计的数据划分：BDDL 物体清单 + 任务文本匹配 + stats 重算，确认留出物体未进入训练",
        "复用 LeRobot 0.6.1、SmolVLA 基座权重与 LIBERO-Object 场景，未修改上游模型与仿真环境",
        "自研部分：prepare_data.py、train.sh、evaluate.py、summarize.py、parse_train_log.py、reproduce.sh",
        "所有曲线与表格均由实际运行日志生成，示例、试跑与正式结果分开标注",
    ], 14)
    textbox(slide, 1.6, 13.6, 30, 2,
            "开源复用：LeRobot（Apache-2.0）、SmolVLA 权重、LIBERO/LIBERO-Object 仿真环境与资产。",
            13, False, GREY)
    footer(slide, 12, 13)

    # 13 Limitations -------------------------------------------------------
    slide = deck.slides.add_slide(blank)
    header(slide, "局限与展望", "")
    bullets(slide, 1.8, 3.8, 30, 6.5, [
        "单一种子、单一模型规模、三个训练任务，无法估计种子间方差",
        "语言条件只有两条人工改写，未系统控制难度、语法距离与词频",
        "未见条件同时改变目标物体、干扰物身份与布局，不能分离各因素的影响",
        "评测基于仿真成功判定，未涉及真实机器人的感知噪声与执行误差",
    ], 16)
    textbox(slide, 1.8, 11.0, 30, 1.5, "后续工作", 22, True, ACCENT)
    bullets(slide, 1.8, 12.8, 30, 5, [
        "增加任务与物体多样性，扩大训练轨迹数量",
        "在改变场景布局的条件下重复实验，分离语言理解与视觉布局利用",
        "多种子重复训练，报告均值与置信区间",
    ], 14)
    footer(slide, 13, 13)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    deck.save(OUT)
    print(f"wrote {OUT} ({len(deck.slides)} slides)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
