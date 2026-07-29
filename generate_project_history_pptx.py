from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


OUT_PATH = Path("gnn_project_history_report.pptx")

BLUE = RGBColor(49, 60, 177)
BLUE_DARK = RGBColor(29, 39, 138)
INK = RGBColor(24, 28, 43)
MUTED = RGBColor(88, 96, 116)
WHITE = RGBColor(255, 255, 255)
PAPER = RGBColor(249, 251, 255)
PALE = RGBColor(232, 239, 255)
GRAY = RGBColor(242, 244, 248)
WARN = RGBColor(255, 241, 218)
WARN_DARK = RGBColor(171, 96, 14)
GREEN = RGBColor(225, 246, 235)
GREEN_DARK = RGBColor(14, 118, 69)
BORDER = RGBColor(191, 200, 224)


def set_run_font(run, size: int, bold: bool = False, color: RGBColor = INK) -> None:
    run.font.name = "Microsoft YaHei"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color


def add_bg(slide, title: str, subtitle: str | None = None) -> None:
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = PAPER

    bar = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, 0, 0, Inches(13.333), Inches(0.74))
    bar.fill.solid()
    bar.fill.fore_color.rgb = BLUE
    bar.line.color.rgb = BLUE

    box = slide.shapes.add_textbox(Inches(0.42), Inches(0.12), Inches(12.4), Inches(0.5))
    tf = box.text_frame
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = title
    set_run_font(run, 22, bold=True, color=WHITE)
    p.alignment = PP_ALIGN.LEFT

    if subtitle:
        sbox = slide.shapes.add_textbox(Inches(0.55), Inches(0.9), Inches(12.2), Inches(0.35))
        stf = sbox.text_frame
        sp = stf.paragraphs[0]
        srun = sp.add_run()
        srun.text = subtitle
        set_run_font(srun, 12, color=BLUE_DARK)


def add_text(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    size: int = 18,
    bold: bool = False,
    color: RGBColor = INK,
    align=PP_ALIGN.LEFT,
) -> None:
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    set_run_font(run, size, bold=bold, color=color)


def add_bullet_list(
    slide,
    bullets: list[str],
    x: float = 0.75,
    y: float = 1.35,
    w: float = 11.8,
    h: float = 5.55,
    size: int = 18,
    color: RGBColor = INK,
) -> None:
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    for idx, bullet in enumerate(bullets):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = bullet
        p.level = 0
        p.space_after = Pt(9)
        for run in p.runs:
            set_run_font(run, size, color=color)


def add_card(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    fill: RGBColor = PALE,
    title_color: RGBColor = BLUE_DARK,
    body_size: int = 15,
) -> None:
    shape = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = BORDER

    add_text(slide, x + 0.18, y + 0.16, w - 0.36, 0.32, title, size=14, bold=True, color=title_color)
    add_text(slide, x + 0.18, y + 0.58, w - 0.36, h - 0.76, body, size=body_size, color=INK)


def add_table(
    slide,
    columns: list[str],
    rows: list[list[str]],
    x: float,
    y: float,
    w: float,
    h: float,
    widths: list[float],
    font_size: int = 10,
) -> None:
    table_shape = slide.shapes.add_table(len(rows) + 1, len(columns), Inches(x), Inches(y), Inches(w), Inches(h))
    table = table_shape.table
    for idx, width in enumerate(widths):
        table.columns[idx].width = Inches(width)

    for col, heading in enumerate(columns):
        cell = table.cell(0, col)
        cell.text = heading
        cell.fill.solid()
        cell.fill.fore_color.rgb = BLUE
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            for run in p.runs:
                set_run_font(run, font_size, bold=True, color=WHITE)

    for r_idx, row in enumerate(rows, start=1):
        for c_idx, value in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = value
            cell.fill.solid()
            cell.fill.fore_color.rgb = PALE if r_idx % 2 else GRAY
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER
                for run in p.runs:
                    set_run_font(run, font_size, color=INK)


def add_slide(prs: Presentation, title: str, subtitle: str | None = None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide, title, subtitle)
    return slide


def title_slide(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor(247, 249, 255)

    banner = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.RECTANGLE,
        Inches(0.55),
        Inches(0.9),
        Inches(12.2),
        Inches(1.35),
    )
    banner.fill.solid()
    banner.fill.fore_color.rgb = BLUE
    banner.line.color.rgb = BLUE

    add_text(slide, 0.85, 1.18, 11.2, 0.7, "晶体带隙预测项目阶段性汇报", 28, True, WHITE)
    add_text(slide, 0.95, 2.55, 11.0, 0.45, "从模型增强到数据质量诊断：HZW 与 Materials Project 复现分析", 20, True, BLUE_DARK)
    add_text(slide, 0.95, 3.15, 11.1, 1.1, "任务：晶体 band gap 回归预测\n模型：CGCNN / ALIGNN / M3GNet / Virtual Node / Attentive Dual Slot\n日期：2026年5月", 18, False, INK)
    add_card(slide, 0.95, 5.15, 11.45, 1.05, "汇报主线", "先汇报模型路线为什么成立，再说明为什么 MP 复现削弱，最后给出数据质量诊断和下一步最小可行方案。", fill=GREEN, title_color=GREEN_DARK)


def main() -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    title_slide(prs)

    slide = add_slide(prs, "1. 汇报主线")
    add_card(slide, 0.75, 1.25, 3.8, 1.7, "研究问题", "局部晶体 GNN 是否缺少全局信息？虚拟节点和注意力机制能否改善 band gap 预测？", GREEN, GREEN_DARK, 14)
    add_card(slide, 4.85, 1.25, 3.8, 1.7, "实验路径", "先在 HZW 数据集上做 E1-E13，再迁移到 MP 和 MP-HZW matched 数据。", PALE, BLUE_DARK, 14)
    add_card(slide, 8.95, 1.25, 3.6, 1.7, "当前结论", "模型差异不能脱离标签源讨论；MP 与 HZW 标签冲突足以改变模型排序。", WARN, WARN_DARK, 14)
    add_bullet_list(
        slide,
        [
            "第一部分：任务背景、基线模型与增强假设。",
            "第二部分：HZW 上 E4/E8/E12/E13 的有效结果。",
            "第三部分：MP 复现后出现的规律失效。",
            "第四部分：MP-HZW 标签冲突诊断与后续方案。",
        ],
        y=3.55,
        size=18,
    )

    slide = add_slide(prs, "2. 任务背景：晶体 band gap 预测")
    add_bullet_list(
        slide,
        [
            "输入是晶体结构：原子种类、坐标、晶格和邻接关系；输出是 band gap 数值。",
            "CGCNN / ALIGNN / M3GNet 的共同核心是局部消息传递：从邻居原子、键长、键角等局部环境学习性质。",
            "我们的出发点：band gap 虽然强依赖局部成键和元素化学，但晶胞尺度、长程结构和全局汇聚也可能影响预测。",
            "因此项目主线是：在忠实复现基线后，逐步加入 virtual node、attention、dual slot 和物理约束。",
        ],
        size=18,
    )

    slide = add_slide(prs, "3. 三个基线模型")
    add_card(slide, 0.65, 1.28, 3.9, 4.9, "CGCNN", "把晶体表示成原子图，边特征主要来自邻居距离；优点是简单、可解释；不足是几何表达较弱。", PALE, BLUE_DARK)
    add_card(slide, 4.72, 1.28, 3.9, 4.9, "ALIGNN", "在原子图之外引入 line graph，对键角和二阶几何关系建模；是本项目最强、最稳定的 baseline。", GREEN, GREEN_DARK)
    add_card(slide, 8.8, 1.28, 3.9, 4.9, "M3GNet", "强调三体相互作用和材料图表示，适合通用势能和结构性质；在本任务中没有超过 ALIGNN。", PALE, BLUE_DARK)

    slide = add_slide(prs, "4. 数据与阶段路线")
    add_table(
        slide,
        ["阶段", "数据范围", "目的", "关键结论"],
        [
            ["HZW", "95,497 条\ntrain/val/test 固定", "先验证模型增强是否有效", "E4 稳定提升，E12/E13 进一步超过 E4"],
            ["MP full", "154,875 条\nMP bandgap", "检验公开数据泛化", "E4 重新最稳，E7/E8/E12 未超过 E4"],
            ["MP-HZW matched", "92,826 条匹配样本", "一比一对齐 HZW 来源", "发现 MP summary 与 HZW band gap 标签冲突"],
            ["Clean matched", "84,830 条类标签一致样本", "排除金属/非金属冲突后重训", "E4 仍最稳，E12/E1 待最终同步"],
        ],
        0.45,
        1.25,
        12.45,
        5.4,
        [1.7, 2.6, 3.5, 4.65],
        11,
    )

    slide = add_slide(prs, "5. 实验矩阵：E1-E13")
    add_table(
        slide,
        ["编号", "模型/方案", "核心作用", "当前判断"],
        [
            ["E1", "CGCNN", "基础晶体图卷积 baseline", "弱于 ALIGNN"],
            ["E2", "ALIGNN", "加入 line graph / 键角信息", "最强 baseline"],
            ["E3", "M3GNet", "三体几何和通用材料图", "本任务未超过 ALIGNN"],
            ["E4", "ALIGNN + 朴素超节点", "全局汇聚", "最稳定增强"],
            ["E5", "ALIGNN + Transformer", "全局注意力 only", "整体负优化/弱收益"],
            ["E6", "ALIGNN + 物理约束", "CBM/VBM 辅助监督", "HZW 有收益"],
            ["E7", "1 个 attentive 超节点", "单槽位注意力超节点", "不稳定，HZW 负优化"],
            ["E8", "2 个 attentive 超节点", "双槽位 + slot attention", "HZW 大结构有收益"],
            ["E10/E11", "E4/E8 路由融合", "阈值或验证集路由", "阈值泛化不稳"],
            ["E12", "size-supervised dual slot", "size 只进 loss 监督 slot2", "HZW 强，MP 弱"],
            ["E13", "E12 + physics", "双槽位 + 物理约束", "HZW 当前最好"],
        ],
        0.35,
        1.05,
        12.65,
        6.1,
        [0.9, 2.6, 4.25, 4.3],
        9,
    )

    slide = add_slide(prs, "6. HZW 阶段：模型增强确实有效")
    add_table(
        slide,
        ["实验", "模型", "MAE", "RMSE", "R²", "结论"],
        [
            ["E1", "CGCNN", "0.3022", "0.5303", "0.8847", "基础 baseline"],
            ["E2", "ALIGNN", "0.2314", "0.4713", "0.9089", "最强 baseline"],
            ["E4", "ALIGNN + VN", "0.2201", "0.4746", "0.9077", "朴素超节点最稳"],
            ["E6", "ALIGNN + physics", "0.2212", "0.4539", "0.9155", "RMSE / R² 更好"],
            ["E8", "2-slot attentive VN", "0.2285", "0.4945", "0.8997", "整体一般，大结构有价值"],
            ["E12", "size-supervised dual slot", "0.2179", "0.4712", "0.9090", "超过 E4"],
            ["E13", "E12 + physics", "0.2119", "0.4616", "0.9126", "HZW 当前最优"],
        ],
        0.55,
        1.25,
        12.2,
        4.9,
        [0.9, 3.0, 1.3, 1.3, 1.25, 4.0],
        10,
    )
    add_text(slide, 0.75, 6.35, 11.9, 0.42, "HZW 上的核心事实：E4 证明“全局汇聚”有效；E12/E13 证明“带监督的第二全局槽位 + 物理约束”还能继续提升。", 14, True, GREEN_DARK)

    slide = add_slide(prs, "7. HZW 分桶：为什么会想到双超节点")
    add_table(
        slide,
        ["原子数桶", "样本数", "E4", "E7", "E8", "观察"],
        [
            ["<=10", "2661", "0.1248", "0.2226", "0.1272", "E4 最稳"],
            ["11-20", "2232", "0.2022", "0.2912", "0.2060", "E4 最稳"],
            ["21-40", "2534", "0.2424", "0.3552", "0.2548", "E4 最稳"],
            ["41-80", "1642", "0.3193", "0.4419", "0.3422", "E4 最稳"],
            ["81-160", "447", "0.3796", "0.5190", "0.3736", "E8 开始反超"],
            [">160", "35", "0.2930", "0.3462", "0.2696", "E8 最好"],
        ],
        0.65,
        1.25,
        12.0,
        4.8,
        [1.4, 1.15, 1.25, 1.25, 1.25, 5.0],
        11,
    )
    add_text(slide, 0.8, 6.25, 11.6, 0.5, "当时的合理推断：小中结构用一个朴素超节点足够，大结构可能需要两个全局槽位承载不同信息。", 15, True, BLUE_DARK)

    slide = add_slide(prs, "8. E12 / E13 的设计逻辑")
    add_card(slide, 0.65, 1.25, 3.9, 4.85, "结构", "以 ALIGNN 为骨架，保留 E4 的主超节点，再加入第二个 attentive global slot。slot2 不是直接硬开，而是由模型学习使用强度。", GREEN, GREEN_DARK, 14)
    add_card(slide, 4.75, 1.25, 3.9, 4.85, "E12 损失", "size 不进入前向 gate；训练时用 num_atoms 构造软标签监督 slot2_gate：\nLoss = MSE(gap) + λ BCE(gate, sigmoid((N-80)/18))", PALE, BLUE_DARK, 13)
    add_card(slide, 8.85, 1.25, 3.85, 4.85, "E13 物理约束", "在 E12 基础上加入 CBM/VBM 辅助监督和 gap = CBM - VBM 一致性约束，并对无效 CBM/VBM 样本使用 mask。", WARN, WARN_DARK, 14)

    slide = add_slide(prs, "9. MP full 复现：规律开始失效")
    add_table(
        slide,
        ["实验", "模型", "MAE", "RMSE", "R²", "Best Epoch"],
        [
            ["E2", "ALIGNN", "0.2397", "-", "-", "-"],
            ["E4", "ALIGNN + VN", "0.2302", "0.5122", "0.8854", "50"],
            ["E7", "1 attentive VN", "0.2390", "0.5273", "0.8785", "48"],
            ["E8", "2 attentive VN", "0.2418", "0.5261", "0.8791", "43"],
            ["E12", "size-supervised", "0.2363", "0.5187", "0.8825", "48"],
        ],
        0.8,
        1.35,
        11.7,
        3.85,
        [0.9, 3.4, 1.35, 1.35, 1.35, 1.5],
        11,
    )
    add_bullet_list(
        slide,
        [
            "MP full 上 E4 重新成为最优，E7/E8/E12 都没有超过 E4。",
            "这说明 HZW 上的“size -> 开第二槽位”假设不能直接迁移到 MP full。",
            "但这还不能直接判定模型失败，因为目标标签来源已经发生变化。",
        ],
        y=5.55,
        h=1.2,
        size=14,
    )

    slide = add_slide(prs, "10. 不是因为 MP 更小")
    add_table(
        slide,
        ["数据集", "样本数", "median N", "mean N", ">80 占比", ">160 占比", "最大 N"],
        [
            ["HZW all", "95,497", "20", "28.90", "5.02%", "0.35%", "432"],
            ["MP full all", "154,875", "20", "31.39", "7.76%", "1.17%", "444"],
            ["HZW test", "9,551", "20", "29.01", "5.05%", "0.37%", "312"],
            ["MP test", "15,488", "20", "31.33", "7.63%", "1.16%", "392"],
        ],
        0.65,
        1.45,
        12.0,
        3.35,
        [1.8, 1.45, 1.35, 1.25, 1.45, 1.45, 1.25],
        12,
    )
    add_bullet_list(
        slide,
        [
            "MP 并不是更小，反而大结构占比更高。",
            "因此“MP 小结构太多，所以双槽位无效”这个解释不成立。",
            "更关键的问题是：num_atoms 在 MP 标签下不是稳定的“需要第二全局槽位”的代理变量。",
        ],
        y=5.25,
        h=1.55,
        size=16,
    )

    slide = add_slide(prs, "11. MP-HZW matched clean 当前结果")
    add_table(
        slide,
        ["实验", "模型", "MAE", "RMSE", "R²", "Metal F1"],
        [
            ["E2", "ALIGNN", "0.2534", "0.5019", "0.8973", "0.8600"],
            ["E3", "M3GNet", "0.2966", "0.5726", "0.8663", "0.8489"],
            ["E4", "ALIGNN + VN", "0.2446", "0.5032", "0.8968", "0.8734"],
            ["E5", "Transformer", "0.2575", "0.5179", "0.8906", "0.8670"],
            ["E7", "1 attentive VN", "0.2478", "0.5089", "0.8944", "0.8734"],
            ["E8", "2 attentive VN", "0.2476", "0.5111", "0.8935", "0.8757"],
            ["E10", "Val-routed E4/E8", "0.2455", "0.5044", "0.8963", "0.8730"],
        ],
        0.55,
        1.25,
        12.2,
        4.9,
        [0.9, 3.45, 1.3, 1.3, 1.25, 1.55],
        10,
    )
    add_text(slide, 0.75, 6.25, 11.8, 0.45, "截至本地同步结果：E4 仍是最稳模型；E1/E12 clean 补跑结果尚未进入本地最终汇总。", 14, True, WARN_DARK)

    slide = add_slide(prs, "12. 硬阈值路由为什么不稳")
    add_table(
        slide,
        ["原子数桶", "样本数", "E4", "E8", "E10", "现象"],
        [
            ["<=10", "2525", "0.1393", "0.1378", "0.1393", "E8 略好但 E10 没用上"],
            ["11-20", "2086", "0.2317", "0.2276", "0.2317", "E8 略好但 E10 没用上"],
            ["21-40", "2300", "0.2955", "0.3008", "0.2955", "E4 更好"],
            ["41-80", "1603", "0.3249", "0.3359", "0.3249", "E4 更好"],
            ["81-160", "429", "0.3364", "0.3643", "0.3620", "阈值路由最吃亏"],
            [">160", "34", "0.4598", "0.3834", "0.3834", "E8 明显更好"],
        ],
        0.55,
        1.2,
        12.25,
        4.9,
        [1.35, 1.1, 1.15, 1.15, 1.15, 5.0],
        10,
    )
    add_text(slide, 0.75, 6.25, 11.9, 0.45, "验证集阈值 82 在 test 上不鲁棒；最优 test 阈值约 144，只保留极大结构给 E8。", 14, True, BLUE_DARK)

    slide = add_slide(prs, "13. 数据质量诊断：金属/非金属冲突")
    add_table(
        slide,
        ["冲突方向", "数量", "比例", "强冲突例子", "解释"],
        [
            ["MP metal-like\nHZW nonmetal", "2,968", "3.197%", "HZW gap >=1 eV: 1,130\n>=2 eV: 447", "MP 认为近似金属，但 HZW 标签有明显带隙"],
            ["MP nonmetal-like\nHZW metal", "5,028", "5.595%", "MP gap >=1 eV: 561\n>=2 eV: 170", "MP 有正带隙，但 HZW 标签是 0"],
            ["删除两类冲突后", "84,830", "-", "train 67,865\nval 8,481\ntest 8,484", "得到 class-consistent matched 数据"],
        ],
        0.55,
        1.25,
        12.2,
        4.8,
        [2.35, 1.2, 1.15, 3.2, 4.0],
        10,
    )
    add_text(slide, 0.75, 6.25, 11.8, 0.45, "这不是简单的噪声点，而是足以影响 metal/nonmetal 边界与回归损失的系统性标签冲突。", 14, True, WARN_DARK)

    slide = add_slide(prs, "14. 数据质量诊断：同类样本仍有大数值差异")
    add_card(slide, 0.75, 1.25, 3.5, 4.8, "数量", "删除金属/非金属冲突后，仍有 1,317 个 same-class 样本满足 |MP gap - HZW gap| > 1 eV。", WARN, WARN_DARK, 17)
    add_card(slide, 4.55, 1.25, 3.5, 4.8, "分布", "这些样本全部属于非金属类；其中 21-40 原子桶最多，为 487 个。", PALE, BLUE_DARK, 17)
    add_card(slide, 8.35, 1.25, 3.5, 4.8, "含义", "即使金属/非金属类别一致，两个标签源在 gap 数值上也存在不可忽略偏差。", GREEN, GREEN_DARK, 17)

    slide = add_slide(prs, "15. 当前最可信解释")
    add_bullet_list(
        slide,
        [
            "HZW 上的 E12/E13 有效，说明 dual-slot + size supervision + physics 的模型假设不是凭空无效。",
            "MP full 和 MP-HZW matched 上 E4 更稳，说明该假设对标签源和数据分布敏感。",
            "MP-HZW matched 不是“同一批结构、同一套标签”的纯复现实验；训练脚本使用的是 MP summary band_gap，而不是 HZW 原始 gap。",
            "金属/非金属冲突和 >1 eV 数值差异会直接改变模型排序，因此当前负结果不能简单解释为模型结构失败。",
        ],
        size=18,
    )

    slide = add_slide(prs, "16. 下一步最小可行方案")
    add_card(slide, 0.65, 1.25, 3.9, 4.9, "数据先收敛", "固定一个标签源：要么用 HZW gap，要么用 MP summary gap。不要在结论里把二者混为同一个监督目标。", GREEN, GREEN_DARK, 14)
    add_card(slide, 4.72, 1.25, 3.9, 4.9, "实验先最小化", "在 clean matched 上先跑 E2 / E4 / E8 / E12 四个核心实验。先判断 dual-slot 是否仍有可重复收益。", PALE, BLUE_DARK, 14)
    add_card(slide, 8.8, 1.25, 3.9, 4.9, "任务再扩展", "band gap 的全局信息收益有限。若继续做全局建模，应考虑 dielectric / piezoelectric / phonon / elastic 等更全局敏感任务。", WARN, WARN_DARK, 14)

    slide = add_slide(prs, "17. 总结")
    add_bullet_list(
        slide,
        [
            "完成了 CGCNN、ALIGNN、M3GNet 三类基线及多种 ALIGNN 增强模型的实验闭环。",
            "HZW 数据上：E4 证明朴素全局超节点有效，E12/E13 进一步取得最优结果。",
            "MP 数据上：复杂注意力超节点没有稳定超过 E4，说明原假设不能直接泛化。",
            "数据诊断显示：MP 与 HZW 标签存在明显冲突，这是当前模型结论不一致的关键混杂因素。",
            "后续应先建立同源、干净、可复现的数据基准，再讨论模型结构是否真的优于 E4。",
        ],
        size=18,
    )

    slide = add_slide(prs, "参考文献")
    refs = [
        "[1] Tian Xie and Jeffrey C. Grossman. Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties. Physical Review Letters, 120(14):145301, 2018.",
        "[2] Kamal Choudhary and Brian DeCost. Atomistic Line Graph Neural Network for improved materials property predictions. npj Computational Materials, 7:185, 2021.",
        "[3] Chi Chen and Shyue Ping Ong. A universal graph deep learning interatomic potential for the periodic table. Nature Computational Science, 2:718-728, 2022.",
        "[4] Anubhav Jain et al. Commentary: The Materials Project: A materials genome approach to accelerating materials innovation. APL Materials, 1:011002, 2013.",
        "[5] Alexander Dunn et al. Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm. npj Computational Materials, 6:138, 2020.",
    ]
    add_bullet_list(slide, refs, x=0.55, y=1.18, w=12.25, h=5.95, size=12)

    prs.save(OUT_PATH)
    print(f"Saved {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
