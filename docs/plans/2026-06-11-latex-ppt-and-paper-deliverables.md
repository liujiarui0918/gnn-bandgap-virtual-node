# LaTeX PPT 与论文初稿交付计划

日期：2026-06-11

## 1. 目标

本计划用于把当前晶体带隙预测项目整理成两个可交付 PDF：

1. 一份 Beamer 风格的汇报 PPT PDF。
2. 一份 LaTeX 论文初稿 PDF。

本轮只先产出计划文档。后续执行时，必须基于本计划生成 `.tex` 源文件并完成 PDF 编译与检查。

## 2. 交付物命名

建议新建以下文件，避免覆盖已有历史汇报：

| 类型 | 源文件 | PDF | 备注 |
| --- | --- | --- | --- |
| 汇报 PPT | `gnn_bandgap_research_final_slides.tex` | `gnn_bandgap_research_final_slides.pdf` | 用 Beamer 编译，作为组会/课堂汇报主材料 |
| 论文初稿 | `gnn_bandgap_research_paper_draft.tex` | `gnn_bandgap_research_paper_draft.pdf` | 用 article/ctexart 编译，作为论文雏形 |
| 可选讲稿 | `gnn_bandgap_research_final_speech.md` | 无 | 若时间允许，可根据 PPT 补一份讲稿 |
| 可选图表脚本 | `scripts/build_report_tables.py` | 无 | 若表格太多，可用脚本从 CSV 生成 LaTeX 表 |

默认放在项目根目录，与已有 `gnn_project_history_report.tex` 和 `matbench_gnn_final_report.tex` 保持一致。

## 3. 已有材料与模板来源

### 3.1 Beamer 模板

优先复用：

- `gnn_project_history_report.tex`
- `matbench_gnn_final_report.tex`
- `matbench_gnn_class_report.tex`
- `matbench_gnn_weekly_report.tex`

采用原因：

- 已配置 `ctex`，能稳定支持中文。
- 已使用 `Berlin` + `whale` Beamer 主题。
- 已处理参考文献黑色字体和 `[1]` 文本编号样式：

```tex
\setbeamertemplate{bibliography item}[text]
\setbeamercolor{bibliography item}{fg=black}
\setbeamercolor{bibliography entry author}{fg=black}
\setbeamercolor{bibliography entry title}{fg=black}
\setbeamercolor{bibliography entry location}{fg=black}
\setbeamercolor{bibliography entry note}{fg=black}
```

这能避免之前参考文献页出现奇怪图标或颜色的问题。

### 3.2 论文模板

项目内暂无专门论文模板，因此建议新建 `ctexart` 风格论文：

```tex
\documentclass[UTF8,a4paper,11pt]{ctexart}
```

核心包：

```tex
\usepackage{geometry}
\usepackage{amsmath, amssymb, bm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{longtable}
\usepackage{multirow}
\usepackage{hyperref}
\usepackage[numbers,sort&compress]{natbib}
```

为提高编译稳健性，第一版论文可以先用 `thebibliography`，不强依赖 BibTeX。若需要投稿级格式，再拆出 `references.bib`。

### 3.3 历史文字材料

可直接吸收内容：

- `gnn_project_history_speech.md`
- `matbench_gnn_progress_talk_track.md`
- `matbench_gnn_weekly_report_speech.md`
- `CODE_IMPLEMENTATION_SUMMARY.md`
- `CURRENT_SESSION.md`
- `docs/2026-04-19-global-attention-vs-virtual-node.md`
- `docs/2026-04-25-mp-vs-hongzhiwei-analysis.md`
- `docs/plans/2026-05-24-mp-hzw-structure-first-audit.md`
- `docs/plans/2026-05-30-relaxed-mp-hzw-t-series.md`
- `docs/plans/2026-06-01-t6-single-factor-ablations.md`
- `docs/plans/2026-06-03-topk-warmup-and-global-decay.md`

### 3.4 代码与配置材料

后续写方法部分时应引用这些实现位置：

- `faithful_gnn/alignn_variants.py`
- `faithful_gnn/alignn_runner.py`
- `run_t_relaxed_matched_experiments.py`
- `export_mp_hzw_final_clean_dataset.py`
- `export_mp_hzw_structure_first_tables.py`
- `audit_mp_hzw_matched_garbage.py`
- `configs/mp/MP_T6_alignn_topk_huber_virtualnode.yaml`
- `configs/mp/MP_T7_alignn_huber_virtualnode.yaml`
- `configs/mp/MP_T8_alignn_topk_mse_virtualnode.yaml`
- `configs/mp/MP_T9_alignn_warmup_topk_huber_virtualnode.yaml`
- `configs/mp/MP_T10_alignn_decay_huber_virtualnode.yaml`
- `configs/E14_alignn_physbranch_virtualnode.yaml`
- `configs/E15_alignn_gainsupervised_virtualnode.yaml`
- `configs/E16_alignn_orthogonal_dualslot.yaml`
- `configs/E17_alignn_complexitysupervised_virtualnode.yaml`
- `configs/E18_alignn_regimemoe_virtualnode.yaml`
- `configs/E19_alignn_metalaware_virtualnode.yaml`
- `configs/mp/MP_E20a_alignn_virtualnode_small_le40.yaml`
- `configs/mp/MP_E20b_alignn_virtualnode_large_gt40.yaml`
- `configs/mp/MP_E21b_alignn_multi_virtualnode_large_gt40.yaml`
- `run_bottom_up_ablation_queue.py`
- `run_e19_queue.py`
- `run_mp_size_split_queue.py`

## 4. 必须锁定的数据与结果口径

### 4.1 数据任务

主任务：

- 晶体 `band_gap` 回归预测。
- 单位：eV。
- 主指标：MAE。
- 辅助指标：RMSE、`R^2`、Metal F1、Best Epoch、训练耗时、峰值显存。

输入：

- 晶体结构。
- 原子种类、坐标、晶格、近邻关系。
- ALIGNN 系列额外使用 line graph 表征键角/二阶几何。

### 4.2 数据阶段

| 阶段 | 样本/记录口径 | 用途 | 需要在报告中讲清楚的点 |
| --- | ---: | --- | --- |
| HZW 原始 bandgap | 清洗后约 `95,497` 条 | 最早 E1-E13 实验 | E4、E12、E13 在 HZW 上有效 |
| MP full | 约 `154,875` 条 | 公开 MP bandgap 泛化 | E4 重新最稳，复杂双槽位没有稳定超过 E4 |
| MP-HZW structure-first audit | 总 `95,462`，结构匹配 `92,787`，缺 MP `2,636`，结构不匹配 `39` | 证明 MP/HZW 数据源不是完全等价 | 结构匹配与标签一致性必须分开讨论 |
| strict clean MP-label | `81,751` accepted，测试集 `8,176` | 曾用于 S 系列较严格清洗 | 删除 gap 差值和体积差过大的版本，后续不作为最终主口径 |
| relaxed clean equal-atoms MP label | `84,550` accepted，train/val/test = `67,640/8,455/8,455` | 当前最终 T 系列 MP 标签口径 | 只删零/非零金属冲突和原子数非 equal，不删 gap 差值大和体积差大 |
| relaxed clean equal-atoms HZW label | `84,550` accepted，train/val/test = `67,640/8,455/8,455` | 当前最终 T 系列 HZW 标签口径 | 同一批结构，换 HZW bandgap 标签 |

### 4.3 数据清洗事实

structure-first audit 关键数字：

| 项目 | 数量 |
| --- | ---: |
| Total rows | `95,462` |
| Structure matched rows | `92,787` |
| Structure unmatched rows | `39` |
| Compatible but unverified rows | `0` |
| Missing MP records | `2,636` |
| `structure_matched_exact` | `92,507` |
| `structure_matched_supercell` | `280` |

structure-matched 内部标签冲突：

| 类型 | 数量 |
| --- | ---: |
| `no_label_conflict` | `85,478` |
| `mp_metal_hzw_nonmetal` | `3,142` |
| `mp_nonmetal_hzw_metal` | `3,046` |
| `same_class_gap_diff_gt_1ev` | `1,121` |

relaxed clean equal-atoms 过滤规则：

- 从 `92,787` 条结构匹配样本出发。
- 删除 `band_gap <= 1e-6` 与 `band_gap > 1e-6` 的零/非零冲突：`7,984` 条。
- 删除 `atom_count_relation != equal`：`252` 条。
- 不再删除 `abs(mp_gap - hzw_gap) > 0.5 eV`。
- 不再删除 `volume_per_atom_rel_diff > 8%`。
- 最终 accepted records：`84,550` 条。

## 5. 实验体系梳理

### 5.1 完整 E 系列谱系

| 编号 | 模型 | 核心意义 | 论文/PPT 中定位 |
| --- | --- | --- | --- |
| E1 | CGCNN | 晶体图卷积 baseline | 基线 |
| E2 | ALIGNN | 原子图 + line graph，显式键角建模 | 最强传统 baseline |
| E3 | M3GNet | 三体几何材料 GNN | 对照 baseline |
| E4 | ALIGNN + single plain virtual node | 加入朴素全局超节点 | 第一条稳定有效增强 |
| E5 | ALIGNN + Transformer | 全局注意力 only | 负例/不稳定增强 |
| E6 | ALIGNN + physics constraint | CBM/VBM 辅助监督和 gap 一致性 | 物理约束尝试，仅在有 CBM/VBM 时完整成立 |
| E7 | ALIGNN + 1 attentive virtual node | 单注意力超节点 | HZW 上明显负优化 |
| E8 | ALIGNN + 2 attentive virtual nodes | 双注意力超节点 | HZW 大结构分桶上有收益 |
| E9/E10 | size-aware / active size-aware gate | 显式尺度 gate 控制第二槽位 | 历史探索，不作为最终主张 |
| E11 | hard/val routed ensemble | 小结构 E4、大结构 E8 的路由组合 | 融合诊断，不是端到端模型 |
| E12 | size-supervised dual slot | size 不进前向，只进 loss 监督第二槽位 | HZW 有效，MP 不稳 |
| E13 | E12 + physics | 双槽位 + 有效 mask 物理约束 | HZW 最优历史结果 |
| E14 | ALIGNN + phys-branch virtual node | 在 E4 主干上加入物理分支残差，尝试把物理先验融入全局节点 | 底层物理增强探索，适合放附录/失败分析 |
| E15 | gain-supervised dual-slot | 用 E4/E8 teacher gain 信号监督第二槽位何时带来收益 | “收益监督”探索，回应 `num_atoms` 代理变量不稳的问题 |
| E16 | orthogonal dual-slot | 强制两个全局槽位表征正交，避免两个 slot 学同一件事 | “强制分工”探索，适合方法消融/附录 |
| E17 | complexity-supervised virtual node | 用结构复杂度代理量监督第二槽位，而非单纯原子数 | “复杂度代理变量”探索，适合讨论 |
| E18 | regime MoE virtual node | 按金属/小带隙/大带隙等任务区间做 mixture-of-experts | “任务分区”探索，适合讨论失败原因 |
| E19 | metal-aware virtual node | 同时做金属分类和非金属 log-gap 回归，处理 0-gap 尖点 | 金属/非金属分治尝试，适合附录 |
| E20a/E20b | size-split E4 experts | MP 上小体系和大体系分别训练 E4 专家 | 大小体系分桶训练诊断 |
| E21b | large-system dual VN expert | MP 大体系上单独训练双全局槽位模型 | 检查大体系是否真的受益于双节点 |

注意：

- E1-E13 是项目早期主线，直接解释从 baseline 到 virtual node、physics、dual-slot 的设计过程。
- E14-E19 是在 MP/HZW 结论冲突后做的底层机制探索，应该完整记录，但不应挤占 PPT 主线。
- E20/E21 是大小体系分桶诊断，不是最终统一模型；适合放在论文附录和 PPT backup。
- 最终论文正文建议重点讲 E1-E13 的研究演化、S/T 系列的最终验证；E14-E21 作为“探索性消融与失败分析”集中整理。
- 命名冲突必须说明：`configs/E13_alignn_sizesupervised_physconstraint.yaml` 是 HZW 早期 `E13 = E12 + physics`；但 `configs/mp/MP_E13_alignn_plain_dual_virtualnode.yaml` 是后期 MP/S/T 系列复用的“双朴素超节点模板”，在报告里应称为 `S13` 或 `T5`，不要再称为早期 E13。

### 5.2 后期 S/T 系列

S 系列用于 strict clean 和朴素双节点诊断；T 系列用于 relaxed clean equal-atoms 最终口径。

最终 T 系列应作为论文主表。以 `run_t_relaxed_matched_experiments.py` 的当前映射为准：

| 编号 | 模型 | 变更点 |
| --- | --- | --- |
| T1 | CGCNN baseline | 基础晶体图卷积 |
| T2 | ALIGNN baseline | 最强基线 |
| T3 | M3GNet baseline | 材料图 baseline |
| T4 | ALIGNN + single plain virtual node | 稳定全局超节点 |
| T5 | ALIGNN + plain dual virtual node | 双朴素超节点；MP label 有完整 metrics 但未进入当前主 `comparison_summary.csv`，HZW label 中断无 test metrics |
| T6 | ALIGNN + top-k virtual node + Huber | 稀疏 node-to-global + 鲁棒损失 |
| T7 | ALIGNN + single plain virtual node + Huber | Huber-only 消融 |
| T8 | ALIGNN + top-k virtual node + MSE | top-k-only 消融 |
| T9 | ALIGNN + warmup top-k + Huber | 先全局汇聚再逐渐打开 top-k |
| T10 | ALIGNN + global broadcast decay + Huber | 训练后期降低 global-to-node 广播 |

注意：

- T 系列编号曾经在讨论中重用过。早期计划里的 `T6=val-routed T4/T5`、`T7=adaptive gated dual plain VN` 已被后续方案替换；最终写作必须以当前 runner 中的 T6/T7/T8/T9/T10 定义为准。
- T5 代表 plain dual virtual node。MP label 有完整 `metrics.json`，可以在附录或隐藏结果表中定量引用；HZW label 只有 5 个 epoch 的中断日志，不能写 test 结果。
- T6 相比 T4 同时改了 top-k 和 Huber。
- T7 是 Huber-only，判断鲁棒损失是否是主收益来源。
- T8 是 top-k-only，判断 top-k 是否独立有效。
- T9/T10 是在 Huber 基础上对信息流做训练过程调度。
- 当前结果显示 top-k 不是稳定主收益来源，不能把 top-k 讲成最终成功机制。

## 6. 必须纳入的最终结果

### 6.1 relaxed clean equal-atoms，MP label

结果源：

`/data/home/tg100025/gnn_work/results/mp_hzw_relaxed_clean_equal_atoms_mp_t/comparison_summary.csv`

| 实验 | 模型 | MAE | RMSE | R2 | Metal F1 | Best Epoch |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| T1 | CGCNN | 0.366338 | 0.612696 | 0.858769 | 0.788246 | 49 |
| T2 | ALIGNN | 0.240043 | 0.490694 | 0.909414 | 0.882637 | 50 |
| T3 | M3GNet | 0.285903 | 0.584315 | 0.871550 | 0.889726 | 41 |
| T4 | single VN | 0.232520 | 0.505610 | 0.903823 | 0.909240 | 49 |
| T6 | top-k + Huber | 0.229585 | 0.512560 | 0.901161 | 0.913312 | 46 |
| T7 | Huber-only | 0.224931 | 0.503546 | 0.904607 | 0.911331 | 49 |
| T8 | top-k-only | 0.230402 | 0.496074 | 0.907417 | 0.905809 | 50 |
| T9 | warmup top-k + Huber | 0.226028 | 0.495817 | 0.907513 | 0.914071 | 50 |
| T10 | global decay + Huber | **0.223686** | 0.497333 | 0.906946 | 0.915455 | 47 |

MP label 主结论：

- T4 相比 T2 仍有提升：`0.240043 -> 0.232520`。
- T7 Huber-only 明显优于 T4：`0.232520 -> 0.224931`。
- T10 在 MP label 上 MAE 最优：`0.223686`。
- T6/T8/T9 表明 top-k 本身并不是稳定主收益，warmup 也没有超过 T7/T10。
- T5 未出现在当前最终汇总中，但 MP label 的完整 `metrics.json` 显示 MAE 为 `0.239348`，弱于 T4/T7/T10；主表可不放，附录应注明。

### 6.2 relaxed clean equal-atoms，HZW label

结果源：

`/data/home/tg100025/gnn_work/results/mp_hzw_relaxed_clean_equal_atoms_hzw_t/comparison_summary.csv`

| 实验 | 模型 | MAE | RMSE | R2 | Metal F1 | Best Epoch |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| T1 | CGCNN | 0.338152 | 0.565454 | 0.878497 | 0.788237 | 49 |
| T2 | ALIGNN | 0.212124 | 0.436147 | 0.927713 | 0.890299 | 50 |
| T3 | M3GNet | 0.256572 | 0.535916 | 0.890859 | 0.892629 | 43 |
| T4 | single VN | 0.201762 | 0.448621 | 0.923519 | 0.909613 | 49 |
| T6 | top-k + Huber | 0.198584 | 0.447143 | 0.924022 | 0.913786 | 46 |
| T7 | Huber-only | **0.194002** | 0.445837 | 0.924465 | 0.915401 | 49 |
| T8 | top-k-only | 0.202603 | 0.448503 | 0.923559 | 0.909881 | 50 |
| T9 | warmup top-k + Huber | 0.197707 | 0.448776 | 0.923466 | 0.916846 | 46 |
| T10 | global decay + Huber | 0.197034 | 0.453592 | 0.921815 | 0.913209 | 45 |

HZW label 主结论：

- T4 相比 T2 提升明显：`0.212124 -> 0.201762`。
- T7 Huber-only 是 HZW label 最优：`0.194002`。
- T10 在 MP label 最优，但在 HZW label 不如 T7，说明 broadcast decay 有数据源依赖。
- 跨两个标签源最稳的主张是：`ALIGNN + single plain virtual node + Huber loss`。
- HZW label 的 T5 没有完整 test metrics，最终主结论不要讨论 HZW-T5 定量表现。

### 6.3 strict clean S 系列朴素双节点诊断

结果源：

`/data/home/tg100025/gnn_work/results/mp_hzw_final_clean_equal_atoms_mp_s/comparison_summary.csv`

关键结果：

| 实验 | 模型 | MAE | 结论 |
| --- | --- | ---: | --- |
| S4 | single plain VN | 0.200094 | strict clean MP label 最强简单模型 |
| S13 | dual plain VN | 0.201181 | 整体略差，但大结构更好 |

分桶诊断：

- `81-160`：S13 优于 S4，`0.307066 < 0.321578`。
- `>160`：S13 优于 S4，`0.266599 < 0.278618`。
- `>80` 合并：S13 优于 S4，`0.304307 < 0.318649`。
- 但 S13 在 `21-40` 和 `41-80` 损失抵消了大结构收益，因此整体没赢。

这部分适合放在 PPT 备份页或论文讨论部分，用来解释为什么“额外全局容量有局部收益，但需要更可靠路由/监督”。

### 6.4 被覆盖/未入主表的结果资产

这次审计发现：部分结果没有进入当前 `comparison_summary.csv`，但远端 `metrics.json` 仍存在；另一些只有目录或 checkpoint，不具备可引用的 test metrics。写 PPT/论文时必须区分。

#### 6.4.1 T5 双朴素超节点

| 数据口径 | 路径状态 | MAE | RMSE | R2 | Metal F1 | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| relaxed MP label | `mp_hzw_relaxed_clean_equal_atoms_mp_t/T5_ALIGNN_plain_dual_virtualnode/metrics.json` 完整存在 | 0.239348 | 0.504425 | 0.904273 | 0.906631 | 完整结果，但没有进入当前主 `comparison_summary.csv` |
| relaxed HZW label | `mp_hzw_relaxed_clean_equal_atoms_hzw_t/T5_ALIGNN_plain_dual_virtualnode/` 只有 `best_model.pt` 和 `train.log` | -- | -- | -- | -- | 只跑到 epoch 5，val MAE 0.4170，无 test metrics，不能引用为结果 |

解释：

- MP-T5 说明双朴素超节点在 relaxed MP label 上没有超过 T4/T7/T10。
- HZW-T5 不完整，不能补进 HZW 最终表。
- 如果论文需要讨论双朴素超节点，更可靠的完整证据是 strict clean 的 `S13` 和 MP-T5。

#### 6.4.2 deprecated old T6/T7

早期 `T6/T7` 编号后来被替换。旧结果保存在：

`/data/home/tg100025/gnn_work/results/mp_hzw_relaxed_clean_equal_atoms_mp_t/deprecated_old_t6_t7/`

| 旧编号 | 旧含义 | MAE | RMSE | R2 | Metal F1 | 备注 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| old T6 | val-routed T4/T5，验证集阈值 80 | 0.232609 | 0.505404 | 0.903901 | 0.908893 | ensemble/router，不是单模型 |
| old T7 | adaptive size-gated plain dual VN | 0.234135 | 0.500410 | 0.905791 | 0.907273 | `slot2_enable_mean=0.122410` |

写作规则：

- 不能和最终 T6/T7 混用。
- 若引用，必须写成 `deprecated old T6/T7` 或“早期 T6/T7 方案”。
- 主线结论仍以当前 runner 中的 T6/T7 为准。

#### 6.4.3 deprecated limited-top-k T9/T10

早期 T9/T10 使用了带原子数区间限制的 top-k。后来按师兄意见改成所有原子规模都启用 top-k/warmup/decay，因此旧版本只适合放附录。

| 数据口径 | 旧实验 | MAE | RMSE | R2 | Metal F1 | 备注 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| relaxed MP label | old T9 limited top-k warmup | 0.225298 | 0.494958 | 0.907833 | 0.913049 | 旧 top-k 限制版本 |
| relaxed MP label | old T10 limited decay | 0.224067 | 0.495228 | 0.907733 | 0.908279 | 旧 top-k 限制版本 |
| relaxed HZW label | old T9 limited top-k warmup | 0.195461 | 0.447637 | 0.923854 | 0.917771 | 旧 top-k 限制版本 |
| relaxed HZW label | old T10 limited decay | 0.194790 | 0.437499 | 0.927264 | 0.916078 | 旧 top-k 限制版本 |

写作规则：

- 主文使用最终 rerun 后的 all-size T9/T10。
- 旧版本可用于附录说明：去掉尺寸限制后，MP-T10 略优，HZW-T10 反而不如 Huber-only T7，说明信息流调度存在数据源依赖。

#### 6.4.4 E14-E19 / E20-E21 结果状态

当前本地和远端 `/data/home/tg100025/gnn_work/results` 审计结果：

| 实验段 | 是否找到完整 metrics | 当前可写内容 |
| --- | --- | --- |
| E14-E18 bottom-up ablation | 未找到完整 `metrics.json`、`comparison_summary.csv`、训练日志或结果目录 | 只能写设计动机、模型结构和探索定位，不能写定量结果 |
| E19 metal-aware | 未找到当前完整结果目录；但 `results/mp_size_split/analysis/mp_large_error_analysis.md` 中保留了 `MP_E19_full` 的大体系子集指标 | 可写“曾用于大体系误差诊断”，不能写完整 test-set 主结果 |
| E20/E21 size split | 找到 `results/mp_size_split/analysis/mp_large_error_analysis.md`，但未找到标准完整 `metrics.json` 汇总 | 可写大体系误差分析，不作为正式统一模型结果 |

E20 相关大体系分析可引用：

| 模型 | 大体系样本数 | MAE | RMSE | 结论 |
| --- | ---: | ---: | ---: | --- |
| MP_E4_full | 3527 | 0.331673 | 0.628684 | full E4 在大体系上仍优于单独训练大体系专家 |
| MP_E19_full | 3527 | 0.335089 | 0.661851 | metal-aware 没有超过 E4 |
| MP_E20b_large_only | 3527 | 0.368394 | 0.683709 | 大体系单独训练反而更差，说明样本少/方差大/过拟合 |

### 6.5 HZW 早期 E 系列核心结果

早期 HZW 结论用于说明研究动机的形成：

| 实验 | 模型 | MAE | 结论 |
| --- | --- | ---: | --- |
| E1 | CGCNN | 0.3022 | 基础 baseline |
| E2 | ALIGNN | 0.2314 | 最强 baseline |
| E4 | ALIGNN + VN | 0.2201 | 朴素全局超节点有效 |
| E6 | physics | 0.2212 | RMSE/R2 更稳 |
| E8 | 2 attentive VN | 0.2285 | 整体不如 E4，但大结构有信号 |
| E12 | size-supervised dual slot | 0.2179 | HZW 上超过 E4 |
| E13 | E12 + physics | 0.2119 | HZW 历史最优 |

使用注意：

- 这不是最终 T 系列主口径。
- 应作为“为什么我们一开始相信双槽位路线”的背景证据。
- 论文中需要明确其数据标签来自 HZW，不能与 MP label 结果直接混为同一实验。

## 7. PPT 结构计划

目标：完整汇报版，约 26-32 页主幻灯片，另加 6-10 页 backup。若需要 6-8 分钟短讲，可后续从中裁剪。

### 7.1 PPT 风格

- 使用 16:9 Beamer：`\documentclass[aspectratio=169]{beamer}`。
- 复用 `gnn_project_history_report.tex` 的主题、页脚、字体和参考文献设置。
- 大表格使用 `\resizebox{\textwidth}{!}{...}` 或拆成两页。
- 公式必须用 `\resizebox` 或分行 `aligned`，避免只显示一半。
- 参考文献页必须使用 `[1]` 文本编号，不使用图标。

### 7.2 PPT 主线

PPT 叙事要从“模型越复杂越好”的早期假设，收敛到“全局虚拟节点 + 鲁棒损失是当前最稳结论”：

1. 晶体 bandgap 预测为什么需要图神经网络。
2. CGCNN/ALIGNN/M3GNet 三个基线怎么建模晶体。
3. 局部消息传递可能存在全局信息瓶颈。
4. E4 证明朴素 virtual node 是稳定增强。
5. E7/E8/E12/E13 说明复杂全局槽位在 HZW 上曾有积极信号。
6. MP 和 matched 数据说明这条复杂路线具有数据源依赖。
7. 数据清洗证明 MP/HZW 标签并非完全等价。
8. 最终 T 系列证明 Huber loss 是最稳定的跨标签源收益。
9. 最终建议：以 ALIGNN + single VN + Huber 作为主模型；T10 可作为 MP label 特化改进。

### 7.3 PPT 逐页计划

| 页码 | 标题 | 内容 |
| ---: | --- | --- |
| 1 | 标题页 | 项目题目、汇报人、日期、任务 |
| 2 | 目录 | 背景、方法、数据清洗、结果、讨论、结论 |
| 3 | 研究问题 | 晶体结构到 bandgap；局部消息传递是否缺全局信息 |
| 4 | 晶体 GNN 输入表示 | 原子节点、边、晶格、周期边界、line graph |
| 5 | 三个基线模型 | CGCNN、ALIGNN、M3GNet 的简明对比 |
| 6 | 项目阶段路线 | HZW、MP full、MP-HZW matched、relaxed clean T-series |
| 7 | E1-E13 主线实验矩阵 | 早期主线探索全景，不展开所有细枝末节 |
| 8 | E4：朴素超节点 | node-to-global 汇聚和 global-to-node 广播示意 |
| 9 | E6/E13：物理约束 | CBM/VBM 辅助监督和有效 mask 的通俗解释 |
| 10 | HZW 早期结果 | E1/E2/E4/E6/E8/E12/E13 总表 |
| 11 | HZW 分桶发现 | E4 小中结构稳，E8 大结构有收益 |
| 12 | E12 设计动机 | 为什么尝试 size-supervised dual slot |
| 13 | E14-E19 底层机制探索 | 物理分支、收益监督、正交、复杂度、MoE、金属感知 |
| 14 | E20/E21 大小体系诊断 | 分桶训练和大体系双节点专家 |
| 15 | MP full 复现失败 | E4 重新最稳，E7/E8/E12 未超过 |
| 16 | 不是 MP 更小 | HZW vs MP 原子数分布对比 |
| 17 | 数据质量转折 | structure-first audit 流程图 |
| 18 | 结构匹配结果 | 92,787 结构匹配、39 不匹配、2,636 缺 MP |
| 19 | 标签冲突结果 | 金属/非金属冲突和 same-class gap 差异 |
| 20 | relaxed clean 口径 | 当前最终 84,550 样本怎么得到 |
| 21 | 最终 T 系列设计 | T1/T2/T3/T4/T6/T7/T8/T9/T10 表 |
| 22 | MP label 最终结果 | T10 最优，T7 稳定优于 T4 |
| 23 | HZW label 最终结果 | T7 最优，Huber 是主收益 |
| 24 | 消融结论 | Huber-only、top-k-only、warmup、decay 的对照解释 |
| 25 | 模型结论 | single VN + Huber 是跨标签源最稳方案 |
| 26 | 物理与建模讨论 | bandgap 的局部化学主导、全局信息收益边界 |
| 27 | 局限性 | 标签源冲突、bandgap 不是最强全局信息任务、复杂 gate 不稳 |
| 28 | 结论 | 三条最终结论 |
| 29 | 参考文献 | CGCNN、ALIGNN、M3GNet、MP、Matbench 等 |

Backup 页：

| 标题 | 内容 |
| --- | --- |
| S4 vs S13 分桶 | 朴素双节点大结构有收益但整体不赢 |
| E14-E21 全量实验索引 | 所有中间探索的模型定义、动机和最终归类 |
| E14-E19 机制失败分析 | 为什么收益监督、正交、复杂度、MoE、metal-aware 没成为最终主线 |
| T 系列训练耗时 | train_seconds、peak_gpu_gb |
| 公式细节 | Huber、top-k、warmup、decay 的数学定义 |
| 数据表来源 | CSV/JSON 路径，保证可追溯 |

## 8. 论文初稿结构计划

目标：中文论文初稿，约 12-18 页；以“可投稿雏形”为结构，但避免过度宣称。

### 8.1 拟定标题

可选标题：

1. `面向晶体带隙预测的全局虚拟节点与鲁棒损失图神经网络研究`
2. `基于 ALIGNN 全局虚拟节点增强的晶体带隙预测及 MP-HZW 标签一致性分析`
3. `Global Virtual-Node Enhanced Crystal Graph Neural Networks for Band Gap Prediction with Cross-Source Label Consistency Analysis`

推荐使用第 2 个，能同时覆盖模型与数据质量主线。

### 8.2 摘要

摘要结构：

1. 背景：晶体带隙预测是材料发现关键任务，GNN 是主流方法。
2. 问题：局部消息传递可能缺少全局信息，且跨数据源标签不一致会影响模型排序。
3. 方法：复现 CGCNN/ALIGNN/M3GNet，提出 ALIGNN + virtual node，进一步做 Huber/top-k/warmup/decay 消融。
4. 数据：HZW、MP full、MP-HZW structure matched，以及 relaxed clean equal-atoms 双标签源。
5. 结果：T7/T10 在最终口径下优于 T4；Huber 是最稳定收益来源。
6. 结论：简单全局虚拟节点和鲁棒损失比复杂双槽位注意力更稳；标签一致性是材料 GNN 比较的关键前提。

### 8.3 正文章节

| 章节 | 目标内容 |
| --- | --- |
| 1 Introduction | 材料 bandgap 预测背景、GNN 现状、全局信息问题、标签一致性问题、本文贡献 |
| 2 Related Work | CGCNN、ALIGNN、M3GNet、虚拟节点/全局 token、鲁棒损失、材料数据集与 Matbench/MP |
| 3 Dataset and Data Auditing | HZW、MP、MP-HZW 匹配流程；structure-first audit；relaxed clean equal-atoms 口径 |
| 4 Methods | CGCNN/ALIGNN/M3GNet；E4 virtual node；physics loss；top-k；Huber；warmup；broadcast decay |
| 5 Experimental Setup | 划分、指标、训练 epoch、优化器、硬件、配置文件、随机种子 |
| 6 Results | HZW 早期、MP full、E14-E21 探索摘要、S 系列、最终 T 系列 MP/HZW 双标签结果 |
| 7 Ablation and Discussion | Huber vs top-k、single vs dual VN、E14-E21 失败/边界、size/gate 失败原因、数据源差异 |
| 8 Limitations | bandgap 局部化学主导、MP/HZW 标签冲突、缺少多随机种子、未覆盖介电/压电/声子等任务 |
| 9 Conclusion | 最稳模型、数据清洗结论、后续方向 |
| References | 数字编号引用 |
| Appendix | E1-E21 全量实验表、超参数、额外表格、分桶结果、数据表路径 |

### 8.4 论文贡献表述

论文中建议把贡献写成三点：

1. 系统复现并比较 CGCNN、ALIGNN、M3GNet 及多种 ALIGNN 全局增强模型，确认 ALIGNN + 朴素 virtual node 是稳定强基线。
2. 通过 MP-HZW structure-first audit 证明结构匹配与标签一致性必须分开处理，揭示两个 bandgap 标签源存在金属/非金属冲突和大幅数值差异。
3. 在 relaxed clean equal-atoms 双标签源上完成 Huber/top-k/warmup/decay 消融，发现 Huber loss 是跨标签源最稳定收益来源，而复杂 top-k/双槽位机制具有明显数据源依赖。

避免使用以下过度表述：

- “彻底解决 bandgap 预测问题”
- “全局注意力显著优于所有模型”
- “双超节点在所有数据集上有效”
- “MP/HZW 是完全相同标签”

推荐谨慎表述：

- “results suggest”
- “在当前数据口径下”
- “跨两个标签源最稳的改进为”
- “top-k 未表现出独立稳定收益”
- “复杂全局容量的收益依赖数据源和样本分布”

## 9. 方法公式计划

### 9.1 主任务损失

MSE baseline：

```tex
L_{\mathrm{MSE}} = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)^2
```

Huber loss：

```tex
L_{\delta}(e)=
\begin{cases}
\frac{1}{2}e^2, & |e|\le \delta,\\
\delta(|e|-\frac{1}{2}\delta), & |e|>\delta.
\end{cases}
```

解释：

- 小误差区间类似 MSE，保留精细拟合。
- 大误差区间类似 MAE，降低异常标签和噪声点对训练的破坏。

### 9.2 Virtual node

全局 token 汇聚：

```tex
\mathbf{g}^{(l)} = \mathrm{Pool}\left(\{\mathbf{h}_i^{(l)}\}_{i=1}^{n}\right)
```

广播回节点：

```tex
\mathbf{h}_i^{(l+1)} =
\mathrm{ALIGNNLayer}(\mathbf{h}_i^{(l)}, \mathcal{G})
+ \phi(\mathbf{g}^{(l)})
```

### 9.3 Top-k global collection

```tex
\mathcal{K} = \mathrm{TopK}\left(s_i, k\right), \quad
\mathbf{g} = \sum_{i\in \mathcal{K}} \alpha_i \mathbf{h}_i
```

其中 `k = clip(ceil(0.2n), k_min, k_max)`，最终全尺寸版本取消了只在特定原子数范围启用 top-k 的限制。

### 9.4 Warmup top-k

```tex
\mathbf{g}_{\mathrm{update}} =
(1-\alpha_t)\mathbf{g}_{\mathrm{mean}} + \alpha_t \mathbf{g}_{\mathrm{topk}}
```

训练前期 `alpha_t=0`，先使用全局均值汇聚；中期线性增大；后期使用 top-k 汇聚。

### 9.5 Global broadcast decay

```tex
\mathbf{h}_i^{(l+1)} =
\mathrm{ALIGNNLayer}(\mathbf{h}_i^{(l)}, \mathcal{G})
+ \beta_t \phi(\mathbf{g}^{(l)})
```

训练后期逐步降低 `beta_t`，减少全局 token 对局部节点表示的噪声注入。

### 9.6 Physics loss

```tex
L = L_{\mathrm{gap}} + \lambda_{\mathrm{aux}}L_{\mathrm{CBM/VBM}}
+ \lambda_{\mathrm{phys}}L_{\mathrm{gap=CBM-VBM}}
+ \lambda_{\mathrm{nonneg}}L_{\mathrm{nonneg}}
```

有效 mask：

```tex
m_i = \mathbb{1}\left(|\mathrm{CBM}_i|>\epsilon \lor |\mathrm{VBM}_i|>\epsilon\right)
```

只在 `m_i=1` 的样本上计算 CBM/VBM 辅助损失和物理一致性损失。

## 10. 图表计划

### 10.1 PPT 图表

| 图表 | 类型 | 来源 |
| --- | --- | --- |
| 实验路线图 | TikZ/表格 | 本计划和历史 docs |
| 数据清洗流程图 | TikZ | structure-first audit summary |
| 模型架构示意 | TikZ/简图 | `alignn_variants.py` |
| HZW 早期结果表 | 表格 | `gnn_project_history_report.tex` |
| MP/HZW final T 结果表 | 表格 | 远端 `comparison_summary.csv` |
| Huber/top-k 消融结论表 | 表格 | T6/T7/T8/T9/T10 |
| S4/S13 大结构分析 | backup 表 | `s4_vs_s13_size_analysis.md` |

### 10.2 论文图表

| 编号 | 标题 | 目的 |
| --- | --- | --- |
| Figure 1 | Overall workflow | 从数据审计到模型训练的完整流程 |
| Figure 2 | ALIGNN + virtual node architecture | 展示主模型结构 |
| Figure 3 | MP-HZW data auditing pipeline | 展示结构匹配和标签冲突 |
| Figure 4 | Huber/top-k/decay variants | 展示 T 系列消融关系 |
| Table 1 | Dataset statistics | 各数据阶段样本数 |
| Table 2 | Model definitions | E/T 系列定义 |
| Table 2b | Full experiment lineage | E1-E21、S 系列、T 系列全量索引 |
| Table 3 | HZW early results | 早期动机 |
| Table 4 | Final MP label T-series results | 最终 MP 标签主结果 |
| Table 5 | Final HZW label T-series results | 最终 HZW 标签主结果 |
| Table 6 | Data quality conflict counts | MP-HZW 标签冲突 |
| Appendix Table A1 | E14-E21 exploratory ablations | 底层机制探索的定义、动机、结论归类 |
| Appendix Table A2 | Hyperparameters | 训练配置 |
| Appendix Table A3 | Size-bucket results | 大结构分析 |

## 11. 参考文献计划

参考文献使用数字编号 `[1]`。PPT 和论文第一版都可用 `thebibliography`，保证编译稳定。

必须包含：

```text
[1] Tian Xie and Jeffrey C. Grossman. Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties. Physical Review Letters, 120(14):145301, 2018.
[2] Kamal Choudhary and Brian DeCost. Atomistic Line Graph Neural Network for improved materials property predictions. npj Computational Materials, 7:185, 2021.
[3] Chi Chen and Shyue Ping Ong. A universal graph deep learning interatomic potential for the periodic table. Nature Computational Science, 2:718-728, 2022.
[4] Anubhav Jain et al. Commentary: The Materials Project: A materials genome approach to accelerating materials innovation. APL Materials, 1:011002, 2013.
[5] Alexander Dunn et al. Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm. npj Computational Materials, 6:138, 2020.
```

可选补充：

- ComFormer 相关论文，用于讨论 attention/long-range crystal transformer。
- Virtual node / graph transformer 相关论文，用于 related work。
- Huber loss 原始或鲁棒回归参考，若论文需要更正式背景。

## 12. 执行步骤

### Step 1：同步结果与核对

必须先把远端最终结果同步到本地，或至少把数值写入报告时注明来源。

推荐同步命令：

```powershell
scp -i .\tg100025.pem -r tg100025@124.220.81.245:/data/home/tg100025/gnn_work/results/mp_hzw_relaxed_clean_equal_atoms_mp_t results\
scp -i .\tg100025.pem -r tg100025@124.220.81.245:/data/home/tg100025/gnn_work/results/mp_hzw_relaxed_clean_equal_atoms_hzw_t results\
```

核对命令：

```powershell
Get-Content results\mp_hzw_relaxed_clean_equal_atoms_mp_t\comparison_summary.csv
Get-Content results\mp_hzw_relaxed_clean_equal_atoms_hzw_t\comparison_summary.csv
```

若远端不可用，则使用本计划中已核对的 CSV 数值，但需要在最终答复中说明结果源为已读取的远端汇总。

### Step 2：生成 PPT `.tex`

基于 `gnn_project_history_report.tex` 改写，不直接在旧文件上编辑。

实现细节：

- 复制其 preamble、theme、footline、bibliography 设置。
- 用 `\section{}` 分为背景、方法、数据、结果、讨论、结论。
- 所有大表格必须先检查宽度。
- 对公式页使用 `\small` 或 `\scriptsize`，复杂公式分行，不再让公式溢出页面。

### Step 3：生成论文 `.tex`

新建 `gnn_bandgap_research_paper_draft.tex`。

实现细节：

- 用 `ctexart`。
- 摘要和关键词放在首页。
- 主文用中文，模型名、变量名保留英文。
- 表格较长时使用 `longtable`。
- 附录放完整实验矩阵和额外分桶结果。
- 第一版先用 `thebibliography`，避免 BibTeX 样式缺失导致编译失败。

### Step 4：编译 PDF

推荐命令：

```powershell
xelatex -interaction=nonstopmode -halt-on-error gnn_bandgap_research_final_slides.tex
xelatex -interaction=nonstopmode -halt-on-error gnn_bandgap_research_final_slides.tex

xelatex -interaction=nonstopmode -halt-on-error gnn_bandgap_research_paper_draft.tex
xelatex -interaction=nonstopmode -halt-on-error gnn_bandgap_research_paper_draft.tex
```

如果 `xelatex` 不存在：

```powershell
where xelatex
```

若本机缺 LaTeX 环境，需要说明无法本地编译，并保留 `.tex` 源文件。

### Step 5：PDF 检查

检查项：

- PDF 文件存在且大小非 0。
- `.log` 中没有 `Fatal error`。
- PPT 第 1 页标题页正常。
- PPT 目录页正常。
- 参考文献页是 `[1]` 文本编号，不是图标。
- 所有公式完整显示，尤其 Huber loss、physics loss、E12 size loss。
- 大结果表没有出界或压到无法阅读。
- 论文页码正常。
- 论文摘要、正文、参考文献、附录都出现。

PowerShell 检查命令：

```powershell
Get-Item gnn_bandgap_research_final_slides.pdf
Get-Item gnn_bandgap_research_paper_draft.pdf
Select-String -Path gnn_bandgap_research_final_slides.log -Pattern "Fatal error|Emergency stop|Undefined control sequence"
Select-String -Path gnn_bandgap_research_paper_draft.log -Pattern "Fatal error|Emergency stop|Undefined control sequence"
```

## 13. 质量控制规则

### 13.1 结论表述

必须坚持以下主结论：

- ALIGNN 是最强传统 baseline。
- 在两个最终标签源上，single plain virtual node 都稳定提升 ALIGNN。
- Huber loss 是最稳定的进一步提升来源。
- MP label 上 T10 最优，但 HZW label 上 T7 最优。
- 跨数据源最稳方案是 T7：`ALIGNN + single plain virtual node + Huber loss`。
- top-k 不应被包装成稳定成功机制；它在 T8 中单独使用时没有稳定收益。
- 双节点/双槽位对大结构有局部信号，但整体收益不稳定，不能作为最终主模型结论。

### 13.2 数据源表述

必须明确区分：

- HZW bandgap 标签。
- MP summary `band_gap` 标签。
- 同一结构匹配不等于同一标签。
- relaxed clean equal-atoms 是当前最终主口径。

禁止混淆：

- 不要把 HZW E13 的 `0.2119` 与最终 relaxed clean HZW T7 的 `0.1940` 当成同一数据划分直接比较。
- 不要把 strict clean S4 的 `0.200094` 与 relaxed clean T4 的 `0.232520/0.201762` 当成同一实验口径。
- 不要把 test-only routing diagnostic 写成正式模型结果。

### 13.3 图表与排版

- PPT 每页不超过一个大表。
- 表格列太多时拆成 MP label 和 HZW label 两页。
- 中文段落不堆满页面，优先用短句和表格。
- 论文可以详细，但 PPT 只保留主线。
- 所有路径和配置放 backup 或论文附录，不放 PPT 主线。

## 14. 风险与应对

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| LaTeX 本地缺环境 | 无法编译 PDF | 先交付 `.tex`，说明 `xelatex` 不可用 |
| 表格过宽 | PPT/PDF 显示不完整 | 使用 `resizebox`、拆页或缩小字体 |
| 参考文献格式错误 | 影响汇报观感 | 使用 `thebibliography` + `[text]` item |
| 结果口径混淆 | 结论被质疑 | 每张结果表写清数据源和清洗口径 |
| 过度宣称 top-k/双节点 | 与实验不符 | 明确 top-k 不稳定，Huber 是主收益 |
| 远端结果未同步 | 无法复核 | 编译前先 `scp` 或 SSH 读取 CSV |
| 中文字体问题 | 编译失败或乱码 | 使用 `ctex` + `xelatex` |

## 15. 可并行任务划分

如果后续使用 subagent，可以按以下边界并行：

| 任务 | 输入 | 输出 | 写入范围 |
| --- | --- | --- | --- |
| Agent A：结果核对 | 远端/local CSV、metadata | 标准化结果表和数据统计 | 只写临时 markdown/CSV |
| Agent B：PPT 写作 | 模板、结果表、历史讲稿 | `gnn_bandgap_research_final_slides.tex` | 只写 PPT 源 |
| Agent C：论文写作 | 计划、历史 docs、结果表 | `gnn_bandgap_research_paper_draft.tex` | 只写论文源 |
| Agent D：编译 QA | `.tex`、LaTeX 环境 | PDF、log 检查报告 | 不改正文，除非修编译错误 |

主 agent 需要最终统一审查：

- 结果数值一致性。
- 结论不互相矛盾。
- 参考文献格式。
- PDF 编译状态。

## 16. 最终验收标准

完成后必须满足：

- `gnn_bandgap_research_final_slides.pdf` 存在并可打开。
- `gnn_bandgap_research_paper_draft.pdf` 存在并可打开。
- PPT 至少包含标题页、目录、背景、方法、数据清洗、结果、讨论、结论、参考文献。
- 论文至少包含标题、摘要、引言、相关工作、数据与审计、方法、实验设置、结果、讨论、结论、参考文献。
- MP/HZW 最终 T 系列结果数值与 `comparison_summary.csv` 一致。
- 参考文献以 `[1]`、`[2]` 形式显示。
- 没有写入 API key、服务器密码、私钥内容等敏感信息。
- 最终答复中说明执行过的编译命令和是否成功。
