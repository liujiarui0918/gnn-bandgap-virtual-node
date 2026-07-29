# 审稿意见整改计划：统计稳健性、基线公平性、机制证据与写作修订

日期：2026-07-14
针对对象：`gnn_bandgap_research_paper_draft_en.tex`（2026-07-12 版）及配套 `gnn_bandgap_research_supplementary_en.tex`
意见来源：师兄转发的 10 条审稿意见（6 条 Major + 4 条 Minor）
文档性质：整改计划。本轮只产出计划；执行时按批次推进，每完成一个批次回到本文档勾销并同步论文。

---

## 执行进度（2026-07-15 更新）

**已完成（本地，无 GPU）：**
- 工作流 A：A2–A8 全部落地并实跑（`scripts/analyze_paired_bootstrap.py`、`analyze_oversmoothing_ci.py`、`build_decay_dynamics_figure.py`、`analyze_cross_label_eval.py`、`extract_hzw_structmeta.py`+`analyze_decay_cohorts.py`、`analyze_strict_subset_metrics.py`、`analyze_metal_f1_thresholds.py`），产物在 `results/analysis/`。A1（远程同步 T2 等预测）**未做**——依赖 ssh，T4−T2/T7−T2 bootstrap 对与 strict 子集 T2 行留待同步后补。
- 工作流 B 前置：`seed_everything` 补 numpy/dgl；`run_t_relaxed_matched_experiments.py` 加 `--seed`/`--results-suffix`（默认行为不变，单测 6/6 过）；新增 `scripts/slurm_t_multiseed_queue.sh`（seeds 1042/2042，核心 6 模型）；`scripts/aggregate_multiseed.py` 就绪并在 seed-42 上验证。
- 工作流 C：主论文与 SI 全面修订并 pdflatex 双遍编译通过（主文 15 页 0 undefined，SI 8 页）。意见 7/8/9/10 写作项全部落地；自查问题 A–N 全部关闭（A 两面板、B/C 公式对齐实现、J 含 fig1 tikz 内公式、E 统一 4 位小数、其余见 diff）。12 条新引用入文（4 条高风险书目已 WebSearch 核实）。fig2/3/4 重生成（描述名标签、两面板账目、符号约定标注）。SI 新增 S8–S16 节（表 S6–S13、图 S4）。response letter 骨架已建并回填真实数字。
- **关键实证结论（已写进论文）**：bootstrap 显示 T10−T7 两源均不显著（预注册规则触发，摘要/结论已降级）；交叉标签评估证明 decay 伤害是训练动态效应；strict 子集显示 MP 上 Huber 增益集中在干净子集（clean −0.0089 vs noisy +0.0140）。

**待远程 GPU（P0/P1/P2 原计划不变）**：过平滑全测试集重跑、敏感性 8 runs、strict-Huber 补跑、M3GNet 调参。**多种子 24 runs 已于 2026-07-19 深夜全部完成并落地论文**（见下）。

**P0-a 多种子最终结果（2026-07-19/20）：**
- 24/24 runs 完成（作业 1881，约 3 天 5 小时），结果已全部同步本地并聚合（`results/analysis/multiseed_{summary,pairs}.csv`）。
- **T4−T2 与 T7−T4 两源全部 3/3 方向一致**（MP：−0.0080±0.0006 / −0.0088±0.0013；HZW：−0.0104±0.0011 / −0.0086±0.0009）——论文主线经受住种子检验。
- **T10−T7 两源都不稳定**（MP −0.0004±0.0017、2/3；HZW +0.0018±0.0023、2/3）——与 bootstrap 独立印证"不可分"，预注册规则触发并已落实到摘要/6.2/7.2/结论。
- 划分不变性冒烟验证通过（sample_id 跨 seed 完全一致）；早停 patience=8 仅在 M3GNet 触发（49 停、best 41，无影响）。
- 论文 Table 3 已改三种子 mean±std；fig3 重建为 6 面板（误差棒+parity+残差并入）；新增 fig5 物理分析图与 6.3 节（元素周期表热图/晶系/带隙分桶）——响应导师"散点拼大图+补物理图"要求。
**导师决策（2026-07-15，已落地）：**
- **多种子加早停**：训练循环已加 `early_stopping_patience`（val-MAE 连续 N epoch 不降即停，仍取 best-val checkpoint）；三个 runner + CLI `--early-stopping-patience` + slurm 脚本 `EARLY_STOP_PATIENCE=8` 全部就位。已实证：18 个 seed-42 run 的最长 pre-best 停滞仅 6 epoch，patience=8 不会改变任何已报结果；论文 5 节复现性段与 letter 意见 1 已写入此说明。
- **HZW 定名**：Hongzhiwei Technology (Shanghai) Co., Ltd.（鸿之微科技（上海）股份有限公司，hzwtech.com）；论文脚注/摘要/数据可用性/letter 全部定稿，删除待确认占位。
- **撤销形成能（意见 4）**：不做第二性质实验，论文收紧为纯带隙研究；Limitations 第一条改为明确的范围声明（HZW 无第二性质标签→做形成能须放弃跨源主轴），letter 意见 4 改为 scope-argument + Okabe VGNN 文献回应，标 [DONE — no new experiment]。

**待外部确认**：MP API key（现已无需——形成能撤销）。实际仅剩 GPU 训练排期一项。

**2026-07-15 远程侧进展：**
- **A1 部分完成**：T1/T2/T3/T6/T8/T9 双源 predictions/metrics/history 已从远程同步到本地（36 个文件）；4 个依赖 T2 的分析已重跑，`paired_bootstrap.csv` 现为完整 10 对——**T4−T2 与 T7−T2 在两源均显著**（mp T4−T2 −0.0075 CI[−0.0121,−0.0028]；hzw −0.0104 CI[−0.0147,−0.0060]），主论文 6.2/7.1 已同步收紧措辞。S 系列明细尚未同步。
- **D0 QOS 探测完成**：`sbatch -p tx4090 --gres=gpu:1` 探测作业 1863 成功分配 RTX 4090 D 并运行（节点 ins-l2irzp0l）——**配额已恢复，工作流 B 可启动**。集群 sbatch 会自动对齐 1 卡 =10 核/90GB 内存。注意 sbatch 需 login shell（`bash -lc`）。
- **多种子队列已发车（P0-a，作业 1881）**：2026-07-15 19:15 提交并进入 R 状态（节点 ins-l2irzp0l）。部署流程：远程原文件已备份至 `~/gnn_backup_pre_multiseed_20260715/` → 8 个文件（CLI、common、三个 runner、slurm 队列、聚合脚本、部署脚本）从 `~/gnn_incoming_multiseed/` 拷入 `/data/home/tg100025/gnn` → 远端 ast 语法校验通过 → sbatch。队列顺序：seed 1042 (mp→hzw) → seed 2042 (mp→hzw)，每源 T1/T2/T3/T4/T7/T10，早停 patience=8，图缓存与数据集目录跨 seed 复用，结果写 `..._t_seed{1042,2042}/`。日志：`gnn_work/logs/slurm-gnn_t_multiseed-1881.out`。预计 2.5–3 天（早停会再省一些）。跑完后本地 `py -3 scripts/aggregate_multiseed.py` 出 mean±std。注意：该服务器 SSH 有连接频率限制（约数分钟一个窗口），检查时失败就等 6 分钟重试。

---

## 0. 结论先行

1. **10 条意见没有任何一条会推翻论文现有主结论**（"single plain virtual node + Huber 是跨标签源最稳方案"）。缺口集中在四类：统计证据（意见 1/3）、基线与超参的公平性说明（意见 2/10）、机制主张的数据支撑（意见 4/5/6）、术语与引用规范（意见 7/8/9）。
2. 整改按**三条并行工作流**组织：
   - **工作流 A：零成本本地分析**（不需要任何 GPU，1–3 天可全部完成）——配对 bootstrap 置信区间、交叉标签评估、strict 子集重评估、decay 受损样本画像、训练动态曲线、F1 阈值敏感性。这些直接实质性回应意见 5、6、8 的大部分和意见 1、3 的一部分。
   - **工作流 B：GPU 补充实验**（合计约 130–140 GPU·h，单张 4090 约 5.5–6 天纯算力，分 P0/P1/P2 三批）——多种子重训、超参敏感性、strict-clean 补跑、形成能第二目标、M3GNet 调参、过平滑全测试集诊断。
   - **工作流 C：写作与引用修订**（与 A/B 并行）——HZW 定义、系列重命名、引用补充、阈值与超参说明、实验设置章节扩写、以及第 7 节列出的自查问题修复。
3. **两个外部依赖必须第一天就去解决**：
   - 远程 Slurm GPU 配额状态（2026-05-25 账号曾被 QOS 封禁 `MaxTRESPerJob=gres/gpu=0`，T 系列训练期间已恢复，**当前状态未知**，直接决定工作流 B 能否启动）；
   - HZW 数据来源的正式披露口径与引用方式（意见 7 的硬依赖，需师兄/导师确认，见第 13 节）。
4. 若 GPU 完全不可用：意见 7/8/9/10 仍可完整回应，意见 5/6 可回应约七成，意见 1/3 可给出部分回应（bootstrap CI + 现有数据的误差棒），意见 2 可给出解释性回应；**只有意见 4（第二性质）强依赖 GPU**。降级预案见第 11 节。

---

## 1. 意见分类与处置总览

| # | 意见要点 | 类型 | 需要 GPU | 批次 | 新实验量 | 零成本部分 | 落点 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 单种子、无统计显著性 | 补实验+统计 | 是 | **P0** | 24 runs ≈ 72 h | 配对 bootstrap CI（立即） | Table 3、Fig 3、摘要、5/6 节、SI |
| 2 | M3GNet 表现异常无解释 | 补实验+写作 | 少量 | P2 | 3 runs ≈ 6–9 h | 配置披露+文献对照 | 4.2、7 节新段、SI |
| 3 | 过平滑诊断仅 240 样本 | 补实验（推理级） | 推理 | **P0** | 推理 ≈ 2–6 h | 现有 240 样本数据先加误差棒 | 7.5 节、Table 4、Fig 4、SI |
| 4 | 缺全局依赖性质验证 | 补实验+数据抓取 | 是 | P2 | 3–4 runs ≈ 12–16 h | 无（标签从未抓取过） | 新小节、Limitations |
| 5 | decay 分化解释靠猜测 | 补分析+少量实验 | 少量 | P1 | 4 runs ≈ 16 h | 四项分析全部零成本 | 7.2 节重写、SI |
| 6 | relaxed 噪声可能掩盖架构增益 | 补实验+复用 S 系列 | 少量 | P1 | 1–2 runs ≈ 4–8 h | strict 测试子集重评估（立即） | 6 节新小节、SI |
| 7 | HZW 未定义、系列代号内部化 | 写作 | 否 | C | — | 全部 | 全文+图内标签重生成 |
| 8 | 金属/非金属 F1 阈值未定义 | 写作+微量分析 | 否 | C | — | 阈值敏感性表（立即） | 3.1/5 节、Table 1/3 |
| 9 | 过平滑与虚拟节点经典引用缺失 | 写作 | 否 | C | — | 全部 | 1/2/7.5 节、参考文献 |
| 10 | 超参选择无说明、无敏感性 | 写作+补实验 | 与 5 共享 | P1 | 4 runs ≈ 16 h（Huber δ） | 超参溯源表（立即） | 4.6/4.7/5 节、SI |

补充：第 7 节另列 **14 项自查发现的问题（A–N）**，包括 Table 1 冲突计数口径不自洽、论文公式与实现不一致（broadcast 无 φ、Huber 实为 smooth-L1/β 形式）等，属于师兄"自己也读一遍"要求的落实，与意见修订一并执行。

---

## 2. 现有资产事实清单（本计划的事实基础，已逐项核实）

### 2.1 结果资产

| 资产 | 位置 | 状态 |
| --- | --- | --- |
| T 系列汇总指标（9 模型 ×2 标签源，含 `train_seconds`/`peak_gpu_gb`） | `results/mp_hzw_relaxed_clean_equal_atoms_{mp,hzw}_t/comparison_summary.csv` | 本地齐全 |
| 逐样本测试集预测 `predictions.csv`（列：`sample_id,target,prediction,split`，test≈8,455 行） | 同上目录下 `T4_*/`、`T7_*/`、`T10_*/` | **本地仅 T4/T7/T10**；T1/T2/T3/T6/T8/T9 需从远程同步（所有 runner 都会写 `predictions.csv`，`faithful_gnn/cgcnn_runner.py:463` 等，远程应该都有） |
| 每 epoch 训练曲线 `history.json`（epoch/lr/train_loss/train_mae/val_loss/val_mae × 50） | T4/T7/T10 子目录 + E 系列 7 个模型 | 本地部分，其余在远程 |
| 训练好的 checkpoint（`best_model.pt`） | 远程 `/data/home/tg100025/gnn_work/results/...` | **本地一个都没有**；过平滑诊断当时是在远程登录节点 CPU 上跑的（脚本注释可证），240 样本正是登录节点算力限制所致 |
| T7/T10 双源逐样本合并表（含 num_atoms/桶/gap_diff/is_metal） | `results/analysis/global_decay_transfer_records.csv`（5.1 MB） | 本地，意见 5 的主数据 |
| decay 桶级分析 | `results/analysis/global_decay_transfer_{grouped_summary,overall,t10_minus_t7}.csv` + `global_decay_transfer_explanation.md` | 本地 |
| 过平滑逐样本×逐层原始数据（240 样本 ×16 层 ×4 模型 ×2 源） | `results/analysis/oversmoothing_hooks_{mp,hzw}/oversmoothing_sample_metrics.csv` | 本地；抽样 id 与分桶配额有完整记录（`_mini_dataset_test240_train8_val8/{splits.json,metadata.json}`，桶配额 41/41/40/40/40/38） |
| S 系列（strict-clean，MP 标签，test n=8,154） | 本地仅 `results/mp_hzw_final_clean_equal_atoms_mp_s/analysis/`（S4 vs S13/S8 分桶）；完整 `comparison_summary.csv` 在远程 | S4=单 VN 0.200094，S13=双 VN 0.201181；**S 系列没有 Huber 版本** |
| E 系列（早期 HZW） | `results/mp_hzw_matched_clean_e1_e12/`（E2/E3/E4/E5/E7/E8/E10 的 metrics+history，无逐样本预测） | 本地部分 |
| 结构审计与冲突清单（material-id 级明细，含 MP id + hc id + 双方 gap + split） | `results/data_quality/structure_first_strict_180/`（论文口径）；另有三个旧版本目录 | 本地齐全 |
| 全量审计主表（95,462 行，含双方公式/gap/is_metal/体积/匹配状态/冲突码） | `results/data_quality/mp_hzw_full_audit_table.remote.csv`（25 MB） | 本地 |
| MP summary 抓取缓存 | `results/data_quality/mp_hzw_summary_fetch_cache.jsonl`（588 MB） | 字段仅 `material_id,band_gap,cbm,vbm,is_metal,formula_pretty,nsites,density,structure`——**没有形成能、没有介电** |
| HZW 原始下载（10 万条，实为纯 JSONL） | `api_downloads/api_downloads/crystal_band_100000.jsonl/crystal_band_100000.jsonl.gz.tmp`（648 MB） | 每条含 `space_group(_number)`、`crystal_system`、`hc_id`、`source_id`(=MP id)、`crystal_band[{band_gap,cbm,vbm,...}]`——**空间群/晶系可作意见 5 的画像协变量**；同样没有形成能/介电 |

### 2.2 代码事实（决定实验实现方式）

| 事项 | 事实 | 位置 |
| --- | --- | --- |
| 随机种子 | 所有 T 系列 YAML 固定 `seed: 42`；`seed_everything` 只 seed `random`+`torch`（**漏 numpy，也未 seed dgl**） | `configs/mp/MP_*.yaml:3`；`faithful_gnn/common.py:16-20`；`alignn_runner.py:787` |
| train/val/test 划分 | **在数据导出时固定**（`random.Random(seed=42)` 打乱后 8/1/1 切分，split 写进 `manifest.jsonl` 每条记录和 `splits.json`）；训练侧从 manifest 读回、`keep_data_order=True`——**换训练 seed 不会改变划分**（正是多种子协议需要的行为，仍需 1-epoch 冒烟验证） | `export_mp_hzw_final_clean_dataset.py:365-374`；`faithful_gnn/common.py:58-71`；`alignn_runner.py:233-257` |
| seed 的 CLI | `run_faithful_experiment.py` 与 `run_t_relaxed_matched_experiments.py` **都没有 `--seed` 参数**；需在 `generate_config`（`run_t_relaxed_matched_experiments.py:227-266`）注入；resume 逻辑按 `metrics.json` 是否存在跳过（:441-444），多种子必须换 results 目录 |
| M3GNet（T3） | **matgl 从头训练**（`M3GNet(...)` 随机初始化，无预训练加载）；cutoff 5.0 / threebody 4.0 / nblocks 3 / units 128 / embedding 64 / max_n=max_l=3 / weighted_atom readout；Adam lr 1e-3、cosine、grad_clip 5、50 epochs、MSE、batch 48–64 AMP；**全仓无任何调参记录**；无 early stopping（按 best-val 存 checkpoint，跑满 50） | `faithful_gnn/m3gnet_runner.py:609-628`；`configs/mp/MP_E3_m3gnet_faithful.yaml`；本地有 `.vendor/matgl/pretrained_models/M3GNet-MP-2018.6.1-Eform/` 可做预训练初始化备选 |
| 金属/非金属 F1 | **非对称双阈值**：真值金属 = `target <= 1e-8`；预测金属 = `prediction <= 0.1 eV`（默认 `metal_threshold=0.1`，T 系列未覆写） | `crystal_gnn/utils.py:62-91` |
| Huber loss | 实现为 `F.smooth_l1_loss(..., beta=0.5)`，即 PyTorch smooth-L1（= Huber/δ 的缩放形式），`final_gap_huber_beta: 0.5`；**全仓只有 0.5 一个值，无扫描** | `faithful_gnn/alignn_runner.py:489-495`；`configs/mp/MP_T7/T10_*.yaml:26-27` |
| 虚拟节点（T4/T7 主干） | `ALIGNNVirtualNodeRegressor`：VN 初值 `zeros(1,256)` 参数；**pool = scatter_mean**；**broadcast 是恒等加法 `h + vn[batch]`，没有 φ 变换**（论文式 (2) 写了 φ(g)，与实现不符）；VN 自更新 φ = 2 层 MLP(256→256, SiLU) 残差式；在 4 个 ALIGNNConv 和 4 个 EdgeGatedGraphConv **每层之前**各交换一次（共 8 次）；读出 = 图级 AvgPooling 与最终 VN concat → `Linear(512,256)+SiLU` → dropout → `Linear(256,1)` | `faithful_gnn/alignn_variants.py:277-326`；`crystal_gnn/utils.py:43-51` |
| top-k（T6/T8/T9） | fraction 0.2、clip [4, 32]、enable 区间默认不限制；T9 warmup 10 epochs + ramp 10 epochs 线性混合 | `alignn_variants.py:329-397, 400-414`；`configs/mp/MP_T6/T9_*.yaml` |
| broadcast decay（T10） | start=10、end=50、min_scale=0.25，线性；scale 只乘在 VN→原子 broadcast 上；**只能改 YAML，无 CLI** | `alignn_variants.py:417-452`（:449 应用点）；`configs/mp/MP_T10_*.yaml:41-43` |
| T 系列导出过滤 | `--metal_conflict_mode zero_vs_nonzero --zero_gap_threshold 1e-6 --disable_gap_diff_filter --disable_volume_filter --require_equal_atom_count` | `run_t_relaxed_matched_experiments.py:374-395` |
| 过平滑诊断脚本 | CLI 已支持全量（`--max-samples` 默认 2048，传 8455 即可；`--device cuda/cpu`）；分层抽样 = 按原子数桶轮转直至配额；依赖 `results-root/<模型目录>/best_model.pt` | `scripts/analyze_oversmoothing_hooks.py:51,117-139` |
| 产图脚本 | fig2/3/4 由 `scripts/build_academic_paper_figures.py` 从 `comparison_summary.csv`、`figures/huber_diagnostics/huber_residual_summary.csv`、oversmoothing final-layer CSV 生成——**图内 T2/T4/T7/T10 标签是脚本写死的，重命名与加误差棒都要改此脚本重生成** | `scripts/build_academic_paper_figures.py:63-65,148-149,223-225` |
| 测试 | `tests/` 有 topk/warmup/decay 调度、runner 配置生成、导出过滤等单测，conftest 用假模块隔离重依赖，本地可跑 | `tests/test_t6_topk_huber_virtualnode.py`、`tests/test_t_relaxed_runner_config.py`、`tests/test_export_mp_hzw_final_clean_dataset.py` 等 |

### 2.3 数据口径关键数字（写作时必须对齐的账）

- 审计：95,462 → 结构匹配 92,787（92,507 exact + 280 supercell）/ 缺 MP 2,636 / 不匹配 39。
- **审计冲突统计口径**（阈值：metal-like ≤0.05 eV 或 `is_metal`，nonmetal ≥0.10 eV）：`mp_metal_hzw_nonmetal` 3,142、`mp_nonmetal_hzw_metal` 3,046、same-class >1 eV 1,121、无任何冲突 85,478。
- **T 系列实际过滤口径**（零隙定义 1e-6 eV）：zero/nonzero 冲突 7,984（MP零/HZW非零 2,935 + HZW零/MP非零 5,049）→ 84,803；原子数关系非 equal 增量 −252 → 84,551；导出期安全检查 −1 → **84,550**；split 67,640/8,455/8,455。
- 两套口径**不能混排在同一列账里**——现在论文 Table 1 恰恰混了（见第 7 节问题 A）。
- strict-clean（S 系列）：`mp_is_metal==hzw_is_metal` 且 |Δgap|≤0.5 eV 且体积差≤8%；equal-atoms 版 test n=8,154（确切总数以远程 metadata 为准，本地未同步）。

### 2.4 基础设施与成本参数

- 远程：`ssh -i tg100025.pem tg100025@124.220.81.245`；项目 `/data/home/tg100025/gnn`；工作根 `/data/home/tg100025/gnn_work`；Slurm 分区 `tx4090`（RTX 4090 24 GB），模板 `scripts/slurm_t_relaxed_mp_queue.sh`（gpu:1 / cpus 10 / mem 80G / 7d）。
- 实测单 run 训练时长（50 epochs，来自 `comparison_summary.csv` 的 `train_seconds`）：

| 模型 | 每 run 用时 | 峰值显存 |
| --- | ---: | ---: |
| CGCNN（T1） | ≈ 0.20 h | 3.0 GB |
| M3GNet（T3） | ≈ 1.5 h | 2.5 GB |
| ALIGNN 系（T2/T4/T7/T10） | ≈ 3.9–4.1 h | 10.2–10.3 GB |
| top-k 系（T6/T8/T9） | ≈ 4.7 h | 10.2 GB |

- 风险记录：2026-05-25 账号 QOS 曾拒绝一切 GPU 作业（`AssocMaxNodePerJobLimit`）；后恢复。**执行第一天必须先跑最小 GPU 探测作业确认配额。**

---

## 3. 逐条意见整改方案

每条按固定格式：意见本质 → 立场 → 行动清单（编号，标注工作流与批次）→ 论文落点 → 验收标准 → 降级方案。

### 3.1 意见 1：多种子与统计显著性 【Major，P0，最高优先级】

**意见本质**：T4/T7/T10 之间 MAE 差异只有 0.001–0.008 eV 量级，单 seed 单 split 无法支撑排序结论。审稿人要求 ≥3 seeds、Table 3 与 Fig 3 报均值±标准差。

**立场**：完全接受。这也是论文 Limitations 第三条自己承认的问题，是全部意见里对结论可信度影响最大的一条。

**行动**：

1.1 【A，立即】**同步远程逐样本预测**：`scp` 拉取 T1/T2/T3/T6/T8/T9 的 `predictions.csv` + `metrics.json` + `history.json`（两标签源），补齐本地缺口。这是多个零成本分析的前置条件。
```powershell
scp -i .\tg100025.pem -r tg100025@124.220.81.245:/data/home/tg100025/gnn_work/results/mp_hzw_relaxed_clean_equal_atoms_mp_t/T2_ALIGNN_baseline results\mp_hzw_relaxed_clean_equal_atoms_mp_t\
# 其余模型与 hzw_t 同理；也可在远程 tar 后一次拉回
```

1.2 【A，立即】**配对 bootstrap 置信区间**（新脚本 `scripts/analyze_paired_bootstrap.py`）：对每个关键模型对（T4−T2、T7−T4、T10−T7、T7−T2、T10−T4），在两标签源上分别做 B=10,000 次对测试样本的有放回重采样，报告 ΔMAE 的 95% CI 与 P(Δ<0)。bootstrap 随机种子固定（如 20260714）写进脚本。它回答"该差异是否超出测试集抽样噪声"，与训练随机性正交，先行给出、且写进 SI。

1.3 【B，P0】**多种子重训**：模型 {T1, T2, T3, T4, T7, T10} × 标签源 {mp, hzw} × 新 seed {1042, 2042}（与已有 42 合成 3 seeds）= **24 runs ≈ 71.5 GPU·h**。审稿人只点名 T2/T4/T7/T10，但 T1/T3 每源每 seed 只多 1.7 h，补齐后整张 Table 3 都有 ±std，值得。T6/T8/T9（top-k 系）保持单 seed，SI 注明原因（非主贡献、成本高）。
   - 代码改动：
     a. `run_t_relaxed_matched_experiments.py` 增加 `--seed`（注入 `config["seed"]`）与 `--results_suffix`（或复用现有 results 目录参数，以复核为准），保证 resume-skip 逻辑按 seed 隔离；
     b. `faithful_gnn/common.py:seed_everything` 补 `np.random.seed(seed)` 与 `dgl.random.seed(seed)`（不影响已完成的 seed-42 结果，只影响新 run）；
     c. 新 Slurm 脚本 `scripts/slurm_t_multiseed_queue.sh`：外层循环 seed，内层沿用现有 mp→hzw 队列；结果目录 `.../mp_hzw_relaxed_clean_equal_atoms_{mp,hzw}_t_seed{1042,2042}/`。
   - **划分不变性冒烟验证**（防御性，必做）：先用 seed=1042 跑 `--epochs 1` 的 T4，diff 其 `predictions.csv` 的 test `sample_id` 集合与 seed-42 版完全一致，确认换 seed 只改初始化/shuffle、不改 split。
1.4 【A，P0 完成后】**聚合与统计报告**（新脚本 `scripts/aggregate_multiseed.py`）：
   - 每模型×源报 mean ± 样本标准差（n=3, ddof=1）；
   - 每对模型报三项：seed 间 mean±std、逐 seed 的配对 bootstrap CI、方向一致性（3/3 seeds 同号？）；
   - **明确不做 n=3 的 t 检验显著性宣称**（功效不足），以"多种子均值±std + 逐 seed 配对 CI + 方向一致性"三件套作为证据标准，并在 5 节写明协议：split 固定、只变训练随机性（初始化+shuffle），与审稿人"固定划分+单一种子"的双重关切分开回应。
1.5 【C】论文落点：Table 3 改为 mean±std（保留单 seed 完整精度值于 SI 逐 seed 明细表）；Fig 3(a,b) 加误差棒、(c) 因子化增量图加逐 seed 散点（改 `build_academic_paper_figures.py`）；摘要与结论的数字改写为 `0.224 ± 0.00x` 形态；Limitations 第三条删除或改写为"已做 3 seeds，更大规模 seed 研究留作未来工作"。

**预注册叙事规则**（结果出来前先定好怎么写，防止事后拼故事）：
- 预期稳健的结论：T4<T2、T7<T4（两源、差距 0.007–0.010 eV 量级，且逐样本配对 CI 大概率不跨 0）→ 维持现有表述；
- 高风险结论：MP 上 T10 vs T7（单 seed 差 0.0012 eV）**很可能统计不可分**。若 CI 跨 0 或 seeds 方向不一致 → 把"T10 是 MP 最优"降级为"T10 与 T7 在 MP 上不可区分，但 T10 系统性降低 bias（0.025→0.014）与 RMSE"；这**反而强化**论文"跨源最稳方案是 T7"的保守主线，摘要与 7.2 节同步弱化"最优"用语。

**验收**：24 个新 `metrics.json` 齐全；聚合表落盘 `results/analysis/multiseed_summary.csv`；Table 3/Fig 3/摘要/结论全部改为 mean±std；SI 有逐 seed 表和 bootstrap 表。

**降级**：GPU 不可用 → 只交付 1.1/1.2（bootstrap CI）+ Limitations 保留声明；response letter 说明算力约束并给出 CI 证据。

### 3.2 意见 2：M3GNet 基线异常 【Major，P2】

**意见本质**：T3(M3GNet) MP MAE 0.286 明显差于 T2(ALIGNN) 0.240，审稿人怀疑基线不公平（没调参、没微调）。

**立场**：部分接受。事实层面可以澄清：(a) M3GNet 是 matgl 实现**从头训练**，不是微调；(b) 配置是 matgl 默认量级的小模型（nblocks 3 / units 128），无任务级调参；(c) M3GNet 的设计目标是通用原子间势/多体几何（论文标题即 universal interatomic potential），带隙单保真回归并非其主打任务，公开基准上 ALIGNN 在 gap 类任务强于多数消息传递模型是普遍格局。但"没调参"是事实，需要补一个小规模公平性检查。

**行动**：

2.1 【C，立即】论文 4.2/5 节披露 T3 完整训练配置（从头训练、超参表、与 T1/T2 的对照——统一进新的"训练配置总表"，与意见 10 合并交付）。
2.2 【C+WebSearch，立即】文献对照段：核对并引用 Matbench `matbench_mp_gap` 榜单（ALIGNN ≈0.186 eV 等，执行时 WebSearch 核实确切数值与引用格式）与 JARVIS leaderboard 上 gap 任务的相对排序；说明 M3GNet 原文没有以带隙回归为主要 benchmark。写进 Discussion 新段落"On the M3GNet baseline"。
2.3 【B，P2】小规模调参检查（MP 标签，单 seed，3 runs ≈ 6–9 h）：
   - M3GNet-A：epochs 50→100（检查欠拟合，T3 best epoch=41/43 说明未必欠拟合，但需实证）；
   - M3GNet-B：lr 1e-3→5e-4；
   - M3GNet-C：units 128→256（容量）。
   取 val 最优者为 "M3GNet (tuned)" 报告：若提升 <0.01 eV，写"经有限调参仍落后，排除明显欠调参假象"；若显著提升，主表加 tuned 行并如实更新相对结论（预注册：无论哪种结果都不影响 VN/Huber 主线，因为主线对照是 T2→T4→T7）。
2.4 【B，P2，可选】预训练初始化对照行 "M3GNet (pretrained-init)"：加载本地 `.vendor/matgl/pretrained_models/M3GNet-MP-2018.6.1-Eform` 主干、换 gap 读出头微调。中等代码量，注意在论文里标注它与"同数据从头训练"的对照原则不同，单列一行。仅在 P0/P1 提前完成时做。

**验收**：论文含 M3GNet 配置披露 + 调参检查结果 + 文献对照三件事；response letter 直接回答"从头训练，配置如表 X，有限调参后结论不变（或如实更新）"。

### 3.3 意见 3：过平滑诊断样本量 【Major，P0（推理级）】

**意见本质**：7.5 节机制证据只用了 240 个分层样本，说服力不足；要求全测试集（8,455）重跑或详细论证抽样代表性+误差棒。

**立场**：完全接受。240 是当时远程登录节点纯 CPU 的算力妥协；诊断本身是推理级任务，`scripts/analyze_oversmoothing_hooks.py` 的 CLI 已支持任意样本量，在 GPU 上全量重跑成本极低。

**行动**：

3.1 【A，立即】**先给现有数据加误差棒**（新脚本 `scripts/analyze_oversmoothing_ci.py`）：对已有 `oversmoothing_sample_metrics.csv`（逐样本逐层）按桶做 bootstrap（B=2,000），给 Table 4 的每个 Δcosine/Δvariance/ΔDirichlet 配 95% CI。这一步不依赖任何远程资源，且脚本此后直接复用于全量数据。
3.2 【B，P0】远程全量重跑（GPU 短作业或计算节点 CPU 作业均可）：
```bash
python scripts/analyze_oversmoothing_hooks.py \
  --results-root /data/home/tg100025/gnn_work/results/mp_hzw_relaxed_clean_equal_atoms_mp_t \
  --output-dir   /data/home/tg100025/gnn_work/results/analysis/oversmoothing_hooks_mp_full \
  --experiments T2 T4 T7 T10 --max-samples 8455 --batch-size 32 --device cuda
# hzw 同理；输出同步回本地 results/analysis/oversmoothing_hooks_{mp,hzw}_full/
```
   预算：GPU 上 8 个模型-源组合合计 ≈ 2 h 以内；CPU 计算节点约 1–2 天（备选）。注意脚本会物化一份全测试 manifest（≈120 MB）与 LMDB 缓存，磁盘预留 ≥5 GB。
3.3 【B，P1，联动意见 1】多种子 checkpoint 就绪后，对 seed-1042/2042 的 T2/T4/T7/T10 重复同一诊断（纯推理，GPU ≈ 4 h），把 7.5 节的机制结论也升级成"跨 3 seeds 一致"——同时回应 Limitations 第五条。
3.4 【C】论文落点：7.5 节与 Table 4 全部换成全测试集数值（±CI）；SI 保留 240 样本旧版并注明两版一致性（如方向一致，这是抽样代表性的直接证据）；SI 增补抽样协议描述（轮转分桶、桶配额 41/41/40/40/40/38、id 清单文件路径）。Fig 4(c,d) 与 SI 逐层图重生成。

**预注册叙事规则**：若全量结果保持"全样本混合、>80 原子子集方向一致"→ 维持 size-conditional 表述；若 >80 原子信号在全量下减弱 → 如实降级为"仅 T4 稳定，T7 部分"或删除机制小节、只保留描述性观察。现有措辞本就保守，风险可控。

**验收**：`oversmoothing_hooks_{mp,hzw}_full/` 落盘；Table 4 为全量+CI；论文不再出现"240 stratified samples"作为主证据（只作为 SI 历史版本）。

### 3.4 意见 4：全局依赖性质的验证 【Major，P2，唯一强依赖 GPU 的新实验】

**意见本质**：带隙偏局部化学，拿它论证"全局虚拟节点"说服力弱；建议加形成能或介电常数等全局依赖更强的目标。

**立场**：接受方向，但要向审稿人讲清两个现实约束并按性价比分层执行：
- **约束一：标签不存在**。核实过 MP 抓取缓存与 HZW 原始下载的全部字段——项目从未抓取过形成能或介电，需新调 MP API；HZW 侧只有带隙/CBM/VBM，**因此第二性质实验只能是 MP 单标签的"架构泛化"验证，不具备跨源维度**，这一点论文和 response letter 都要明说。
- **约束二：物理定位**。形成能其实也偏组成/局部；真正强全局的（介电、声子）在 MP 上样本量小（数千）。所以叙事上以"第二性质泛化性"为主张，"全局性"引用文献支撑（Okabe et al. 2024 VGNN 已证明虚拟节点在声子全谱预测上的价值，PDF 在项目根目录）。

**行动**：

4.1 【A，P2 前置】扩展抓取：改 `fetch_mp_hzw_summary_cache.py` 的 `DEFAULT_FIELDS`（:17）追加 `formation_energy_per_atom,energy_above_hull`，对 84,550 个 `source_id` 批量抓取，写**新缓存文件**（不覆盖 588 MB 旧缓存）。依赖 MP API key 可用（第 13 节确认项）。预计 ≤1 h。
4.2 【A，P2 前置】数据集导出：复制现有 relaxed manifest、以 `formation_energy_per_atom` 替换标签、**split 列原样保留**（同一 67,640/8,455/8,455 划分，保证与带隙实验同构可比），产出 `data/mp_hzw_relaxed_clean_equal_atoms_mp_eform/`。实现上优先给 loader 加 `label_key` 配置（小改动）；metadata 写明标签语义。metal-F1 对该任务无意义，报告时跳过。
4.3 【B，P2】训练：{T2, T4, T7} × eform × seed 42 = 3 runs ≈ 12.3 h（T10 可选 +4 h）。检验问题明确写成两条：VN 增益（T4−T2）与 Huber 增益（T7−T4）是否在第二性质上重现。
4.4 【C】论文落点：Results 新小节 "Generalization to a second property (MP formation energy)" + Limitations 相应收缩；Related Work/Discussion 引 VGNN 声子文献补"全局性质"论据。
4.5 【B，机动，可选加强】若 P0/P1 提前完成：Matbench `dielectric`（n≈4,764，log 目标）上跑 T2/T4/T7 小实验（每 run <1 h）。项目早期就是 matbench 起家，执行前先核查 `crystal_gnn/` 是否留有 matbench 加载路径；若无现成管道则放弃，不为此新建管道。

**预注册叙事规则**：若 VN/Huber 增益在形成能上重现 → 贡献叙事升级"两种性质上一致"；若不重现 → 如实报告，并把论文定位收紧为"带隙任务上的稳健性与标签一致性研究"（依旧成立，因为论文主贡献本来就是审计+鲁棒损失）。两种结果都可发表，不许挑数据。

**验收**：新数据集 metadata + 3–4 个 metrics.json + 论文新小节；response letter 说明 HZW 无对应标签导致单源设计。

**降级**：GPU 或 API key 不可用 → 不做实验；Discussion 引 VGNN + Matbench 文献论证方向，Limitations 保留原第一条，response letter 承认为 future work（最弱回应，尽量避免）。

### 3.5 意见 5：broadcast decay 分化的机制证据 【Major，P1，零成本分析为主】

**意见本质**：7.2 节"HZW 需要持续全局校准"目前是猜测；要求基于数据的直接证据（如挑出 HZW 上因 decay 变差的晶体做特征分析）。

**立场**：完全接受，而且这是性价比最高的一条——五项分析里四项零成本，数据全在本地。

**行动**：

5.1 【A，立即】**训练动态证据**（新脚本 `scripts/build_decay_dynamics_figure.py`）：读 T7/T10 双源 `history.json`，画 val-MAE 曲线并阴影标注 decay 窗口（epoch 10→50，β 1.0→0.25），叠加逐 epoch T10−T7 差值曲线。检验点：HZW 的 T10 劣化是否恰在 decay 深水区（β 变小的后段）才出现、MP 是否相反。产出 SI 新图。
5.2 【A，立即】**交叉标签评估**（新脚本 `scripts/analyze_cross_label_eval.py`）：同一批测试结构下，把 MP 训练的 T7/T10 预测对 HZW 标签重算指标、反向亦然，得到 2×2（训练标签 × 评估标签）矩阵。由于结构与 split 完全相同，此实验零成本且能分解"decay 的伤害来自训练信号还是评估标签"：若 MP 训练的 T10 在 HZW 标签下也劣于 MP 训练的 T7 → 标签评估效应；若只在 HZW 训练时劣化 → 训练动态效应。这是目前论文完全没有利用的双标签设计红利。
5.3 【A，立即】**受损群体画像**（新脚本 `scripts/analyze_decay_cohorts.py` + 一次性 `scripts/extract_hzw_structmeta.py`）：
   - 先从 648 MB HZW 原始 JSONL 流式抽出 `hc_id → (space_group_number, crystal_system)` 查找表（一次性，~分钟级）；
   - 基于 `global_decay_transfer_records.csv`，取 HZW 上 ΔAE(T10−T7) 最差前 10% 与最好前 10% 两个群体，对比：元素族构成（是否含过渡金属/卤素/氧族等）、晶系/空间群分布、gap 区间 × 原子数桶交叉表、|MP−HZW| 差值分布、CBM/VBM 缺失率；比例差异配 bootstrap CI。
   - 已有桶级结论（劣化集中在 41–160+ 原子、非零 gap ≥0.5 eV、标签差 ≥0.1 eV 的样本）是起点，本分析把它细化到化学/结构维度，产出"哪类晶体依赖持续全局校准"的可陈述画像。
5.4 【A/B 之间，P1 分析】**broadcast 通道量化**（改 `ALIGNNVirtualNodeRegressor`/`ALIGNNDecayVirtualNodeRegressor` 加可选诊断缓冲，或新脚本 `scripts/analyze_broadcast_norms.py`）：对 T7 与 T10 最终 checkpoint，逐层记录 ‖g‖₂ 与 ‖g‖₂/‖h‖₂ 均值（240 样本 mini 集足够，登录节点 CPU 可跑）。检验点：HZW-T7 的全局通道相对幅度是否高于 MP-T7（支撑"HZW 更依赖全局校准"），T10 是否如设计把该通道压低。需要远程 checkpoint，推理级成本。
5.5 【B，P1，与意见 10 共享】**剂量-响应实验**：`broadcast_min_scale ∈ {0.5, 0.75}` × 双源 = 4 runs ≈ 16.4 h（0.25=T10、1.0≈T7 已有）。预注册预期：HZW 的 MAE 随 min_scale 单调恢复 → decay 强度与伤害的因果证据；MP 出现最优中间值或平坦。
5.6 【C】7.2 节重写：从"one plausible interpretation"改为按证据罗列——训练动态（5.1）、交叉评估分解（5.2）、群体画像（5.3）、通道幅度（5.4）、剂量响应（5.5），推测性语句仅保留在综合小结。SI 增补对应表图。

**验收**：五项产物落盘 `results/analysis/decay_mechanism/`；7.2 节每个主张后面都有编号证据；response letter 逐点映射。

### 3.6 意见 6：strict-clean 子集消融 【Major，P1，大半资产已存在】

**意见本质**：relaxed 口径保留了 same-class >1 eV 的标签分歧，架构增益可能被噪声掩盖；要求在"严格一致"的子集上对比 T2/T4/T7。

**立场**：接受，且项目已有大半资产——S 系列就是 strict-clean（`mp_is_metal==hzw_is_metal`、|Δgap|≤0.5 eV、体积差≤8%）训练的 ALIGNN(S2)/单 VN(S4)，只缺 Huber 版本。用"测试子集重评（零成本）+ strict 训练补跑（1–2 runs）"双层回应。

**行动**：

6.1 【A，立即】**strict-consistent 测试子集重评**（新脚本 `scripts/analyze_strict_subset_metrics.py`）：用审计主表按 S 系列同款阈值给 8,455 个测试样本打"干净/噪声"标记，对 T2/T4/T7/T10（依赖 1.1 同步 T2 预测）在**干净子集**与**噪声补集**分别重算 MAE/RMSE，双标签源。回答"同一模型在干净测试样本上的架构增益是否更大"。注意在论文里写清方法学定位：训练仍在 relaxed 上，本项测"评估口径"效应，与 6.2 的"训练口径"效应互补。
6.2 【B，P1】**strict 训练补跑**：在 strict-clean equal-atoms MP 数据集上补 1 run "S-Huber"（variant=virtualnode + huber β0.5，配置克隆 S4 改 loss）≈ 4 h；连同远程已有 S2/S4 组成审稿人要的 T2/T4/T7 三元组的 strict 版。可选 +1 run S-decay。前置：从远程同步 S 系列 `comparison_summary.csv` 与 S2/S4 的 metrics/predictions。
6.3 【B，机动，可选】strict-clean 的 HZW 标签版数据集目前不存在；若时间富余可导出并跑 3 runs 补全跨源 strict 对照，否则在论文注明 strict 对照以 MP 标签为准（S 系列历史即如此）。
6.4 【C】论文落点：Results 加小节或 SI 表 "Effect of label-noise filtering"：两个维度（评估子集 × 训练口径）下的 T2→T4→T7 增益对照；Discussion 讨论增益随口径的变化方向。写作警示：strict 与 relaxed 的样本量（≈81.5k vs 84.6k）和测试集（8,154 vs 8,455）都不同，**只比较口径内的相对增益，不跨口径比绝对 MAE**（这是 06-11 计划文档已立的规矩，保持）。

**预注册叙事规则**：若干净子集上 T4−T2 增益更大 → 支持"噪声部分掩盖架构增益"，强化 VN 主张；若增益反而缩小 → 说明 VN 增益部分来自对噪声样本的鲁棒拟合，与 Huber 的作用有重叠——同样是有价值的诚实结论，按实际写。

**验收**：子集重评表 + strict-Huber metrics 落盘；论文有明确的"噪声是否掩盖架构增益"答案。

### 3.7 意见 7：HZW 全称与系列代号 【Minor 但工作量大，工作流 C】

**意见本质**：HZW 全文未定义未引用；T/E/S 系列是内部实验笔记黑话。

**立场**：完全接受。HZW=鸿之微（Hongzhiwei Technology），数据经其云 API（`api-sxs.cloud.hzwtech.com` 的 `crystal_band` 端点）获取，100,000 条含 MP 交叉引用（`source_id`）的带隙记录，记录 ID 为 `hc_id`。**正式披露口径（公司全称、DFT 软件/泛函、引用条目、许可）需师兄/导师确认**——列为外部依赖（第 13 节）。

**行动**：

7.1 【外部依赖，第一天发起】向师兄/导师确认：数据集正式名称与出处描述、计算方法（DS-PAW？何种泛函？）、可否公开披露、是否有可引用文献或产品文档、数据使用许可表述。得到答复前，论文先用占位模板：
   > "the HZW dataset, a proprietary band-gap dataset provided by Hongzhiwei Technology (Shanghai) and computed with [DFT code/functional, to be confirmed]; each record cross-references a Materials Project entry"
   并在 Data availability 写 "used with permission; access subject to the provider's terms"。
7.2 【C】**系列重命名，推荐方案 A（描述名为主、代号为索引）**：
   - 表格保留 ID 列（审稿人并未要求消灭代号，只要求可读），正文与图注一律用描述名；首次出现处一句话定义每个系列（"models are indexed T1–T10; the T series denotes the final matched-benchmark stage"）。
   - 全文替换映射表：

| 代号 | 正文用名 |
| --- | --- |
| T1/T2/T3 | CGCNN / ALIGNN / M3GNet（直接用模型名） |
| T4 | ALIGNN+VN |
| T7 | ALIGNN+VN+Huber |
| T10 | ALIGNN+VN+Huber+decay |
| T6/T8/T9 | top-k 变体，仅 SI 出现，SI 开头统一定义 |
| E 系列 | "exploratory HZW-stage models"，仅 SI；正文 6.1 的 "E13" 提法改为描述性表述或删除 |
| S 系列 | "strict-filter stage models"，正文出现处（6.1、7.4）改写 |
   - 不选择全局改号（如 B1/V1）的原因：SI、结果目录、脚本、图内标签全部要连锁改，出错面大且无增益。
7.3 【C】执行细节：`gnn_bandgap_research_paper_draft_en.tex` 与 SI 全文替换；**fig3/fig4 图内标签是 `build_academic_paper_figures.py` 写死的 T 代号，需加标签映射后重生成**；摘要与 Keywords 里的 "HZW""global token" 同步处理；历史遗留检查——早期文档中 T6/T7 曾指别的模型（已废弃 val-routed/size-gated 方案），确保重命名不复活旧义。

**验收**：全文（含图内、SI、图注、关键词）无未定义缩写；grep `HZW` 首次出现处有全称+引用/脚注；grep `T-series|E-series|S-series` 每处都有定义或已替换。

### 3.8 意见 8：金属/非金属 F1 阈值 【Minor，工作流 C + 零成本分析】

**意见本质**：Table 3 的 F1 没说金属判定阈值。

**立场**：完全接受。已核实实现（`crystal_gnn/utils.py:62-91`）：**真值金属 = band gap ≤ 1e-8 eV（即数据库报告零隙）；预测金属 = 预测值 ≤ 0.1 eV**。非对称的原因：回归输出几乎不会精确为 0，需要容差。另注意项目里共有**三套金属相关阈值**并存，必须一次性讲清楚（也是第 7 节问题 A 的根源）：

| 用途 | 定义 | 出处 |
| --- | --- | --- |
| 审计冲突统计 | metal-like：`is_metal` 或 gap≤0.05 eV；nonmetal：gap≥0.10 eV | `docs/plans/2026-05-24-mp-hzw-structure-first-audit.md` |
| T 系列数据过滤 | 零隙 = gap ≤ 1e-6 eV，零/非零不一致即剔除 | `run_t_relaxed_matched_experiments.py:374-395` |
| F1 评估 | 真值 ≤1e-8；预测 ≤0.1 eV | `crystal_gnn/utils.py:62-91` |

**行动**：

8.1 【C】5 节（Experimental Setup）加一段 F1 协议定义；3.2 节加"阈值清单"表或脚注（上表内容），并顺带说明：由于零/非零冲突已在过滤时剔除，最终数据集里 MP 与 HZW 的金属真值标签完全一致——这解释了 Table 3 中 T1 两源 F1 几乎相同的现象。
8.2 【A，立即】阈值敏感性小表（新脚本 `scripts/analyze_metal_f1_thresholds.py`）：预测阈值 ∈ {0.01, 0.05, 0.1} 下重算 T2/T4/T7/T10 双源 F1（依赖 1.1 的 T2 预测同步），SI 一张表，证明模型排序不依赖阈值选择。

**验收**：论文任何 F1 出现处都可回溯到明确定义；SI 有敏感性表。

### 3.9 意见 9：过平滑与虚拟节点经典引用 【Minor，工作流 C】

**意见本质**：引言/相关工作缺图机器学习侧的经典文献（over-smoothing、virtual node/global token）。

**立场**：完全接受。补充清单如下（执行时逐条 WebSearch 核对卷/页/年份再录入 `thebibliography`；标 ★ 为必须，其余按篇幅取舍）：

| 主题 | 条目 | 用途/插入位置 |
| --- | --- | --- |
| ★ over-smoothing | Li, Han, Wu. *Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning*. AAAI 2018 | Intro 第 3 段 "oversmoothing" 处；2.2 |
| ★ over-smoothing | Oono, Suzuki. *Graph Neural Networks Exponentially Lose Expressive Power for Node Classification*. ICLR 2020 | 同上 |
| ★ 度量依据 | Cai, Wang. *A Note on Over-Smoothing for Graph Neural Networks*. arXiv:2006.13318 | 7.5 节 Dirichlet energy 指标的出处（我们用了该度量却没引，必须补） |
| 可选 | Rusch, Bronstein, Mishra. *A Survey on Oversmoothing in GNNs*. 2023 | 2.2 综述性引用 |
| ★ virtual node | Gilmer et al. *Neural Message Passing for Quantum Chemistry*. ICML 2017 | 2.2 虚拟节点最早实践（master node） |
| ★ global token | Ying et al. *Do Transformers Really Perform Badly for Graph Representation?*（Graphormer）. NeurIPS 2021 | 2.2 全局 token/[VNode] |
| ★ 理论 | Cai et al. *On the Connection Between MPNN and Graph Transformer*. ICML 2023 | 2.2："VN+MPNN 可近似全局注意力"，直接支撑本文"VN 是注意力的廉价替代"论点 |
| ★ 材料+VN | Okabe et al. *Virtual node graph neural network for full phonon prediction*. Nature Computational Science 2024 | 2.2 与 4.4 节（意见 4 的文献论据；PDF 已在项目根目录） |
| ★ 晶体长程 | Yan et al. *Periodic Graph Transformers for Crystal Material Property Prediction*（Matformer）. NeurIPS 2022 | 2.1/2.2 晶体全局建模现状 |
| 可选 | ComFormer. ICLR 2024（PDF 在库） / Lin et al. PotNet. ICML 2023 | 同上 |
| 可选 | Schütt et al. SchNet. J. Chem. Phys. 2018；Chen et al. MEGNet. Chem. Mater. 2019 | 2.1 晶体 GNN 谱系补全 |
| 可选 | Perdew. *Density functional theory and the band gap problem*. IJQC 1985；或 Borlido et al. 泛函带隙基准（npj Comput. Mater. 2020） | 3 节/7.2：为"两个 DFT 数据库带隙系统性不一致"提供物理背景 |
| 待定 | HZW/DS-PAW 数据来源文献 | 依赖 7.1 的确认结果 |

**行动**：9.1【C】按上表补 `thebibliography` 并在正文挂引；9.2【C】逐句检查"引用-主张匹配"：Intro 第 3 段（oversmoothing 动机句）、2.2 全段、7.5 首段目前**零引用**，是审稿人点名的位置。

**验收**：新增 8–12 条文献；上述三处每个技术主张都有出处。

### 3.10 意见 10：超参选择说明与敏感性 【Minor，工作流 C + P1】

**意见本质**：Huber δ、top-k 比例、decay 时间表像凭空给定；要求说明来源+最简敏感性分析。

**立场**：完全接受。事实：这些值是设计时的经验一次性选定（δ=0.5、fraction=0.2 clip[4,32]、warmup 10+10、decay 10→50→0.25），**没有做过任何扫描**（全仓核实）。诚实披露 + 补两组敏感性（decay 组与意见 5 共享同一批 GPU）。

**行动**：

10.1 【C】4.6/4.7/5 节各加"选择依据"句：明确为先验经验值、未做逐变体调参、所有变体共享同一训练预算（这本身是公平性设计）；同时修正 Huber 公式表述（见第 7 节问题 C：实现是 `smooth_l1(beta=0.5)`，即标准 Huber 除以 δ 的缩放形式，论文公式应与实现一致或注明缩放）。
10.2 【B，P1】敏感性实验 8 runs ≈ 33 h：
   - Huber δ ∈ {0.25, 1.0} × 双源（δ=0.5 已有）= 4 runs；
   - decay `min_scale ∈ {0.5, 0.75}` × 双源 = 4 runs（**与 5.5 是同一批，双重记账只跑一次**）。
   - top-k 参数不扫，理由写明：T8 已证明 top-k 无独立稳定增益、非论文主张（7.3 节现有结论），扫描它不改变任何结论。
10.3 【C】SI 新表：敏感性结果 + **完整训练配置总表**（T1/T2/T3 及各变体的优化器、lr、scheduler、batch、epochs、AMP、seed、硬件 RTX 4090、关键软件版本）——这张表同时服务意见 2 与第 7 节问题 G（现在 5 节连优化器都没写，正式投稿必补）。

**验收**：论文无"凭空常数"；SI 有敏感性表与配置总表；δ 与 min_scale 的鲁棒区间有一句话结论。

---

## 4. GPU 实验总排期与实现细节

### 4.1 批次表

| 批次 | 内容 | runs | GPU·h（估） | 依赖 |
| --- | --- | ---: | ---: | --- |
| **P0-a** | 多种子：{T1,T2,T3,T4,T7,T10} × {mp,hzw} × {1042,2042} | 24 | ≈ 71.5 | seed CLI 改造 + 冒烟验证 |
| **P0-b** | 过平滑全测试集诊断（8 模型-源组合，推理） | — | ≈ 2（GPU） | 远程 checkpoint 在位 |
| **P1-a** | decay min_scale {0.5,0.75} × 双源（意见 5+10 共享） | 4 | ≈ 16.4 | 无 |
| **P1-b** | Huber δ {0.25,1.0} × 双源 | 4 | ≈ 16.4 | 无 |
| **P1-c** | strict-clean MP 补跑 S-Huber（可选 +S-decay） | 1–2 | ≈ 4–8 | strict 数据集远程在位 |
| **P1-d** | 多种子 checkpoint 的过平滑诊断（推理） | — | ≈ 4 | P0-a |
| **P2-a** | 形成能：{T2,T4,T7}(+T10) × mp-eform | 3–4 | ≈ 12.3–16.4 | MP API 抓取 + 导出 |
| **P2-b** | M3GNet 调参 {epochs100, lr5e-4, units256} × MP | 3 | ≈ 6–9 | 无 |
| 合计 | | 39–41 | **≈ 133–144** | 单卡连续 ≈ 5.5–6 天 |

三档预算方案：**只做 P0** ≈ 74 h（意见 1/3 达标，其余靠零成本分析与写作）；**P0+P1** ≈ 115 h（再拿下 5/6/10）；**全量** ≈ 140 h。建议至少 P0+P1。

### 4.2 seed 注入的代码改动清单（P0-a 前置，半天）

1. `faithful_gnn/common.py:16-20`：`seed_everything` 补 `numpy` 与 `dgl.random.seed`；
2. `run_t_relaxed_matched_experiments.py`：加 `--seed`（默认 42，注入 `config["seed"]`）与结果目录后缀参数；核对无 CLI 冲突后本地跑 `tests/test_t_relaxed_runner_config.py` 回归；
3. 新增 `scripts/slurm_t_multiseed_queue.sh`（克隆现有队列脚本，循环 seed × label_source，`RESULTS_DIR` 带 `_seed{N}` 后缀）；
4. 冒烟验证（远程，≈20 min）：seed=1042、`--epochs 1`、仅 T4/MP → 断言 test `sample_id` 集合与 seed-42 完全一致（划分不变性）+ 权重确实不同（train loss 首 epoch 不同即可）。

### 4.3 目录与防覆盖约定

- 新训练结果：`results/mp_hzw_relaxed_clean_equal_atoms_{mp,hzw}_t_seed{1042,2042}/`、`.../sensitivity_{huber_delta,decay_minscale}/`、`.../mp_hzw_relaxed_clean_equal_atoms_mp_eform_t/`；
- 新分析产物统一进 `results/analysis/`：`multiseed_summary.csv`、`paired_bootstrap.csv`、`cross_label_eval.csv`、`decay_mechanism/`、`strict_subset_metrics.csv`、`metal_f1_thresholds.csv`、`oversmoothing_hooks_{mp,hzw}_full/`；
- **绝不覆盖现有 seed-42 结果目录**；论文旧数值在多种子版定稿前保持可复现。

### 4.4 监控与安全

- 沿用 `squeue -a` / `sinfo -p tx4090` / `scontrol show nodes` 三连检查（集群 `PrivateData=jobs,usage`，见 05-30 计划）；
- 第一天先提交 `--wrap=hostname` 最小 GPU 探测作业验证 QOS；
- 任何脚本/文档不得写入 pem 私钥内容与 API key（沿用 CURRENT_SESSION.md 安全规则）。

---

## 5. 零成本本地分析任务清单（工作流 A，全部可立即开工）

| # | 脚本（新建） | 输入 | 输出 | 服务意见 |
| --- | --- | --- | --- | --- |
| A1 | 远程同步（命令，非脚本） | 远程 T1/T2/T3/T6/T8/T9 predictions/metrics/history；S 系列汇总与 S2/S4 明细 | 本地 results/ 补齐 | 1/5/6/8 前置 |
| A2 | `scripts/analyze_paired_bootstrap.py` | 各模型 `predictions.csv` | `results/analysis/paired_bootstrap.csv` + SI 表 | 1 |
| A3 | `scripts/analyze_oversmoothing_ci.py` | `oversmoothing_sample_metrics.csv`（现有 240 版与后续全量版通用） | Δ 指标 ±95% CI 表 | 3 |
| A4 | `scripts/build_decay_dynamics_figure.py` | T7/T10 双源 `history.json` | SI 训练动态图 | 5 |
| A5 | `scripts/analyze_cross_label_eval.py` | T4/T7/T10（+T2）双源 `predictions.csv` | 2×2 交叉评估矩阵 CSV+表 | 5 |
| A6 | `scripts/extract_hzw_structmeta.py` + `scripts/analyze_decay_cohorts.py` | HZW 原始 JSONL、`global_decay_transfer_records.csv`、审计主表 | 受损/受益群体画像表 | 5 |
| A7 | `scripts/analyze_strict_subset_metrics.py` | 审计主表 + 各模型 `predictions.csv` | 干净子集/噪声补集指标表 | 6 |
| A8 | `scripts/analyze_metal_f1_thresholds.py` | 各模型 `predictions.csv` | F1 阈值敏感性表 | 8 |

约定：所有分析脚本固定随机种子、纯读入不改原始数据、输出带生成时间与输入文件清单的 header（可追溯性）；每个脚本配一个最小 pytest（合成小数据验证统计逻辑，尤其 bootstrap 的配对方式）。

---

## 6. 写作修订逐节清单（工作流 C）

对 `gnn_bandgap_research_paper_draft_en.tex`：

| 位置 | 修改 |
| --- | --- |
| 标题/作者区 | 补作者单位与邮箱；`\date` 按投稿要求处理 |
| 摘要 | HZW 首次出现给全称；数字改 mean±std、精度降到 3 位有效数字（`0.2237±0.00x eV` 或 meV）；"best model" 措辞按预注册规则待 P0 结果定稿 |
| Keywords | 定义/替换 HZW；检查术语与正文一致 |
| 1 Introduction | 第 3 段 oversmoothing/virtual node 处挂新引用（Li/Oono/Gilmer/Graphormer/VGNN）；贡献段第三条补"多种子与敏感性验证" |
| 2 Related Work | 2.2 扩写为有引用的虚拟节点/全局 token 综述（Gilmer→Graphormer→Cai2023→材料界 VGNN/Matformer）；2.1 可选补 SchNet/MEGNet；2.3 可选补 DFT 带隙误差文献 |
| 3 Data | HZW 数据来源正式描述（依赖外部确认）；**重构 Table 1 为两面板**（审计冲突统计 @0.05/0.10/1eV 阈值 vs 实际过滤 @1e-6 + equal-atoms，账目各自闭合，见第 7 节问题 A）；新增阈值清单表/脚注；`fig2_data_audit` 同步重生成 |
| 4 Methods | 式(2) 改为与实现一致（broadcast 恒等加法、VN 残差 MLP 更新、读出 concat+fusion，见问题 B）；4.6 Huber 公式与 smooth-L1(β) 实现对齐并写明 δ=0.5 选择依据；4.7 补 decay/top-k 参数依据句；图 1 caption 的 α_l 与正文 β_t 统一符号 |
| 5 Experimental Setup | 大扩写：训练配置总表（优化器/lr/scheduler/batch/epochs/AMP/硬件/软件版本）、seed 协议（3 seeds、split 固定、seed 影响范围）、F1 定义、统计方法（bootstrap 协议）说明 |
| 6 Results | Table 3 → mean±std + 显著性标注；Fig 3 加误差棒与逐 seed 散点；新增 strict-clean 对照小节（意见 6）与形成能小节（意见 4）；系列代号按 7.2 映射替换 |
| 7 Discussion | 7.2 按五类证据重写（意见 5）；7.3 补 top-k 不扫参的理由；新增 "On the M3GNet baseline" 段（意见 2）；7.5 换全测试集数值+CI（意见 3） |
| 8 Limitations | 删/改已解决条目（多种子、240 样本、单一性质），保留真实剩余限制（HZW 单源专有、形成能非强全局性质、seed 数仍有限） |
| 9 Conclusion | 数字与措辞随主表联动更新 |
| Data availability | 实质化：承诺代码仓库 + MP 标签子集数据发布（Zenodo 等），HZW 标签按许可说明（问题 F/H） |
| References | 第 3.9 节清单落地 |

对 SI（`gnn_bandgap_research_supplementary_en.tex`）：逐 seed 明细表、bootstrap 表、敏感性两表、训练配置总表、过平滑全量表+240 对照、decay 机制四件套（动态图/交叉矩阵/画像表/剂量曲线）、strict 对照表、F1 阈值表、E/S 系列术语定义段。

新交付物：`docs/reviewer_response_letter_draft_en.md`——英文逐条回应骨架（意见→改动→论文位置→新证据），附中文对照要点，供师兄审阅。

图表重生成清单：`fig2_data_audit`（数字修正）、`fig3_main_results`（误差棒+改名）、`fig4_mechanism_diagnostics`（全量过平滑+改名）、SI 各图；全部经 `scripts/build_academic_paper_figures.py` 等脚本产出，禁止手改 PDF。

---

## 7. 自查发现的额外问题（意见之外，一并修）

落实师兄"用 AI 生成的自己也读一遍"的要求，本轮通读论文+核对代码/数据后发现以下问题，全部纳入修订：

| # | 问题 | 事实依据 | 处理 |
| --- | --- | --- | --- |
| A | **Table 1 冲突计数口径不自洽**：表中 3,142/3,046/85,478 来自审计统计口径（0.05/0.10/1 eV 阈值，且 85,478 已剔除 >1 eV 行），而"Final equal-atom set 84,550"来自 1e-6 零/非零过滤（7,984 条，为 2,935+5,049）——两套账混排，读者按行相减对不上（85,478−84,550=928≠252） | 第 2.3 节账目；`run_t_relaxed_matched_experiments.py:374-395`；05-30/06-11 计划 | Table 1 拆两面板，正文 3.2 改写；fig2 重生成 |
| B | **式(2) 与实现不符**：论文写 `h + φ(g)`，实现是恒等加法 `h + vn`（无 φ）；φ(MLP) 实际用在 VN 自身残差更新 `vn ← vn + MLP(mean(h))` | `alignn_variants.py:277-326` | 修公式与 4.3 描述；给出完整交换方程 |
| C | **Huber 公式与实现的缩放差异**：实现为 `F.smooth_l1_loss(beta=0.5)` = Huber_δ/δ；论文式(4)是标准 Huber。极值点相同但整体尺度差 1/δ 倍（影响与其它 loss 的相对权重表述） | `alignn_runner.py:489-495` | 4.6 用 smooth-L1 形式或注明缩放 |
| D | 舍入不一致：T7 HZW MAE 论文/SI 表间出现 0.194002 与 0.194001 | SI Table S4 vs 主表 | 统一以 `metrics.json` 原值+统一舍入规则 |
| E | MAE 报 6 位小数（0.223686 eV），远超物理与统计意义 | 全文 | 统一 3 位有效数字或 meV；多种子后自然带 ±std |
| F | Data availability 只说 "results/ 目录"，对外部读者无意义 | 论文 339-341 行 | 见第 6 节 Data availability 项 |
| G | 5 节缺优化器/lr/batch/硬件——正式论文必需 | 论文 200-204 行 | 训练配置总表（与意见 10 合并） |
| H | 无代码/数据发布声明与许可信息 | — | 与 F 合并处理；HZW 依赖外部确认 |
| I | Fig 3(c) "positive=improvement" 与 SI decay 表 "positive=T10 worse" 符号约定相反 | 论文 238 行 vs SI S4 | 统一约定或图注双向明示 |
| J | Fig 1 caption 用 α_l（层下标），正文 decay 公式用 β_t（epoch 下标）——decay 是 epoch 调度，α_l 双重错误 | 论文 124 vs 196 行 | 统一为 β_t |
| K | 摘要"the best model is ALIGNN with ... decay"依赖 0.0012 eV 的单 seed 差距 | 摘要 | 按 3.1 预注册规则随 P0 结果改写 |
| L | 正文 6.1 出现 "E13" 但 E 系列语义只在 SI 定义 | 论文 229 行 | 与意见 7 重命名一并处理 |
| M | `seed_everything` 漏 numpy/dgl（复现性声明的准确性） | `common.py:16-20` | 修代码；5 节复现性描述如实 |
| N | T1 两源 metal F1 几乎相同（0.788246 vs 0.788237）可能被审稿人当疑点 | Table 3 | 8.1 的解释句顺带覆盖（金属真值两源一致） |

---

## 8. 里程碑与时间表（日历约 2.5–4 周，含排队缓冲）

| 阶段 | 内容 | 产出 |
| --- | --- | --- |
| **D0（第 1 天）** | GPU 探测作业验证 QOS；发起 HZW 披露口径确认与 MP API key 确认；A1 远程同步；seed CLI 改造+单测+冒烟 | 配额结论、资产补齐、P0 就绪 |
| D1–3 | 提交 P0-a（24 runs ≈ 3 天）与 P0-b；并行完成 A2–A8 全部零成本分析；开工工作流 C 的写作项（意见 7/8/9/10 文字部分+第 7 节问题） | 零成本分析全套 + 论文写作类修订初稿 |
| D4–6 | P0 收尾→跑聚合→更新 Table 3/Fig 3/摘要/结论；提交 P1（≈ 1.5–2 天）；7.5 节全量过平滑改写 | 主表多种子版论文 |
| D7–10 | P1 分析（decay 剂量+Huber 敏感性+strict 补跑）→ 7.2 节重写、6 节 strict 小节、SI 敏感性表；提交 P2 | 意见 5/6/10 完整回应 |
| D11–14 | P2 分析（形成能小节、M3GNet 段落）；response letter 成稿 | 意见 2/4 回应 |
| **收尾（0.5–1 天，必做）** | **作者全文通读**（师兄要求）：逐数字回溯到 CSV 来源、逐主张核对证据编号、术语一致性 grep（HZW/系列名/阈值/符号）、两文档交叉数字 diff、重编译并按 06-11 计划的 PDF 检查单验收 | 终稿 + SI + response letter |

关键路径：P0-a 的 72 GPU·h。若集群允许同时占 2 张卡（历史 association 显示上限 `gres/gpu=2`，需实测），P0 可压缩到 1.5 天。

---

## 9. 预注册叙事规则汇总（防止结果驱动的事后修辞）

| 可能结果 | 论文处理 |
| --- | --- |
| MP 上 T10−T7 不显著（大概率） | "T10 与 T7 统计不可分；decay 的价值在 bias/RMSE 而非 MAE"；主推 T7，摘要同步弱化 |
| T7−T4 或 T4−T2 有 seed 翻转（小概率） | 如实报告翻转率，把对应主张降为"平均而言"并给逐 seed 表 |
| 过平滑全量结果方向不变 | 维持 size-conditional 表述并升级为全量证据 |
| 过平滑 >80 原子信号减弱 | 降级或移除机制主张，仅保留描述性观察 |
| strict 子集架构增益变大 | "标签噪声部分掩盖架构增益"，强化 VN 主张 |
| strict 子集增益变小 | "VN 增益部分体现为对噪声样本的鲁棒性"，与 Huber 关系如实讨论 |
| 形成能上 VN/Huber 增益重现 | 贡献叙事升级为跨性质 |
| 形成能上无增益 | 论文定位收紧为带隙任务研究，Limitations 保留 |
| M3GNet 调参后仍落后 | "排除欠调参假象"段落 |
| M3GNet 调参后大幅追上 | 主表加 tuned 行并如实更新基线结论（不影响 VN/Huber 主线） |

---

## 10. 风险与降级预案

| 风险 | 概率 | 影响 | 缓解/降级 |
| --- | --- | --- | --- |
| Slurm QOS 再次拒绝 GPU 作业 | 中 | 工作流 B 全停 | D0 即探测；立刻找平台/导师协调；期间全力推进 A/C；最坏交付"CI+现有数据误差棒"版本并在 letter 说明 |
| 远程 checkpoint/predictions 已被清理 | 低 | 零成本分析受损、过平滑无法重跑 | D0 的 A1 同步即是核查；若缺失，受影响模型重训并入 P0 预算 |
| 多种子推翻 T10 MP 最优 | 中高 | 摘要/结论改写 | 已预注册（第 9 节），属于计划内结果而非事故 |
| MP API key 失效/限流 | 低 | 意见 4 延期 | 提前申请；formation energy 字段抓取量小（84,550 条 id 查询） |
| HZW 披露口径不获批 | 低中 | 意见 7 只能匿名化描述 | 备选表述："a proprietary DFT band-gap database provided by a commercial materials-simulation vendor, used with permission"；在 letter 说明保密约束 |
| 时间超支 | 中 | 全链拖长 | 优先级固定为 P0>P1>P2；P2 可整体降级为 letter 中的 planned work（不推荐但可行） |
| 多脚本产出数字与论文手抄不一致 | 中 | 可信度事故 | 所有表格数字从 CSV 程序化生成（延续 build_academic_paper_figures 模式）；收尾日做数字 diff 核对 |

---

## 11. 总验收清单

- [ ] 意见 1：Table 3/Fig 3 为 3-seed mean±std；SI 有逐 seed 表 + 配对 bootstrap 表；结论措辞符合预注册规则
- [ ] 意见 2：M3GNet 配置披露 + 调参检查 + 文献对照三件齐
- [ ] 意见 3：过平滑诊断覆盖全测试集（两源×4 模型），表格带 CI；抽样协议写入 SI
- [ ] 意见 4：形成能小节落地（或 letter 说明降级原因）；单源约束已声明
- [ ] 意见 5：7.2 节五类证据（动态/交叉评估/画像/通道幅度/剂量响应）替换猜测性解释
- [ ] 意见 6：strict 测试子集重评 + strict 训练三元组对照两个维度都有结果
- [ ] 意见 7：HZW 全称+出处+许可表述；系列代号按映射表替换；图内标签重生成
- [ ] 意见 8：F1 双阈值定义入文；三套阈值清单表；敏感性表入 SI
- [ ] 意见 9：8–12 条新文献挂到位；三处零引用段落修复
- [ ] 意见 10：超参溯源说明 + δ/min_scale 敏感性 + 训练配置总表
- [ ] 第 7 节问题 A–N 全部关闭
- [ ] response letter 英文稿完成并经师兄过目
- [ ] 作者全文通读完成：数字可回溯、术语一致、主/SI/图三方数值 diff 为零、PDF 编译无 fatal error

---

## 12. 需要师兄/导师决策或提供的信息（第一天发出）

1. **HZW 数据集**的正式名称、DFT 软件与泛函、可引用条目、公开披露与数据发布许可（意见 7 与 Data availability 的硬依赖）。
2. **算力**：tx4090 分区当前 QOS 配额；能否同时占 2 卡；预计可用总时长（决定选三档预算的哪一档）。
3. **MP API key** 是否可用（形成能抓取）。
4. **投稿目标**（期刊/会议）：决定格式（bibtex 化、字数）与意见 4 的必要深度。
5. **seed 数量**：3 个是否满足目标 venue 惯例；若需 5 seeds，P0 预算 ×2（≈143 GPU·h 仅 P0），建议先 3 后补。

---

## 执行进度（2026-07-22 更新）— GPU 批次 P0/P1/P2 全部完成

"全跑"队列（Slurm job 2055 + 补跑 2088）已全部结束，14/14 项完成，结果已入文：

- **意见 3（P0，oversmoothing 全测试集）**：8,455×2 源重跑完成。>80 原子子集 n=422：T4 与 T7 三指标 CI 两源全部不跨 0（较 240 样本 pilot 更强），T10 仅剩 HZW Dirichlet 跨 0。信号保留并加强，按预注册升级为全量证据。Table 4、7.6 节、Fig 4(c,d)、SI Table S12 已改为全量口径；240 样本 pilot 保留为代表性交叉检查。
- **意见 6（P1，strict-Huber）**：S14 = strict-clean + VN + Huber 跑完，MAE 0.1936（vs S4 的 0.2001，−0.0064），与 relaxed 口径 T4→T7 增益同量级；写入 7.1 与 SI strict 小节。
- **意见 5/10（P1，敏感性 8 runs）**：Huber β∈{0.25,1.0}、min_scale∈{0.5,0.75}×两源完成。β 三档增益全保持（跨 β ≤0.006 eV，量级同种子噪声）；HZW 上 MAE 随 decay 减弱单调恢复至 T7（剂量-响应闭环）。新增 SI Table S15；正文 4.6/4.7 两处占位句已填实。
- **意见 2（P2，M3GNet 调参 3 runs）**：ep100→0.2867、lr5e-4→0.2912、units256→0.3112，无一接近 T2 的 0.2400；"架构错配而非欠调参"写入 7.4 段落与新 SI Table S16。
- **工程**：`run_t_relaxed_matched_experiments.py` 新增通用 `--override`（测试 9/9 过）；`run_s_clean_matched_experiments.py`/S14 模板/`analyze_oversmoothing_hooks.py`（sys.path 修复）已同步远端；队列脚本 `scripts/slurm_reviewer_p1p2_queue.sh`。
- **文档状态**：正文 17 页、SI 10 页（S15/S16 新增于末尾，S1–S14 编号未动），均 0 error / 0 undefined ref；response letter 10 条意见全部 `[DONE]`（仅意见 7 的 DFT 泛函小注仍等 provider 信息，不阻塞）。

剩余唯一开口项：意见 7 的 DFT 代码/泛函 footnote（等师兄/鸿之微提供方法学句子）。
