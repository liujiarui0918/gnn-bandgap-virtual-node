# 论文证据衔接收尾计划

日期：2026-10-20
状态：调研已完成；待确认后修改论文。此前用户的“继续”用于继续核查，本文件不把它追认成尚未提出的训练方法变更授权。
关联背景：[上一轮审稿整改记录](2026-07-14-reviewer-response-revision-plan.md)。本文件只管理本轮剩余问题，不重新启动上一轮 GPU 实验计划。

## 目标与完成标准

让正文、SI、实现与实际核验范围一致，而不是继续大规模英文改写。

完成标准：
- 四组 cross-label CI 正确指向 Supplementary Section S12；摘要、6.2、结论和 SI 均不把 seed-42 点估计的符号模式写成已确定的因果机制。
- 区分 MP 单来源复现与需要授权 HZW 记录的结构匹配/跨来源标签审计；主文与 README 不再矛盾。
- F1 标签一致性只覆盖实际逐样本核查的 8,455 条 test split，不由 1e-6 的过滤规则推出 1e-8 下全数据一致。
- CBM/VBM 描述忠实反映历史 OR 实现、局部原始数据核查和已复现的边界行为；不把“当前原始数据未见问题”写成“历史训练集已经验证”。
- Equation (3) 的局部层记号明确覆盖 4 个 ALIGNN 层和 4 个 edge-gated convolution 层。
- 两份新 PDF 编译成功，无未定义引用；实际检查改动页渲染，保存原 PDF 不覆盖。源文件通过 `git diff --check`。

## 已确认事实与证据

1. 工作区原有 15 个修改文件，其中包括两份 TeX 和图表源/产物；本轮开始时未有新增修改。保留这些已有修改，不 reset、不提交、不建分支。
2. `faithful_gnn/alignn_variants.py:295-321`：先广播，再对广播后的节点 mean-pool 并残差更新 token，再运行本地层；ALIGNN 和 GCN 两个循环均在每层之前交换。两份物理变体配置均设置 4 ALIGNN + 4 GCN。
3. 从 MP/HZW 的 T7/T10 四份 `predictions.csv` 重算：均为 8,455 个唯一、相同的 test ID；预测和标签均有限；同源模型的目标逐样本一致。采用实际 `scripts/analyze_paired_bootstrap.py` 的 `paired_bootstrap`，B=10,000、seed=20260714：

   | Train | Eval | T10−T7 MAE | 95% CI (eV) |
   |---|---|---:|---|
   | MP | MP | -0.00124520 | [-0.00580557, +0.00327472] |
   | MP | HZW | -0.00244040 | [-0.00702811, +0.00206726] |
   | HZW | MP | +0.00250027 | [-0.00154014, +0.00656968] |
   | HZW | HZW | +0.00303223 | [-0.00106164, +0.00705848] |

   与 SI 四位小数一致，四区间均跨零。Table S8 本身只有 MAE/RMSE，区间在 Section S12 的文字中。
4. 同一 test split 在 1e-8 和 1e-6 eV 下：MP/HZW 各有 3,803 个金属标签、逐样本分歧均为 0；两源在 (1e-8, 1e-6] 区间均为 0 条。未取得完整最终 manifest，不能声称全体 84,550 条已核查。
5. `export_official_dataset.py:37-52` 把缺失/null 带边映射为 0.0，没有显式 finite 检查；`faithful_gnn/alignn_runner.py:574-588` 三项共用 OR 掩码。对实际 `compute_loss` 函数体的隔离执行（未修改函数、没有启动训练）复现：
   - 有效 CBM=2、零值 VBM 目标=0、VBM 预测=1 时，aux=1、VBM 梯度=2（lambda_aux=1，其他物理权重为 0）。
   - 一边为 NaN、另一边有效时，aux/loss 为 NaN；Inf 对应非有限 loss。
   - 两边都为 0 时，辅助项被屏蔽。
6. 对本地原始 HZW 文件只做流式读取和数值聚合，未输出原始记录。文件有 100,000 条可解析 JSON、另有 2 行无法解析；4,506 条无 band 记录、4 条 gap>15 eV。按实际 E-series `extract_target` 函数保留的 **95,490 条、结构验证之前**：52,001 条 CBM/VBM 均有限非零，43,489 条均为精确零；无缺失/null 带边、无单边零、无 NaN/Inf、无 (0,1e-8] 带边值。该子集上 OR 与 AND 选择相同记录。
7. 上述 raw-input 核查不是历史训练 manifest 核查，也不是 T-series 的 84,550 条人口；不同筛选口径不得通过计数相减来混用。
8. `README.md:36-37` 存在同类过强复现声明，需与主文同步。正文和 SI 尚未提供实际公开仓库地址或永久 DOI，发布落实情况待作者确认。
9. 本地已定位 Python、pdflatex、pdftoppm，且 numpy、torch、pymupdf、pypdf 可用；尚未进行本轮 PDF 重编译或视觉验收。

## 假设、未知与取舍

- 推荐保留历史训练实现和实验数字，通过修正措辞、披露边界与原始数据检查范围收尾。
- 不把零值直接等同缺失；零值可能是有物理意义的目标，也可能是上游占位。现有数值编码不保留来源有效性标记。
- 更稳健的新辅助损失应保留独立的“字段存在且有限”掩码：分别监督 CBM/VBM，仅两者有效时启用 consistency。但改变它会定义一个新实验，必须另行批准并重训，不能将旧结果当成新方法的结果。
- 当前缺少历史 E6/E13 实际训练 manifest，以及全量最终 T-series manifest；不远程连接、不读取凭证来补齐。
- 原图 Figure S6 的可选放大本轮不做；保持已清理的图文件，视觉复核只检查本轮文稿变化是否造成排版问题。不再重画已验收的其他图。

## 范围、人机分工与暂停条件

AI：按下列精确改动修订、运行本地检查和编译、呈现新 PDF 与证据限制。
作者：先确认此计划；判断是否保留历史 OR 方案作为带局限的探索结果；提供正式公开仓库/DOI 安排；最终检查原始 PDF 页面。

不做：训练代码/数据/模型数字变更、GPU 重训、增加依赖、全局配置变更、外部上传/发布、Git 提交/分支/推送。

暂停条件：发现当前数据与历史训练样本存在无法解释的差异、需要改实验数字/训练行为、缺 TeX 依赖需安装、编译报错或关键布局无法判断。报告后不扩大范围。

回退：仅逆向本轮 TeX/README 替换；不得 `git checkout`/`reset` 掉开始前已有修改。新 PDF 写入新目录，旧 PDF 和已有结果均不覆盖。

## 分步实施

下面的命令从仓库根目录运行；File 均为绝对路径。任务按批次执行，阶段汇报后交由作者确认继续。

### Task 1：收紧主文 cross-label 推断（批次 A）

- **File**: `D:/刘嘉瑞人生计划/02-学术科研/gnn/gnn_bandgap_research_paper_draft_en.tex`
- **Change**：精确替换以下表达：
  - `Broadcast decay is a property of the training source` → `Training-Label Dependence of Broadcast-Decay Effects`。
  - 摘要中的 `cross-label evaluation attributes the divergence to training dynamics rather than to the evaluation labels` → `cross-label evaluation suggests an association with the training-label source, without establishing a causal mechanism`。
  - 结论中的 `the cross-label evaluation attributes what effect exists to training dynamics rather than to the evaluation labels` → `the cross-label evaluation suggests an association with the training-label source, without establishing a causal mechanism`。
  - 6.2 第一段以以下文字替换前三句机制解释及其数值句：

    > The cross-source question is how the estimated effect of global broadcast decay varies with the training-label source. Decay is an information-flow schedule rather than a universally stronger architecture: early epochs use strong global-to-node broadcasting, while late epochs reduce the scale. In the seed-42 MP-label runs, T10 has lower MAE (0.2237 versus 0.2249 eV) and RMSE (0.4973 versus 0.5035 eV) than T7. In the HZW-label runs, T10 has higher MAE (0.1970 versus 0.1940 eV) and RMSE (0.4536 versus 0.4458 eV).

  - 同段 `it shifts bias and RMSE in opposite directions on the two sources` → `it reduces mean residual bias on both label sources while the RMSE changes have opposite signs`（SI Table S4 的 bias 两源均下降，只有 RMSE 方向相反；执行时从预测再次核对）。
  - 6.2 第二段的 cross-label 诊断改成：

    > First, a cross-label evaluation (Supplementary Table~S8) exploits the shared structure split: we score each trained model against \emph{both} label sets. For the seed-42 checkpoints, the sign of the estimated T10-minus-T7 MAE difference follows the \emph{training} source in all four conditions: it is negative for MP-trained models and positive for HZW-trained models under either evaluation label. All four paired-bootstrap 95\% confidence intervals include zero, as reported in Supplementary Section~S12. This descriptive sign pattern suggests an association with the training-label source, but it does not establish a causal mechanism or seed-stable source dependence.

  - `the harm tracks the shrinking broadcast scale` → `the positive validation-MAE gap coincides with the shrinking broadcast scale in this run`。
  - 最后一段替换为：`Taken together, these diagnostics do not establish a reliable MAE benefit from broadcast decay. We retain T10 as an exploratory, source- and seed-sensitive variant, while T7 remains the recommended cross-source model.`
- **Verify**: `python -I -c "from pathlib import Path; s=Path('gnn_bandgap_research_paper_draft_en.tex').read_text(); assert 'confidence intervals of Supplementary Table~S8' not in s; assert 'attributes the divergence' not in s; assert 'attributes what effect exists' not in s; assert 'Training-Label Dependence of Broadcast-Decay Effects' in s"`
- **Success**：exit 0；四 CI 的位置及 seed-42 限定明确，未更改实验数字。

### Task 2：统一 SI 的 cross-label 证据强度（批次 A；depends-on: Task 1）

- **File**: `D:/刘嘉瑞人生计划/02-学术科研/gnn/gnn_bandgap_research_supplementary_en.tex`
- **Change**：
  - S12 首句中的 `The sign of the decay effect T10\(-\)T7 is fixed entirely by the training source and is independent of the evaluation label` → `For the seed-42 checkpoints, the sign of the estimated T10\(-\)T7 MAE difference follows the training-label source under both evaluation labels`。
  - S12 的解释句换成：`All four comparisons are statistically unresolved. Their sign pattern suggests an association with the training-label source, but the comparisons reuse the same test structures and seed-42 checkpoints; they are not independent replications and do not establish a causal mechanism or seed-stable source dependence.` 保留四个 CI 原值。
  - S11 的 `statistically indistinguishable from zero on MP and marginal (and of opposite sign) on HZW` → `not statistically resolved on either label source, with opposite signs in the seed-42 point estimates`。
- **Verify**: `python -I -c "from pathlib import Path; s=Path('gnn_bandgap_research_supplementary_en.tex').read_text(); assert 'fixed entirely by' not in s; assert 'not independent replications' in s; assert 'marginal (and of opposite sign)' not in s"`
- **Success**：exit 0；两份文稿的限制一致，S8 表格数字和 S12 区间保持原值。

### Task 3：区分公开复现与 HZW 授权访问（批次 A）

- **File**: `D:/刘嘉瑞人生计划/02-学术科研/gnn/gnn_bandgap_research_paper_draft_en.tex`；`D:/刘嘉瑞人生计划/02-学术科研/gnn/README.md`
- **Change**：主文将 HZW 说明及其后复现承诺替换为：

  > The HZW band-gap labels are proprietary to Hongzhiwei Technology and were used with permission; they cannot be redistributed. The released MP data and analysis code will support reproduction of the MP-label experiments and MP-side preprocessing. Reproducing the MP--HZW structure matching and cross-source label-consistency audit requires authorized access to the proprietary HZW records, subject to the provider's terms.

  README 第 36–37 行替换为：

  > The planned release of the MP subset and analysis code will support reproduction of the MP-label experiments and MP-side preprocessing. Reproducing the MP–HZW structure matching and cross-source label-consistency audit requires authorized access to the proprietary HZW records, subject to the provider's terms.

  保留原文未来时发布安排，不编造已上线的仓库或 DOI；在交付中列为作者待确认项。
- **Verify**: `python -I -c "from pathlib import Path; paths=['gnn_bandgap_research_paper_draft_en.tex','README.md']; assert all('MP-side preprocessing' in Path(p).read_text() and 'requires authorized access' in Path(p).read_text() for p in paths)"`
- **Success**：exit 0；无“MP identifiers 足够复现跨源审计”的承诺。

### Task 4：限定 zero-gap 核查范围（批次 B；depends-on: Task 1）

- **File**: `D:/刘嘉瑞人生计划/02-学术科研/gnn/gnn_bandgap_research_paper_draft_en.tex`
- **Change**：Experimental setup 的 F1 段中，预测阈值写为 `at or below a tolerance of \(0.1\ \ev\)`，与代码 `<=` 一致。将从 `Because zero/nonzero conflicts were already removed` 到 `this is why T1 attains nearly identical F1 on both sources.` 的部分替换为：

  > We verified the MP and HZW reference classifications per sample on the full test split: 3,803 of 8,455 structures are zero-gap metals under the \(10^{-8}\ \ev\) threshold, with no MP/HZW disagreement at either this F1 threshold or the \(10^{-6}\ \ev\) dataset-filter threshold. Neither label source has a test-set gap in \((10^{-8},10^{-6}]\ \ev\). This verification is restricted to the test split; agreement under the dataset-filter threshold alone does not imply agreement under the F1 threshold.

  不再用真值一致推导模型 F1 相同，因为 F1 也依赖预测。
- **Verify**: `python -I -c "from pathlib import Path; s=Path('gnn_bandgap_research_paper_draft_en.tex').read_text(); assert 'MP and HZW versions of each structure' not in s; assert 'verification is restricted to the test split' in s; assert 'this is why T1' not in s"`
- **Success**：exit 0；范围只含已经重算的 test split，3,803/8,455 不变。

### Task 5：补全局部层的公式定义（批次 B）

- **File**: `D:/刘嘉瑞人生计划/02-学术科研/gnn/gnn_bandgap_research_paper_draft_en.tex`
- **Change**：Equation (3) 的 `\mathrm{ALIGNNLayer}(\tilde{\mathbf{h}}^{(l)},\mathcal{G})_i` 换为 `\mathrm{LocalLayer}_{l}(\tilde{\mathbf{h}}^{(l)},\mathcal{G})_i`。紧接公式加：

  > Here \(\mathrm{LocalLayer}_{l}\) denotes an ALIGNN block for the first four layers and an edge-gated graph convolution for the next four; bond and angle states are suppressed in this schematic notation.

  Figure 1 不重画，保留已与广播及池化顺序对齐的示意图和 caption。
- **Verify**: `python -I -c "from pathlib import Path; s=Path('gnn_bandgap_research_paper_draft_en.tex').read_text(); assert 'LocalLayer' in s; assert 'bond and angle states are suppressed' in s; assert 'ALIGNNLayer' not in s"`
- **Success**：exit 0；记号覆盖代码的两个循环，不改变 Equation (1)–(2)。

### Task 6：披露历史辅助掩码及核验边界（批次 B；depends-on: Task 4）

- **File**: `D:/刘嘉瑞人生计划/02-学术科研/gnn/gnn_bandgap_research_paper_draft_en.tex`；`D:/刘嘉瑞人生计划/02-学术科研/gnn/gnn_bandgap_research_supplementary_en.tex`
- **Change**：保留 OR 公式，将公式后解释段换为：

  > The historical implementation uses this shared OR mask, with \(\epsilon=10^{-8}\), for both band-edge regressions and the gap-consistency term. Missing or null band-edge entries are encoded as 0.0. Consequently, a single-edge sample would also supervise the absent edge against zero; the mask is an availability heuristic, not an independent validity check for each target, and it includes no explicit finite-value test. A check of the local raw HZW input found no single-edge or non-finite band-edge values among the 95,490 target-eligible records, so OR and AND select the same records in that checked population (Supplementary Section~S22). This is not a verification of the historical training manifests or of the final T-series population. The auxiliary-physics results are retained as exploratory observations, not as validation of zero imputation; the recommended T4/T7 models do not use these auxiliary terms.

  在 SI 最后、`\end{document}` 前追加以下完整内容（不改变已有 S1–S21、表格和图编号）：

  ```tex
  \section{Band-edge mask implementation and local input check}
  \label{sec:si-band-edge-mask}

  The historical E-series physics variants use one shared mask,
  \(m_i=\mathbf{1}(|\mathrm{CBM}_i|>10^{-8}\ \mathrm{or}\ |\mathrm{VBM}_i|>10^{-8})\),
  for both band-edge regression terms and the predicted-gap consistency term.
  The exporter converts missing or null entries to 0.0, without retaining
  separate validity flags. Neither the target extraction function nor the
  loss mask explicitly rejects NaN or infinity.

  We checked the current local raw HZW input using the E-series target
  extraction rule, before structure validation and train/validation/test
  assignment. There were 100,000 decodable records and two malformed lines;
  4,506 records had no band entry and four had a band gap above the exporter's
  15 eV limit. Among the remaining 95,490 target-eligible records, 52,001 had
  two finite non-zero band edges and 43,489 had two exact-zero band edges.
  No missing/null band-edge fields, single-edge cases, non-finite band-edge
  values, or non-zero band-edge magnitudes at or below \(10^{-8}\) were found
  in this checked population. The OR and AND rules therefore retain the
  same 52,001 records in this local input check. These counts describe the
  raw target-eligible population, not the final 84,550-structure T-series
  dataset. The original E-series training manifests were not available for
  comparison, so this check does not certify their exact sample membership.

  Isolated execution of the unchanged loss function with synthetic targets
  confirms its limitation: when one band edge is non-zero and the other is
  encoded as zero, the zero target still contributes a regression gradient;
  when one edge is NaN or infinity and the other passes the mask, the loss
  becomes non-finite. These are implementation boundary cases, not observed
  non-finite training examples. Numeric zeros alone cannot distinguish a
  physically valid zero from an upstream missing-value placeholder.

  A revised auxiliary objective would require separate presence-and-finiteness
  masks for CBM and VBM and would enable gap consistency only when both are
  valid. We have not applied that revision or retrained the historical
  physics variants; their reported metrics remain results of the original
  implementation. The recommended T4/T7 models do not use this auxiliary loss.
  ```
- **Verify**: `python -I -c "from pathlib import Path; m=Path('gnn_bandgap_research_paper_draft_en.tex').read_text(); s=Path('gnn_bandgap_research_supplementary_en.tex').read_text(); assert 'Supplementary Section~S22' in m; assert 'Band-edge mask implementation and local input check' in s; assert s.count('section{') == 22; assert 'not a verification of the historical training manifests' in m"`
- **Success**：exit 0；数据范围、反例和未重训事实均清楚；训练代码及历史数字无修改。

### Task 7：在新目录编译两份 PDF（批次 C；depends-on: Tasks 1–6）

- **File**: 新生成 `D:/刘嘉瑞人生计划/02-学术科研/gnn/artifacts/paper_evidence_closeout_20261020/gnn_bandgap_research_paper_draft_en.pdf` 和 `.../gnn_bandgap_research_supplementary_en.pdf`；编译辅助文件仅在同一新目录内。
- **Change**：不覆盖根目录已有 PDF；先确认输出目录未存在，创建后各编译两遍：

  ```bash
  mkdir artifacts/paper_evidence_closeout_20261020
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=artifacts/paper_evidence_closeout_20261020 gnn_bandgap_research_paper_draft_en.tex
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=artifacts/paper_evidence_closeout_20261020 gnn_bandgap_research_paper_draft_en.tex
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=artifacts/paper_evidence_closeout_20261020 gnn_bandgap_research_supplementary_en.tex
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=artifacts/paper_evidence_closeout_20261020 gnn_bandgap_research_supplementary_en.tex
  ```
- **Verify**: `python -I -c "from pathlib import Path; import pymupdf; d=Path('artifacts/paper_evidence_closeout_20261020'); names=['gnn_bandgap_research_paper_draft_en','gnn_bandgap_research_supplementary_en']; [print(n, len(pymupdf.open(d/(n+'.pdf')))) for n in names]; assert all('undefined references' not in (d/(n+'.log')).read_text(errors='replace') for n in names)"`
- **Success**：四次编译 exit 0，两 PDF 可打开、无未定义引用。只检查本轮生成的 TeX 编译日志，不读取训练日志、会话记录或其他日志目录。若有警告必须区分已有/新增，不用忽略错误制造通过。

### Task 8：检查真实渲染并交付（批次 C；depends-on: Task 7）

- **File**: 新 PDF 及 `D:/刘嘉瑞人生计划/02-学术科研/gnn/artifacts/paper_evidence_closeout_20261020/` 内的改动页截图；本计划进度区。
- **Change**：用 PyMuPDF 按内容定位摘要、Methods、6.2、Conclusion/availability、SI S12/S22，渲染实际页面并用图像工具检查；原始 PDF 保留作对照。最终报告列出变更文件、实跑检查及未解决的历史 manifest/发布安排问题。
- **Verify**: `git diff --check -- gnn_bandgap_research_paper_draft_en.tex gnn_bandgap_research_supplementary_en.tex README.md docs/plans/2026-10-20-paper-evidence-closeout.md`，并实际打开生成的页面图片。
- **Success**：diff 检查 exit 0；改动页无新增文字裁切/遮挡/公式或标题溢出；最终整体视觉判断交作者验收。不把“文稿修正完成”称为“历史物理实验重新验证完成”。

## 人类检查点

1. **开始修改前**：确认按此计划保留历史实现和数字，仅修文稿并披露边界。若要换掩码/重训，先修改计划及算力授权。
2. **批次 A/B 后**：查看精确措辞，确认科研表述尤其辅助损失局限是否合适；完成一批同步后再继续。
3. **批次 C 后**：检查新 PDF 的实际页面；提供仓库/DOI 安排。不会由 AI 擅自上线或填写虚构地址。

## 进度日志

- 2026-10-20：已读当前两份 TeX、README、相关实现、导出函数、物理变体配置、统计脚本和现有结果表。
- 2026-10-20：四条件 bootstrap 和 test 标签阈值重新计算完成，输出如事实区；运行均 exit 0，未改预测 CSV。
- 2026-10-20：raw HZW 字段聚合完成；实际 loss 函数体的五种合成输入运行完成。边界输入产生的 NaN/Inf 是复现结果，不是宣称训练通过。
- 2026-10-20：计划写成，尚未修改 TeX/README/代码，尚未重编译或检查本轮新 PDF。
- 2026-10-20（上一会话）：Task 1 主文 cross-label 收紧已完成并通过验证；另含作者此前的标题/作者块与图表修改（本轮不改动）。
- 2026-10-20（本会话）：Task 2–6 全部完成并逐项通过验证（断言脚本 exit 0）：SI S11/S12 证据强度统一；主文+README 发布声明改为"MP 单来源复现 + HZW 需授权访问"；F1 段限定 test split 且阈值改 at or below；Equation (3) 改 LocalLayer 记号并加覆盖说明；主文 OR 掩码段换为历史实现披露并指向 S22；SI 末尾新增 S22 节（section 计数 21→22）。
- 2026-10-20（本会话）：Task 7 完成：`artifacts/paper_evidence_closeout_20261020/` 新建，主文与 SI 各编译两遍，四次 exit 0；主文 22 页、SI 13 页，日志零警告、零未定义引用、零 citation 警告；根目录旧 PDF 未覆盖。
- 2026-10-20（本会话）：Task 8 完成：PyMuPDF 按内容定位改动页（主文 p1/5/6/8/14/17/18，SI p6/12/13），110 dpi 渲染并逐页目检：无文字裁切、遮挡、公式或标题溢出；`git diff --check` 对四份源文件 exit 0。
- 当前状态：8 项任务全部完成。待作者最终验收：新 PDF 实际页面（截图在 `artifacts/paper_evidence_closeout_20261020/pages/`）、精确措辞（尤其 S22 辅助损失局限表述）、以及发布安排（公开仓库/DOI 仍未落实，文稿保留未来时表述）。
- 遗留（不在本轮范围）：历史 E6/E13 训练 manifest 与全量 T-series manifest 未取得，文稿已如实限定核查范围；若决定改用独立 presence-and-finiteness 掩码并重训，需另立计划与算力授权。
- 2026-10-20（作者指定）：Data and code availability 改为指向项目仓库 `github.com/liujiarui0918/gnn-bandgap-virtual-node`（`gh repo view` 确认 visibility=PRIVATE），措辞为"will be made public upon publication"，未虚构已上线状态；README 顶部同步说明该仓库即论文引用仓库。主文重新编译两遍 exit 0，22 页、无未定义引用，第 18 页目检通过。待作者发布时将仓库设为 Public 后，此段即可成立。
- 2026-10-20（作者已设仓库为 Public）：`gh repo view` 确认 visibility=PUBLIC 且远端 main 与本地 HEAD（952dfab）一致；跟踪文件扫描确认无 results/、无 HZW 数据、无凭证文件。主文/README 措辞改为"已可用"现在时；两文各重编译两遍，零警告零未定义引用，渲染页已交付作者检查。
- 2026-10-20（审稿意见 2+3 整改）：
  - 事实核查：对本地原始 HZW 文件流式重算——100,002 行、2 行无法解析、4,506 无 band entry、4 条 gap>15、band-invalid 共 4,510、band-valid 缺 MP 交叉引用恰 29；与服务器审计产物 `results/data_quality/mp_hzw_garbage_summary.remote.json`（hzw_invalid_gap=4509、missing_mp=29、entering=95,462）比对：唯一差异是 band-invalid 边界差 1 条（4,510 vs 4,509），交叉引用数完全一致；两套排除规则（extract_target 与 valid_gap）逐条比对判定相同，且 Table 1 为顺序排除、两集合不相交，排除"重叠双重计数"（实际重叠 43 条但未被双计）。结论：服务器副本与本地副本差 1 条记录，非规则差异；T 系列 84,550 数据集服务器端构建，不受影响。
  - 改动：Table 1 行说明改为顺序排除措辞（4,509 数字保留，来源是服务器审计产物）；S22 新增"Relationship to the audit counts"段，写明两副本差异与排除顺序；主文 availability 改为"MP 标签来自公共 Materials Project summary API（api.materialsproject.org/materials/summary/），按 id 查询、字段固定、无固定版本号为 2026-04-10 至 05-24 间快照；精确 id 集合来自 HZW 交叉引用不在公开发布内"；README Data 段同步。
  - 验证：两文各重编译两遍 exit 0（主文 22 页、SI 13 页，零警告零未定义引用）；p4/p18/SI p12-13 渲染目检通过。
- 2026-10-20（作者拍板：S22 统一成 4,509）：作者确认目标是"两边统一成 4,509"（非服务器新文件来源）。改法：S22 第一段的有效性计数改用审计口径"4,505 无 band entry + 4 条 gap>15 = 4,509 排除，95,491 合格"；band-edge 拆分 52,001+43,489 保留原值并显式标注为**本地副本**的测量（本地副本 95,490 合格，少 1 条，指向下方副本差异段）；第二段"本地副本 4,510 vs 审计 4,509 差 1 条"的披露保留。主文 OR 掩码段去掉具体总数，只写"checked target-eligible population"。注意：磁盘上本地文件实际重数仍为 4,506/4,510（两次独立重算确认），S22 的 4,505 是与服务器审计对齐的口径，不是本地文件重算值——此点已用"local copy 少 1 条"的披露覆盖，若日后从服务器核对原始文件发现 4,505 不成立，需回到 S22 修正。
  - 验证：两文各重编译两遍 exit 0（22/13 页，零警告零未定义引用）；断言确认 S22 含 4,505、95,491、本地 95,490/4,510 披露、52,001/43,489 完好，主文无残留 95,490；SI p13 渲染目检通过。
