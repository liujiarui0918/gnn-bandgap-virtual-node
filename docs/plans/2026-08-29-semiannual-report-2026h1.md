# Plan：2026 年上半年研究团队综合汇报 Pympress PDF

日期：2026-08-29

## 1. 目标与通俗解释

为刘嘉瑞制作一份面向研究团队半年度评议的汇报材料，评价周期为 **2026 年 1 月至 6 月**。正文严格控制为 **14 页**，围绕“目标—证据—偏差—下一步”展开，不做日常工作流水账；另设 3 页证明材料附件，不计入正文页数。

汇报主线采用用户确认的 **成果答辩型方案 A**：以晶体带隙 GNN 项目形成可投稿论文为上半年核心目标，把学术进展与工程贡献分开陈述；将 2026 年 7—8 月的英文稿完善明确标记为“周期后补充进展”，不计入上半年目标完成度。

## 2. 完成标准

- 生成 14 页正文，覆盖管理办法第十一条要求的十类内容。
- 生成 3 页附件，用于结果表、数据审计证据和文件索引。
- 生成普通 16:9 PDF 和 Pympress 双屏 PDF；双屏右侧包含逐页中文讲稿。
- 生成独立 Markdown 讲稿，正文控制在约 15 分钟，并提供 12 分钟删减方案。
- 正文页码显示 `1/14` 至 `14/14`；附件显示“附件 1—3”，不计入正文页数。
- 周期内事实与周期后补充进展在版面和讲稿中显式区分。
- 学术成果和工程贡献分别陈述，不重复计算同一产物。
- 不把团队共同成果表述为个人独立成果；统一使用“在导师指导与团队协作下，我主要承担/参与”。
- 两份 LaTeX 日志无 overfull、undefined control sequence 或 LaTeX error。
- 全部页面完成渲染检查；普通 PDF 不含讲稿文本，双屏 PDF 包含讲稿文本；字体嵌入。
- `git diff --check` 返回 0；不改动已有未提交文件 `期刊投稿目标清单.md`。

## 3. 当前事实

### 3.1 管理办法

- 正文原则上不超过 15 页，必要证明材料可作为附件。
- 个人汇报 15 分钟，提问讨论 7 分钟，主持人归纳 3 分钟。
- 重点核实目标、证据、个人贡献边界、未完成原因、下一周期目标与资源需求。
- 学术进展和工程贡献必须分开汇报、分别评价。
- 管理办法附件名单将姓名误写为“刘家瑞”；用户已明确确认正确姓名为“刘嘉瑞”。
- 未公开论文、源码、原始数据和数据资产需要最小披露；PPT 只展示摘要与统计证据。

### 3.2 个人信息和发展路径

- 姓名：刘嘉瑞。
- 身份：物理系研究生，2026 年秋季进入研二。
- 研究方向：物理人工智能 / 机器学习科研。
- 当前发展路径：以学术研究为主，同时申请工程准入，按“复合型（学术主线 + 工程进入）”表述。
- 下一周期：继续跟随胡顺波导师深入机器学习科研，同时兼顾刘俊杰导师的量子计算课题。
- 工程意向：进入。

### 3.3 2026 年 1—6 月可核验项目进展

- 1—2 月：未找到稳定可归档的项目产物；在汇报中如实写为研究方向准备期，不虚构完成项。
- 3 月：建立 GNN/材料性质预测文献基础，保留多篇核心论文 PDF。
- 4 月：完成 CGCNN、ALIGNN、M3GNet 基线框架与多种 virtual-node/physics 变体；建立训练、监控和汇报材料。
- 5 月：完成 MP 迁移、MP/HZW matched 数据构建和 structure-first audit；识别标签源冲突；遇到 Slurm 账户 GPU QOS 阻塞。
- 6 月：完成 relaxed clean equal-atoms 双标签 T 系列验证，形成 15 页中文论文初稿 PDF。

### 3.4 关键数据与模型证据

- 审计总记录：95,462。
- 结构匹配：92,787；结构不匹配：39；缺 MP 记录：2,636。
- 结构匹配内部标签冲突：
  - MP metal / HZW nonmetal：3,142；
  - MP nonmetal / HZW metal：3,046；
  - 同类别但 gap 差值大于 1 eV：1,121。
- 最终 relaxed clean equal-atoms：84,550 条，同一结构划分分别使用 MP 和 HZW 标签。
- MP label：T2 ALIGNN 0.240043；T4 single VN 0.232520；T7 VN + Huber 0.224931；T10 decay + Huber 0.223686。
- HZW label：T2 ALIGNN 0.212124；T4 single VN 0.201762；T7 VN + Huber 0.194002；T10 decay + Huber 0.197034。
- 相对 ALIGNN：MP T7 改善 6.30%，MP T10 改善 6.81%；HZW T7 改善 8.54%。
- 上半年结果资产：25 个完整 `metrics.json`；最终双标签 T 系列各 9 个完整实验；49 个在 6 月 30 日前形成的 YAML 配置；6 个测试文件；多套数据审计、队列和监控脚本。
- 截至 2026-06-11：中文论文初稿 PDF 15 页。
- 周期后补充：截至 2026-08-25，英文稿 PDF 22 页，已进入投稿准备阶段；该状态只作为补充，不计入上半年完成度。

## 4. 假设与未知

- 周期初没有正式书面 KPI，因此第 2 页明确写“无书面 KPI，按可核验产物回溯目标”。
- “文章已写好准备投稿”是当前状态，不等于 6 月 30 日前已投稿。
- 工程项目尚未给出具体项目名称、职责和验收标准，因此只报告上半年形成的研究工程化能力，并将“申请进入、待明确任务书”放到下一周期。
- 个人贡献无法仅凭仓库证明独立归属，统一采用团队协作边界表述，不声称独立完成论文或全部代码。
- 汇报具体日期尚未提供，封面使用“2026 年 8 月”而非虚构某一天。

## 5. 范围与非目标

### 范围

- 2026 年上半年晶体带隙 GNN 项目的目标、过程、数据、模型、论文和工程化产物。
- 阴性结果、GPU 资源阻塞、数据源冲突和论文未投稿等偏差。
- 2026 年下半年学术目标、量子计算协同、工程准入意向和资源需求。
- 1 页周期后补充状态，明确日期和口径。

### 非目标

- 不把 7—8 月的新实验、英文写作或扩散模型论文阅读计入上半年绩效。
- 不展示敏感源码、远程服务器地址、密钥、原始大表或未公开论文全文。
- 不决定论文署名、工程准入最终结论或成果权属。
- 不创建 Git commit、分支或远程推送。
- 不修改项目训练代码、结果文件和用户已有未提交文档。

## 6. 人机分工

- AI：读取管理办法和项目证据；整理事实口径；生成图表、Pympress 源文件、PDF 和讲稿；执行自动与视觉检查。
- 人类：最终确认复合型发展路径是否符合团队口径；确认工程项目进入后的具体任务；彩排并决定是否删减附件或技术细节；对个人贡献表述做最终验收。

## 7. 方案与取舍

### 已选方案

采用“成果答辩型 A + 论文投稿主线”，原因是最贴合评议表，也能把项目从失败假设到数据审计和论文初稿的研究闭环讲清楚。

### 放弃方案

- 放弃纯时间线：容易变成流水账，不利于 15 分钟评议。
- 放弃纯论文汇报：无法充分覆盖个人发展、工程贡献、资源需求和工程准入意向。
- 不直接复用 29 页技术汇报：正文页数和管理目标均不符合本次要求。

## 8. 正文 14 页规约

1. **封面与个人信息**：姓名、评价周期、物理人工智能、物理系研究生、进入研二、复合型倾向。
2. **周期目标与完成度**：无书面 KPI；回溯目标为形成可投稿研究闭环；完成/部分完成/未完成清单。
3. **上半年关键时间线**：1—2 月准备、3 月文献、4 月框架与模型、5 月迁移与审计、6 月验证与初稿。
4. **研究问题与技术路线**：band gap、ALIGNN、全局信息、跨源标签一致性。
5. **学术进展一：模型谱系**：baseline → single VN → complex variants → Huber/decay。
6. **学术进展二：数据审计转折**：95,462 → 92,787；标签冲突三类数字。
7. **学术进展三：双标签最终结果**：MP/HZW T2、T4、T7、T10 对比图和改善百分比。
8. **当前学术结论与个人研究贡献**：single VN、Huber、数据审计、复杂模型边界。
9. **成果与验证材料**：metrics、配置、测试、论文初稿；周期后英文稿单独标注。
10. **工程任务、交付和个人贡献**：训练框架、数据流水线、Slurm 队列、监控、可复现实验资产；与学术结论分开。
11. **未达预期与阴性结果**：复杂 attention/dual-slot/top-k 不稳，路由假设失败，GPU QOS 阻塞，未在周期内投稿。
12. **当前问题与风险**：多 seed 不足、标签噪声、投稿不确定性、双课题时间分配、工程任务未明确。
13. **下一周期目标与里程碑**：7—8 月已完成状态；9 月投稿；9—10 月补验证；10—12 月新机器学习与量子计算子课题。
14. **发展路径、工程意向与资源需求**：复合型、工程进入；GPU、投稿内审、双导师优先级、明确工程任务书；三句总结。

## 9. 附件规约

- **附件 1**：MP/HZW T 系列完整指标表。
- **附件 2**：数据审计口径与证据路径。
- **附件 3**：代码、配置、测试、论文和报告的文件索引。

## 10. 文件与影响面

新增：

- `C:\Users\ljr-w\刘嘉瑞人生计划\02-学术科研\gnn\docs\plans\2026-08-29-semiannual-report-2026h1.md`
- `C:\Users\ljr-w\刘嘉瑞人生计划\02-学术科研\gnn\semiannual_report_2026h1_slides.tex`
- `C:\Users\ljr-w\刘嘉瑞人生计划\02-学术科研\gnn\semiannual_report_2026h1_slides.pdf`
- `C:\Users\ljr-w\刘嘉瑞人生计划\02-学术科研\gnn\semiannual_report_2026h1_slides_pympress.pdf`
- `C:\Users\ljr-w\刘嘉瑞人生计划\02-学术科研\gnn\semiannual_report_2026h1_speech.md`
- `C:\Users\ljr-w\刘嘉瑞人生计划\02-学术科研\gnn\figures\semiannual_report_2026h1\*.tex|*.pdf`
- `C:\Users\ljr-w\刘嘉瑞人生计划\02-学术科研\gnn\artifacts\semiannual_report_2026h1\` 下的抽取文本和渲染检查产物。

不修改已有代码、论文、结果和用户未提交文件。

## 11. 可执行任务

### Task 1：生成双标签结果矢量图

- **File**: `C:\Users\ljr-w\刘嘉瑞人生计划\02-学术科研\gnn\figures\semiannual_report_2026h1\dual_label_results.py` 和 `dual_label_results.pdf`
- **Change**: 读取两个 `comparison_summary.csv`，只绘制 T2/T4/T7/T10 的 MP/HZW MAE；标注 6.30%、6.81%、8.54% 改善和“越低越好”。
- **Verify**: `python figures\semiannual_report_2026h1\dual_label_results.py && pdfinfo figures\semiannual_report_2026h1\dual_label_results.pdf`
- **Success**: 命令退出 0；PDF 为 1 页；输出中包含四个实验和两套标签的精确 MAE。

### Task 2：生成数据审计矢量图

- **File**: `C:\Users\ljr-w\刘嘉瑞人生计划\02-学术科研\gnn\figures\semiannual_report_2026h1\data_audit_summary.py` 和 `data_audit_summary.pdf`
- **Change**: 从 `mp_hzw_structure_first_summary.json` 读取总数、结构匹配、缺失和三类标签冲突，绘制漏斗与冲突条形图；不嵌入原始记录。
- **Verify**: `python figures\semiannual_report_2026h1\data_audit_summary.py && pdfinfo figures\semiannual_report_2026h1\data_audit_summary.pdf`
- **Success**: 命令退出 0；PDF 为 1 页；图中显示 95,462、92,787、39、2,636、3,142、3,046、1,121。

### Task 3：编写 14 页正文和 3 页附件

- **Depends on**: Task 1, Task 2
- **File**: `C:\Users\ljr-w\刘嘉瑞人生计划\02-学术科研\gnn\semiannual_report_2026h1_slides.tex`
- **Change**: 复用 Berlin/whale/ctex/16:9/Pympress notes 模板；按第 8、9 节逐页实现；主正文页脚固定 `/14`，附件改为“附件 1—3”；每页加入中文 `\speakernote{}`。
- **Verify**: `python -c "from pathlib import Path; s=Path('semiannual_report_2026h1_slides.tex').read_text(encoding='utf-8'); print(s.count('\\begin{frame}'), s.count('\\speakernote{'))"`
- **Success**: 输出为 `17 17`；源码包含“2026 年 1—6 月”“周期后补充”“工程意向：进入”和“刘嘉瑞”。

### Task 4：编写独立逐页讲稿

- **Depends on**: Task 3
- **File**: `C:\Users\ljr-w\刘嘉瑞人生计划\02-学术科研\gnn\semiannual_report_2026h1_speech.md`
- **Change**: 为 14 页正文和 3 页附件编写口语稿；正文总时长 13—15 分钟；首页给出 12 分钟删减方案；附可能追问短答案。
- **Verify**: `python -c "from pathlib import Path; s=Path('semiannual_report_2026h1_speech.md').read_text(encoding='utf-8'); print(sum(1 for l in s.splitlines() if l.startswith('## 第 ')))"`
- **Success**: 输出 `17`；包含“团队协作边界”“未在周期内投稿”“工程项目进入”。

### Task 5：编译普通版 PDF

- **Depends on**: Task 3
- **File**: `C:\Users\ljr-w\刘嘉瑞人生计划\02-学术科研\gnn\semiannual_report_2026h1_slides.pdf`
- **Change**: 使用 XeLaTeX 编译两遍，普通版隐藏备注。
- **Verify**: `xelatex -interaction=nonstopmode -halt-on-error semiannual_report_2026h1_slides.tex`（两遍），随后 `pdfinfo semiannual_report_2026h1_slides.pdf`。
- **Success**: 17 页；页面尺寸约 `453.54 x 255.12 pt`；日志无 fatal/overfull。

### Task 6：编译 Pympress 双屏 PDF

- **Depends on**: Task 3
- **File**: `C:\Users\ljr-w\刘嘉瑞人生计划\02-学术科研\gnn\semiannual_report_2026h1_slides_pympress.pdf`
- **Change**: 通过 `\def\pympressnotesright{1}` 编译两遍，右侧显示逐页中文讲稿。
- **Verify**: `xelatex -interaction=nonstopmode -halt-on-error -jobname=semiannual_report_2026h1_slides_pympress "\def\pympressnotesright{1}\input{semiannual_report_2026h1_slides.tex}"`（两遍），随后 `pdfinfo`。
- **Success**: 17 页；页面宽度约 `907.09 pt`；日志无 fatal/overfull。

### Task 7：执行文本层和字体检查

- **Depends on**: Task 5, Task 6
- **File**: 两份 PDF 和 `C:\Users\ljr-w\刘嘉瑞人生计划\02-学术科研\gnn\artifacts\semiannual_report_2026h1\text_checks\`
- **Change**: 用 `pdftotext` 检查普通版不含开场讲稿而双屏版包含；用 `pdffonts` 检查中文字体嵌入；检查正文页码和附件标题。
- **Verify**: `pdftotext -layout ...`、`pdffonts ...`、Python 文本断言脚本。
- **Success**: 普通版 `opening_note=False`；双屏版 `opening_note=True`；Microsoft YaHei 等中文字体 `emb=yes`；正文与附件关键词存在。

### Task 8：执行全页视觉和仓库检查

- **Depends on**: Task 5, Task 6, Task 7
- **File**: `C:\Users\ljr-w\刘嘉瑞人生计划\02-学术科研\gnn\artifacts\semiannual_report_2026h1\rendered_*\`
- **Change**: 用 `pdftoppm` 渲染普通版全部 17 页并生成联系表；抽查双屏版封面、结果页、阴性结果页、下一周期页和最长讲稿页；执行 `git diff --check` 和 `git status --short`。
- **Verify**: `pdftoppm ...`、图像人工检查、`git diff --check`。
- **Success**: 无文字/图表裁切；双屏左右内容完整；`git diff --check` 退出 0；只有本任务新增文件和用户原有 `期刊投稿目标清单.md` 修改。

## 12. 风险、暂停条件与回退

- 如果结果图与 CSV 不一致，立即停止制稿并以 CSV 为准修正。
- 如果某项个人贡献无法从现有材料或用户确认中支撑，改成“参与/主要承担”或移入待确认，不做独立归属声明。
- 如果 14 页无法容纳全部十类内容，优先压缩技术细节到附件，不增加正文页数。
- 如果双屏备注溢出，缩短 `\note{}`，详细内容保留在 Markdown。
- 如果表格在 `pgfpages` 下消失，单独编译成矢量 PDF 后嵌入。
- 所有文件使用独立命名；删除新增文件即可回退，不影响现有项目。

## 13. 人类检查点

- 计划确认点：用户已确认方案 A 和姓名，授权继续执行。
- 交付验收点：用户需确认“复合型（学术主线 + 工程进入）”是否符合团队正式口径。
- 汇报前：用 Pympress 彩排一次，确认 15 分钟；对未公开论文内容和工程准入措辞做最终审阅。

## 14. 进度日志

- **已完成**：管理办法抽取与口径核验；14 页正文 + 3 页附件；普通放映 PDF；Pympress 双屏 PDF；独立逐页念稿；数据审计图和双标签结果图。
- **最终编译**：普通版和双屏版均从当前源文件用 XeLaTeX 各编译两遍；普通版 17 页、`453.54 x 255.12 pt`；双屏版 17 页、`907.09 x 255.12 pt`。
- **最终机器验收**：普通版不含讲稿文本，双屏版含讲稿文本；中文字体已嵌入；两版均通过 PDF 文本和字体断言；最终日志无 `LaTeX Error`、未定义控制序列或 `Overfull`。
- **最终视觉验收**：普通版 17 页全部渲染；检查封面、结果页、工程页、风险页、下一周期页及 3 页附件；双屏版抽查封面、结果页、风险页、总结页和附件页，未发现明显裁切或重叠。
- **独立讲稿**：`semiannual_report_2026h1_speech.md` 含 17 页讲稿、约 15 分钟版本、12 分钟删减方案和常见追问速答。
- **仓库检查**：`git diff --check` 返回 0；未修改已有的 `期刊投稿目标清单.md`。
- **剩余人类检查点**：汇报前用 Pympress 打开双屏 PDF 彩排一次，按实际语速确认是否采用 12 分钟删减方案，并最终确认工程进入措辞符合团队正式口径。
- **阻塞**：无。
