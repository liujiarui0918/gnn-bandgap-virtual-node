# 理论物理启发的扩散模型论文组会交付计划

日期：2026-08-29

## 1. 目标与通俗解释

为 2026-08-30 的组会准备一份可以直接讲的中文 PDF 演示稿，介绍两篇近几年真正把理论物理方程用于扩散/生成过程设计的论文：

1. **PFGM++: Unlocking the Potential of Physics-Inspired Generative Models**（ICML 2023）。
2. **Speed-Accuracy Relations for Diffusion Models: Wisdom from Nonequilibrium Thermodynamics and Optimal Transport**（Physical Review X 2025）。

第一篇回答“能否用电静力学的 Poisson 场重新设计去噪轨迹”；第二篇回答“能否用非平衡热力学与最优传输给加噪日程建立理论准则”。二者一篇偏模型构造，一篇偏理论边界，组合后能形成完整故事。

## 2. 完成标准

- 生成一份约 12--15 分钟、约 14 页的 Beamer 演示稿。
- 版式复用本项目既有的 16:9、`Berlin + whale`、中文 `ctex` 风格。
- 每页包含 `\note{}` 中文讲稿，可在 Pympress 演讲者视图中直接念。
- 生成普通放映 PDF 和“左侧幻灯片、右侧讲稿”的 Pympress PDF。
- 另生成逐页中文讲稿 Markdown，便于手机或打印查看。
- 关键论文信息、公式、图和实验数字可追溯到官方论文原文。
- 编译无致命错误；检查 PDF 页数、字体/图片溢出和代表性页面渲染。
- 执行 `git diff --check`，不修改已有未提交文件 `期刊投稿目标清单.md`。

## 3. 当前事实

- 已找到既有模板：`gnn_bandgap_research_final_slides.tex`。
- 模板使用 `ctex`、Beamer 16:9、`Berlin`/`whale`，并通过 `\speakernote{}` 写 Pympress 备注。
- 本机已有 `pdflatex`、`xelatex`、`latexmk`、`pympress`、`pdftoppm` 和 `pdfinfo`。
- 两篇论文的官方 PDF 和 arXiv 源码已下载到：
  - `artifacts/diffusion_physics_group_meeting/sources/pfgmpp_pmlr.pdf`
  - `artifacts/diffusion_physics_group_meeting/sources/speed_accuracy_arxiv.pdf`
- 已核验的 PFGM++ 主干：高维电场、`r=||z||` 对称约化、沿电场 ODE 生成、有限 `D` 的鲁棒性/刚性权衡、`D -> infinity` 恢复扩散模型。
- 已核验的 PRX 主干：Fokker--Planck/Langevin 表述、熵产生率、速度--精度不等式、以 2-Wasserstein 测地线最小化扩散速度代价。

## 4. 假设与未知

- 假设组会听众了解基础机器学习，但不一定熟悉随机热力学，因此保留一页扩散模型基础和一页物理量翻译。
- 假设目标是“能讲清楚并经得起一般追问”，而非逐式证明定理；完整证明不进入正文。
- 未知具体发言时长；默认按 12--15 分钟设计。若现场只有 10 分钟，可跳过标注为“可压缩”的细节页。
- 原论文图可能包含较小英文标注；将只使用经实际渲染检查、在投影上仍可辨认的图，并在旁边补中文结论。

## 5. 范围与非目标

### 本轮范围

- 两篇论文的研究问题、物理来源、关键公式、方法流程、主要实验、优缺点和共同启发。
- 基础扩散模型与两篇工作的关系。
- 逐页口语化讲稿和可能追问提示。

### 非目标

- 不复现训练或重新跑论文实验。
- 不完整推导全部定理。
- 不介绍泛化的“physics-informed diffusion”应用论文，避免选题发散。
- 不创建 Git commit、分支或远程推送。
- 不修改项目已有研究代码和无关文档。

## 6. 人机分工

- **AI**：检索与核验原文、选择主线、制作图文和公式、写讲稿、编译、渲染检查、报告剩余风险。
- **人类**：最终确认姓名/日期、实际发言时长、是否删减细节页，并在 Pympress 中做一次口头彩排与最终视觉验收。

## 7. 方案与取舍

### 采用方案

- 论文一选 PFGM++：物理方程直接改变扰动核和逆向生成轨迹，实验结果成熟，适合解释“物理结构如何替代纯经验设计”。
- 论文二选 PRX 2025：从非平衡热力学推导噪声日程与生成误差的关系，直接覆盖“加噪怎样更合理”。
- 统一叙事：**物理不仅能提供类比，还能提供轨迹、损失、超参数和可验证的上界。**

### 放弃方案

- 不选仅在 loss 中加入 PDE residual 的领域应用论文：它们更偏约束生成结果，而非直接改造一般扩散模型的加噪/去噪机制。
- 不选两篇都来自 Poisson-flow 系列：会导致内容重复，无法覆盖噪声日程的热力学优化。

## 8. 文件与影响面

预计新增：

- `docs/plans/2026-08-29-physics-informed-diffusion-group-meeting.md`
- `physics_diffusion_group_meeting_slides.tex`
- `physics_diffusion_group_meeting_slides.pdf`
- `physics_diffusion_group_meeting_slides_pympress.pdf`
- `physics_diffusion_group_meeting_speech.md`
- `artifacts/diffusion_physics_group_meeting/` 下的论文、源码和渲染检查产物

不会覆盖已有 PPT、论文或结果文件。

## 9. 分步实施与验证

1. **锁定论文证据**：核对作者、年份、期刊/会议、主公式和结果表。验证：官方页面、官方 PDF、论文源码三者一致。
2. **写幻灯片结构**：控制在约 14 页。验证：每页只有一个主要结论，整套可在 12--15 分钟讲完。
3. **写逐页讲稿**：每页约 40--90 秒，公式页先说直觉再说符号。验证：讲稿不依赖未展示的图或公式。
4. **编译两个 PDF**：普通版隐藏备注；Pympress 版右侧显示备注。验证：`pdfinfo` 页数和页面尺寸正确。
5. **视觉抽查**：将所有页或代表页渲染为 PNG，检查文字、公式、原论文图、脚注和备注是否截断。
6. **最终检查**：检查 LaTeX 日志中的 overfull/undefined、执行 `git diff --check`、列出实际命令和结果。

## 10. 风险、暂停条件与回退

- 若原图投影不可读，改为自绘简图或裁剪局部，不强行塞入整张图。
- 若讲稿使 Pympress 右侧页溢出，缩短口语稿并把补充内容移至 Markdown。
- 若公式符号与原文不一致，停止排版并回查论文源码，不凭记忆修补。
- 若 LaTeX 新增包需要联网安装或改变全局配置，优先删减依赖；不自动修改系统级配置。
- 回退方式：所有新增文件使用独立命名，删除新增文件即可恢复，不影响现有项目。

## 11. 人类检查点

- 交付后，人类需要做三件事：
  1. 确认封面姓名和日期；
  2. 用 Pympress 打开双屏 PDF，检查备注字体大小；
  3. 按 Markdown 讲稿彩排一次，根据实际时长决定是否跳过“理论推导细节”页。

## 12. 进度日志

- **已完成**：两篇论文原文核验；16 页 Beamer 演示稿；普通放映 PDF；Pympress 双屏 PDF；逐页中文念稿；原论文图和矢量对比表。
- **论文主线**：PFGM++（ICML 2023）负责解释 Poisson 电场如何重写生成轨迹；Speed--Accuracy Relations（Physical Review X 2025）负责解释非平衡热力学、熵产生和 Wasserstein 路径如何约束噪声日程。
- **最终编译**：两种版本均从当前源文件用 XeLaTeX 各编译两遍；普通版 16 页、`453.54 x 255.12 pt`；双屏版 16 页、`907.09 x 255.12 pt`。
- **最终机器验收**：普通版不含讲稿文本，双屏版含讲稿文本；中文字体和公式字体已嵌入；最终日志无 `LaTeX Error`、未定义控制序列或 `Overfull`。
- **最终视觉验收**：普通版 16 页全部渲染；检查基础页、PFGM++ 公式/实验页、PRX 理论/OT 页、总结页和参考文献页；双屏版抽查公式、实验、总结和参考文献页，未发现明显裁切或重叠。
- **独立讲稿**：`physics_diffusion_group_meeting_speech.md` 含 12—15 分钟版本、10/8 分钟删减路径和答疑提示。
- **仓库检查**：`git diff --check` 返回 0；未修改已有研究代码和 `期刊投稿目标清单.md`。
- **剩余人类检查点**：组会前用 Pympress 打开双屏 PDF 彩排一次，根据实际发言时长决定是否跳过理论细节页。
- **阻塞**：无。
