# 晶体图神经网络实验落地方案 (PRD)

> 目标：在鸿之微 crystal_band 数据集上完成 3 个基线 + 3 个魔改实验，50 epoch，单卡 3090，一天内出结果。

---

## 1. 数据集概况

| 项目 | 详情 |
|------|------|
| 来源 | 鸿之微平台 API (`api-sxs.cloud.hzwtech.com`) |
| 文件 | `crystal_band_100000.jsonl.gz` (~619 MB 压缩) |
| 样本数 | 100,000 条晶体结构 |
| 格式 | 每行一条 JSON，包含 pymatgen `Structure` 对象 + `crystal_band` 字段 |
| 预测目标 | **band_gap** (回归任务，单位 eV) |
| 辅助字段 | `cbm`, `vbm`, `ef_er_mi`, `gap_type`, `space_group`, `formula`, `density` 等 |
| 结构信息 | lattice (a/b/c/alpha/beta/gamma) + sites (元素种类 + 分数坐标 + 笛卡尔坐标) |

### 1.1 数据预处理流程

```
crystal_band_100000.jsonl.gz
  ├─ 解压 → 逐行读取 JSON
  ├─ 提取 pymatgen Structure 对象 + band_gap 标签
  ├─ 过滤异常值 (band_gap < 0 或 > 15 eV 的样本)
  ├─ 划分: train / val / test = 80% / 10% / 10%  (固定随机种子 42)
  └─ 按模型要求转换图结构 (见各实验节)
```

### 1.2 数据规模估算

- 使用全部 10 万条数据
- 若显存不足，可降至 5 万条 (前 50k)，但优先尝试全量

---

## 2. 硬件与资源约束

| 资源 | 规格 |
|------|------|
| GPU | NVIDIA RTX 3090 × 1 (24 GB VRAM) |
| 目标时间 | **24 小时内完成全部 6 组实验** |
| Epoch | 每组实验统一 **50 epoch** |
| 框架 | PyTorch + PyTorch Geometric / DGL |

### 2.1 时间预算分配

| 阶段 | 预估时间 |
|------|----------|
| 数据预处理 + 图构建缓存 | ~1 小时 |
| 第一轮：3 个基线并行 (2+1) | ~6-8 小时 |
| 第二轮：3 个魔改并行 (2+1) | ~8-10 小时 |
| 结果汇总与可视化 | ~1 小时 |
| **合计** | **~16-20 小时** |

### 2.2 并行策略

3090 单卡 24GB 无法同时跑两个大模型，但可以通过以下策略加速：

- **串行中的微并行**：数据预处理用 CPU 多进程，GPU 只跑训练
- **图缓存复用**：CGCNN 和 ALIGNN 共享原子图构建，只需额外构建 line graph
- **Batch size 自适应**：根据模型显存占用动态调整，最大化 GPU 利用率
- **使用 `tmux` 多窗口**：一个实验结束后自动启动下一个 (脚本编排)

**推荐执行顺序**（按显存从小到大，失败风险从低到高）：

```
第一轮 (基线):
  [窗口1] CGCNN (显存 ~4-6 GB, 最快)
  → 完成后立即启动 CGCNN+Transformer 魔改

  [窗口1 续] ALIGNN (显存 ~8-12 GB)
  → 完成后立即启动 ALIGNN+VirtualNode 魔改

  [窗口1 续] M3GNet (显存 ~10-14 GB)
  → 完成后立即启动 M3GNet+PhysicsConstraint 魔改
```

实际上单卡只能串行训练，用自动化脚本链式执行即可。

---

## 3. 实验总览

| 编号 | 实验名称 | 类型 | 基于 | 核心改动 | 论文依据 |
|------|----------|------|------|----------|----------|
| E1 | CGCNN-Baseline | 基线 | - | 无 | Xie & Grossman, 2018 |
| E2 | ALIGNN-Baseline | 基线 | - | 无 | Choudhary & DeCost, 2021 |
| E3 | M3GNet-Baseline | 基线 | - | 无 | Chen & Ong, 2022 |
| E4 | ALIGNN-Transformer | 魔改 | ALIGNN | +Global Graph Attention | ComFormer / MGT |
| E5 | CGCNN-VirtualNode | 魔改 | CGCNN | +Virtual Node 层级聚合 | HC-GNN / VGNN |
| E6 | M3GNet-PhysConstraint | 魔改 | M3GNet | +能量守恒物理约束损失 | Hamiltonian GNN / MACE |

---

## 4. 各实验详细设计

### 4.1 E1: CGCNN-Baseline

**模型概述**：Crystal Graph Convolutional Neural Network，将晶体构建为原子图（节点=原子，边=邻居原子对），用高斯距离展开作边特征，多层图卷积后全局池化做回归。

**实现方案**：使用 `matbench` 或 CGCNN 官方 PyTorch 实现。

| 超参数 | 值 |
|--------|-----|
| 截断半径 | 8.0 Å |
| 最大邻居数 | 12 |
| 高斯展开数 | 41 |
| 原子嵌入维度 | 64 |
| 图卷积层数 | 3 |
| 隐藏层维度 | 128 |
| Batch size | 256 |
| 学习率 | 1e-3 (Adam) |
| 调度器 | ReduceLROnPlateau (patience=5, factor=0.5) |
| Epoch | 50 |
| 预估显存 | ~4-6 GB |
| 预估单 epoch | ~2-3 min → 总计 ~2 小时 |

**评估指标**：MAE (eV), RMSE (eV), R²

---

### 4.2 E2: ALIGNN-Baseline

**模型概述**：Atomistic Line Graph Neural Network，在原子图之上再构建 line graph（边→节点，角度→边），通过交替在原子图和线图上做消息传递来捕获键角信息。

**实现方案**：使用 `alignn` 官方 Python 包 (`pip install alignn`)。

| 超参数 | 值 |
|--------|-----|
| 截断半径 | 8.0 Å |
| 最大邻居数 | 12 |
| 原子特征维度 | 64 |
| 边特征维度 | 64 |
| ALIGNN 层数 | 4 |
| GCN 层数 (后续) | 4 |
| 隐藏层维度 | 256 |
| Batch size | 64 (line graph 较耗显存) |
| 学习率 | 1e-3 (AdamW, weight_decay=1e-5) |
| 调度器 | OneCycleLR |
| Epoch | 50 |
| 预估显存 | ~8-12 GB |
| 预估单 epoch | ~5-6 min → 总计 ~4.5 小时 |

---

### 4.3 E3: M3GNet-Baseline

**模型概述**：Materials 3-body Graph Network，引入三体交互项（原子 i-j-k 的角度信息），使用等变的球谐基函数展开，适合势能面和材料性质预测。

**实现方案**：使用 `matgl` 包 (`pip install matgl`)，M3GNet 预训练模型做 fine-tune 或从头训练。

| 超参数 | 值 |
|--------|-----|
| 截断半径 | 5.0 Å |
| 三体截断 | 4.0 Å |
| 最大 l (球谐) | 3 |
| 最大 n (径向基) | 3 |
| 隐藏维度 | 128 |
| 交互层数 | 3 |
| Batch size | 32 (三体信息显存开销大) |
| 学习率 | 1e-3 (Adam) |
| 调度器 | ReduceLROnPlateau |
| Epoch | 50 |
| 预估显存 | ~10-14 GB |
| 预估单 epoch | ~6-8 min → 总计 ~5.5 小时 |

> **注意**：若从预训练权重 fine-tune，学习率降至 1e-4，可能收敛更快。

---

### 4.4 E4: ALIGNN-Transformer (方向一：Transformer/长程注意力)

**论文依据**：ComFormer (完备图表示 + Transformer) + MGT (local/line/global 三图 + 多头注意力)

**魔改思路**：在 ALIGNN 的原子图-线图双图架构之上，**增加一个 Global Graph + Multi-Head Self-Attention 层**，让模型能捕获长程相互作用。

**具体改动**：

```
原版 ALIGNN:
  AtomGraph ←→ LineGraph  (交替消息传递 × N 层)
  → Global Pooling → MLP → band_gap

魔改 ALIGNN-Transformer:
  AtomGraph ←→ LineGraph  (交替消息传递 × N 层)
  → [新增] Global Self-Attention Block (2层 Transformer Encoder)
      - 将所有原子节点特征作为 token 序列
      - 加入可学习的周期位置编码 (lattice-aware positional encoding)
      - Multi-Head Attention (4 heads, dim=256)
      - FFN + LayerNorm + Residual
  → Global Pooling → MLP → band_gap
```

**关键实现细节**：
- 位置编码：使用分数坐标 (a,b,c) 经 sinusoidal 编码注入，类似 ComFormer 的 lattice representation
- Attention mask：无 (全连接，让模型自行学习长程/短程权重)
- Transformer 参数量控制：2 层 × 4 头 × 256 dim ≈ 额外 1.3M 参数

| 超参数 | 值 |
|--------|-----|
| ALIGNN 部分 | 同 E2 |
| Transformer 层数 | 2 |
| Attention heads | 4 |
| FFN 维度 | 512 |
| Dropout | 0.1 |
| Batch size | 48 (略小于 E2) |
| Epoch | 50 |
| 预估显存 | ~12-16 GB |
| 预估单 epoch | ~7-8 min → 总计 ~6 小时 |

**期望观察**：
- 对含长程静电作用的体系（如离子晶体）band_gap 预测精度提升
- Attention map 可视化：是否关注到远距离原子对

---

### 4.5 E5: CGCNN-VirtualNode (方向二：虚拟节点/层级图)

**论文依据**：HC-GNN (层级消息传递) + VGNN (虚拟节点承载全局信息)

**魔改思路**：在 CGCNN 图中**引入一个 Virtual Node (VN)**，该节点与所有真实原子节点相连，每层卷积后做双向消息传递：VN 汇聚全局信息 → 广播回各原子节点。同时引入简单的**基序层级**：按配位多面体聚类产生 super-node。

**具体改动**：

```
原版 CGCNN:
  AtomGraph (nodes=atoms, edges=neighbors)
  → GraphConv × 3 → Global Mean Pool → MLP → band_gap

魔改 CGCNN-VirtualNode:
  AtomGraph + 1 个 Virtual Node (VN)
  → 每层 GraphConv 后:
      1. 正常原子间消息传递
      2. VN ← Σ(所有原子特征) / N  (全局汇聚)
      3. 每个原子 ← 原子特征 + VN 特征  (全局广播)
  → GraphConv+VN × 4 层
  → [VN 最终特征 || Global Mean Pool] concat
  → MLP → band_gap
```

**关键实现细节**：
- Virtual Node 初始化为可学习向量
- 汇聚用 mean aggregation，广播用 add + LayerNorm
- 这是 OGB 中广泛验证的 VN 方案，实现简单但有效
- 层级 super-node 作为可选增强：按 Voronoi 配位环境聚类

| 超参数 | 值 |
|--------|-----|
| CGCNN 部分 | 同 E1，层数增至 4 |
| VN 维度 | 128 (与隐藏层同) |
| VN 聚合方式 | Mean |
| VN 广播方式 | Add + LayerNorm |
| Batch size | 256 |
| Epoch | 50 |
| 预估显存 | ~5-7 GB |
| 预估单 epoch | ~2.5-3 min → 总计 ~2.5 小时 |

**期望观察**：
- Over-squashing 缓解：大晶胞体系精度提升
- VN 特征可视化：学到的全局表示是否有物理意义

---

### 4.6 E6: M3GNet-PhysConstraint (方向三：物理约束)

**论文依据**：Hamiltonian GNN (物理输入表示) + MACE (高阶等变 + 物理约束)

**魔改思路**：在 M3GNet 训练中**引入物理约束正则化损失**，利用数据集中 CBM/VBM 信息构建辅助约束，同时在消息传递中引入简化版高阶交互项。

**具体改动**：

```
原版 M3GNet:
  Loss = MSE(pred_band_gap, true_band_gap)

魔改 M3GNet-PhysConstraint:
  多任务输出头:
    - head_bandgap: 预测 band_gap
    - head_cbm: 预测 CBM (导带底)
    - head_vbm: 预测 VBM (价带顶)

  物理约束损失:
    L_main   = MSE(pred_bandgap, true_bandgap)
    L_aux    = MSE(pred_cbm, true_cbm) + MSE(pred_vbm, true_vbm)
    L_phys   = MSE(pred_bandgap, pred_cbm - pred_vbm)  ← 自洽约束
    L_nonneg = ReLU(-pred_bandgap)²                     ← band_gap ≥ 0

    Loss = L_main + 0.3 * L_aux + 0.5 * L_phys + 0.1 * L_nonneg
```

**关键实现细节**：
- 多任务头共享 backbone (M3GNet encoder)，各自独立 MLP head
- `L_phys` 强制 band_gap = CBM - VBM 的自洽性
- `L_nonneg` 约束预测值非负（物理先验）
- 对 `gap_type == "-"` (金属，band_gap=0) 的样本，额外加权确保正确分类

| 超参数 | 值 |
|--------|-----|
| M3GNet 部分 | 同 E3 |
| 辅助任务权重 λ_aux | 0.3 |
| 自洽约束权重 λ_phys | 0.5 |
| 非负约束权重 λ_nonneg | 0.1 |
| Batch size | 32 |
| Epoch | 50 |
| 预估显存 | ~12-15 GB |
| 预估单 epoch | ~7-8 min → 总计 ~6 小时 |

**期望观察**：
- 金属/非金属分类准确率提升
- band_gap 预测值不再出现负值
- CBM-VBM 自洽性提高 → 物理可解释性增强

---

## 5. 统一评估框架

### 5.1 评估指标

| 指标 | 说明 |
|------|------|
| **MAE** (eV) | 主指标，与文献可比 |
| **RMSE** (eV) | 衡量大误差敏感度 |
| **R²** | 整体拟合质量 |
| 金属分类 F1 | band_gap=0 vs >0 的二分类 F1 |
| 训练时间 | 总训练时间 (小时) |
| 显存峰值 | nvidia-smi 记录 (GB) |

### 5.2 统一训练配置

```yaml
# 所有实验共用
seed: 42
train_ratio: 0.8
val_ratio: 0.1
test_ratio: 0.1
epochs: 50
early_stopping_patience: 10  # 可选，防止过拟合
save_best_model: true        # 按 val MAE 保存最优
log_interval: 1              # 每 epoch 记录
device: cuda:0
num_workers: 4               # DataLoader
pin_memory: true
mixed_precision: true         # AMP fp16 加速，节省显存
```

### 5.3 日志与检查点

```
results/
├── E1_CGCNN_baseline/
│   ├── config.yaml
│   ├── train.log
│   ├── best_model.pt
│   ├── metrics.json          # {mae, rmse, r2, ...}
│   └── learning_curve.png
├── E2_ALIGNN_baseline/
├── E3_M3GNet_baseline/
├── E4_ALIGNN_transformer/
├── E5_CGCNN_virtualnode/
├── E6_M3GNet_physconstraint/
└── comparison_summary.csv    # 汇总对比表
```

---

## 6. 执行时间线 (单卡串行)

```
  时间轴 (小时)
  0        2        4        6        8       10       12       14       16       18       20
  |--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
  [== 数据预处理 ==]
           [=== E1: CGCNN ===]
                    [===== E5: CGCNN+VN =====]
                              [======= E2: ALIGNN ========]
                                                [========= E4: ALIGNN+Trans =========]
                                                                    [====== E3: M3GNet ======]
                                                                                  [======= E6: M3GNet+Phys =======]
  ^                                                                                              ^
  开始                                                                                          ~20h 完成
```

**执行顺序逻辑**：
1. 同一基线的 baseline 先跑，紧接着跑其魔改版，方便对比
2. CGCNN 系列最轻量，先跑；M3GNet 系列最重，最后跑
3. 每个实验结束自动触发下一个 (脚本链式调用)

---

## 7. 自动化运行脚本框架

```bash
#!/bin/bash
# run_all_experiments.sh
# 在 3090 服务器上一键串行执行全部 6 组实验

set -e
export CUDA_VISIBLE_DEVICES=0

echo "===== [$(date)] 数据预处理 ====="
python preprocess.py --data_path data/crystal_band_100000.jsonl \
                     --output_dir data/processed/ \
                     --seed 42

echo "===== [$(date)] E1: CGCNN Baseline ====="
python train.py --model cgcnn --config configs/E1_cgcnn.yaml \
                --output_dir results/E1_CGCNN_baseline/

echo "===== [$(date)] E5: CGCNN + VirtualNode ====="
python train.py --model cgcnn_vn --config configs/E5_cgcnn_vn.yaml \
                --output_dir results/E5_CGCNN_virtualnode/

echo "===== [$(date)] E2: ALIGNN Baseline ====="
python train.py --model alignn --config configs/E2_alignn.yaml \
                --output_dir results/E2_ALIGNN_baseline/

echo "===== [$(date)] E4: ALIGNN + Transformer ====="
python train.py --model alignn_trans --config configs/E4_alignn_trans.yaml \
                --output_dir results/E4_ALIGNN_transformer/

echo "===== [$(date)] E3: M3GNet Baseline ====="
python train.py --model m3gnet --config configs/E3_m3gnet.yaml \
                --output_dir results/E3_M3GNet_baseline/

echo "===== [$(date)] E6: M3GNet + PhysConstraint ====="
python train.py --model m3gnet_phys --config configs/E6_m3gnet_phys.yaml \
                --output_dir results/E6_M3GNet_physconstraint/

echo "===== [$(date)] 生成对比报告 ====="
python summarize.py --results_dir results/ --output results/comparison_summary.csv

echo "===== 全部完成! ====="
```

---

## 8. 显存优化备案

若某个实验 OOM，按以下优先级降级：

| 优先级 | 措施 | 影响 |
|--------|------|------|
| 1 | 开启 AMP (fp16) | 显存减半，精度几乎无损 |
| 2 | 降低 batch size (÷2) | 训练变慢，精度不变 |
| 3 | gradient accumulation | 等效大 batch，速度略降 |
| 4 | 减少隐藏维度 (÷2) | 模型容量下降 |
| 5 | 减少数据量至 50k | 最后手段 |

---

## 9. 期望结果与分析框架

### 9.1 结果对比表 (模板)

| 实验 | MAE (eV) | RMSE (eV) | R² | 金属F1 | 时间 (h) | 显存 (GB) |
|------|----------|-----------|-----|--------|----------|-----------|
| E1 CGCNN | - | - | - | - | - | - |
| E2 ALIGNN | - | - | - | - | - | - |
| E3 M3GNet | - | - | - | - | - | - |
| E4 ALIGNN+Trans | - | - | - | - | - | - |
| E5 CGCNN+VN | - | - | - | - | - | - |
| E6 M3GNet+Phys | - | - | - | - | - | - |

### 9.2 分析维度

1. **基线间对比**：CGCNN vs ALIGNN vs M3GNet → 验证"键角信息"和"三体信息"的增益
2. **魔改 vs 基线**：每个魔改与其对应基线对比 → 验证三种改进方向的有效性
   - E4 vs E2: Transformer 长程注意力的增益
   - E5 vs E1: 虚拟节点全局信息传递的增益
   - E6 vs E3: 物理约束的增益
3. **误差分析**：按晶体体系 (金属/半导体/绝缘体) 分组统计误差
4. **效率分析**：精度提升 vs 额外计算开销的 trade-off

### 9.3 可视化输出

- 6 条 learning curve (train/val MAE vs epoch)
- 预测值 vs 真实值散点图 (parity plot)
- 误差分布直方图
- (E4) Attention map 热力图示例
- (E6) 物理约束前后负值预测比例对比

---

## 10. 依赖环境 (服务器安装)

```bash
# 基础环境
conda create -n gnn python=3.10 -y
conda activate gnn

# PyTorch + CUDA
pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118

# 图神经网络框架
pip install torch-geometric
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html

# 材料科学工具
pip install pymatgen matgl alignn jarvis-tools

# 通用工具
pip install numpy pandas scikit-learn matplotlib tqdm tensorboard pyyaml
```

---

## 11. 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| M3GNet 显存 OOM | 中 | 实验延迟 | batch=16 + gradient accumulation |
| 数据解压后过大 (>50GB) | 低 | 磁盘不足 | 流式读取，不全量加载内存 |
| ALIGNN line graph 构建慢 | 高 | 预处理耗时 | 预计算缓存到磁盘 |
| 50 epoch 未收敛 | 中 | 结果不理想 | 用 learning curve 判断，关注趋势而非绝对值 |
| band_gap=0 样本过多 (数据不平衡) | 中 | 回归偏差 | 可选加权采样或分桶评估 |

---

## 12. 交付物清单

- [ ] 数据预处理脚本 (`preprocess.py`)
- [ ] 6 个实验配置文件 (`configs/E1-E6.yaml`)
- [ ] 统一训练入口 (`train.py`)
- [ ] 3 个魔改模型代码 (`models/alignn_transformer.py`, `models/cgcnn_virtualnode.py`, `models/m3gnet_physconstraint.py`)
- [ ] 自动化执行脚本 (`run_all_experiments.sh`)
- [ ] 结果汇总脚本 (`summarize.py`)
- [ ] 最终对比报告 (`results/comparison_summary.csv` + 可视化图)
