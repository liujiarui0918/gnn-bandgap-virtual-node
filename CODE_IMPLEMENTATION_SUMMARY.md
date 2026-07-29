# 代码思路总览

这份总结对应当前项目里最终用于跑完六组实验的 **faithful 复现链路**，不是最早那套原型代码。

项目里实际上有两层代码：

- 旧原型层：`crystal_gnn/*`、`train.py`、`preprocess.py`
- 最终实验层：`export_official_dataset.py`、`run_faithful_experiment.py`、`faithful_gnn/*`、`supervise_training.py`、`training_watchdog.py`、`remote_training_monitor.py`、`summarize.py`

如果要给师兄讲“现在真正怎么跑起来的”，建议只讲第二层。

## 1. 项目目标

项目目标是在鸿之微 `crystal_band_100000` 数据上，完成一套尽量忠实论文范式的晶体图神经网络 band gap 预测实验，并做六组对比：

- E1：CGCNN baseline
- E2：ALIGNN baseline
- E3：M3GNet baseline
- E4：ALIGNN + Virtual Node
- E5：ALIGNN + Transformer
- E6：ALIGNN + Physics Constraint

核心任务是：

- 主任务：`band_gap` 回归
- 辅助信息：`cbm`、`vbm`
- 其中只有 E6 显式把 `cbm/vbm` 放进训练损失

注意一件事：

- E4、E5、E6 是三组 **独立增强实验**
- 它们都是“在 ALIGNN baseline 上单独改一处”
- 不是 E4 -> E5 -> E6 逐层叠加

## 2. 总体代码架构

整体数据流和训练流可以概括成 6 步：

1. 从鸿之微原始 JSONL 导出统一官方数据集
2. 固定随机种子做 train/val/test 划分
3. 把同一份数据适配给 CGCNN / ALIGNN / M3GNet 三类模型
4. 通过统一入口脚本按配置文件调度不同 runner
5. 每个实验单独产出日志、指标、预测结果
6. 由 supervisor/watchdog/monitor 负责编排、巡检、自愈和汇总

最关键的文件职责如下：

| 文件 | 作用 |
| --- | --- |
| `export_official_dataset.py` | 从原始 JSONL 导出统一数据集、固定划分 |
| `run_faithful_experiment.py` | 单实验统一入口，按 `runner` 分发 |
| `faithful_gnn/common.py` | 公共工具：seed、日志、metrics、manifest 读取 |
| `faithful_gnn/cgcnn_runner.py` | CGCNN faithful 训练逻辑 |
| `faithful_gnn/alignn_runner.py` | ALIGNN 及其变体训练逻辑 |
| `faithful_gnn/alignn_variants.py` | E2/E4/E5/E6 的模型结构定义 |
| `faithful_gnn/m3gnet_runner.py` | M3GNet faithful 训练逻辑 |
| `supervise_training.py` | 顺序编排 preprocess -> E1..E6 -> summarize |
| `training_watchdog.py` | 远端 Linux 看门狗，自愈和异常打包 |
| `remote_training_monitor.py` | 本地轮询远端状态，发现未知错误后升级处理 |
| `summarize.py` | 汇总所有 `metrics.json` 到总表 |

## 3. 数据预处理与官方划分

这部分在 `export_official_dataset.py`。

它做的事不是“复杂特征工程”，而是把原始鸿之微数据整理成一个所有模型都能复用的统一中间层。

### 3.1 输入数据

- 原始文件：`crystal_band_100000.jsonl` 或 `.jsonl.gz`
- 每条样本里包含：
  - 晶体结构 `structure`
  - 公式 `formula`
  - 带隙信息 `crystal_band`
  - 其中包含 `band_gap`、`cbm`、`vbm`

### 3.2 清洗逻辑

导出时主要做了这些过滤：

- `crystal_band` 为空的样本丢弃
- `band_gap` 缺失的样本丢弃
- `band_gap < 0` 或 `band_gap > 15` 的异常样本丢弃
- `pymatgen.Structure.from_dict(...)` 解析失败的样本丢弃

这里的目标是先保证“结构合法 + 标签合法”，不要把脏数据带进后面的图构建和训练。

### 3.3 统一中间格式

导出后会生成：

- `manifest.jsonl`
- `metadata.json`
- `splits.json`
- `structures/*.cif`
- `cgcnn/train|val|test/id_prop.csv`
- `cgcnn/train|val|test/*.cif`
- `cgcnn/train|val|test/atom_init.json`

其中 `manifest.jsonl` 是最关键的统一中间层，每条记录大致包含：

- `sample_id`
- `source_id`
- `formula`
- `density`
- `band_gap`
- `cbm`
- `vbm`
- `structure`
- `split`

后面的 ALIGNN 和 M3GNet 都是从这个 manifest 再转自己需要的图格式。

### 3.4 固定划分

划分逻辑是固定随机种子 `42`，按：

- train：0.8
- val：0.1
- test：0.1

清洗后有效样本数是：

- 总计：95,497
- train：76,397
- val：9,549
- test：9,551

这样做的意义是：所有模型和增强实验共享完全相同的数据划分，结果可直接横向对比。

## 4. 单实验统一入口

这部分在 `run_faithful_experiment.py`。

思路很简单：不让每个模型各写一套入口脚本，而是统一使用：

```bash
python run_faithful_experiment.py --config configs/xxx.yaml --output_dir results/xxx
```

入口脚本会做三件事：

1. 读取 YAML 配置
2. 接收命令行 override，比如 `batch_size`、`epochs`、`lr`
3. 按 `config["runner"]` 分发到具体 runner

分发规则是：

- `cgcnn` -> `faithful_gnn.cgcnn_runner.run_experiment`
- `alignn` -> `faithful_gnn.alignn_runner.run_experiment`
- `m3gnet` -> `faithful_gnn.m3gnet_runner.run_experiment`

这个设计的好处是：

- 六组实验都走同一套执行接口
- supervisor 只要记住“配置文件 + 输出目录”即可
- 后续加新实验时，只需要新增配置或新增 variant

## 5. 公共训练基础设施

这部分在 `faithful_gnn/common.py`。

它把所有 runner 共用的逻辑抽出来了：

- `seed_everything`：统一随机种子
- `setup_output_dir`：创建输出目录和日志文件
- `load_manifest`：读取 `manifest.jsonl` 并恢复 split
- `save_json`：保存 `history.json`、`metrics.json`
- `save_predictions`：保存 `predictions.csv`
- `device_from_config`：自动选 CPU/GPU
- `peak_gpu_gb`：记录峰值显存

所以三类 runner 的结构很像：

1. 加载数据
2. 构建模型
3. 训练 50 个 epoch
4. 用最优 checkpoint 测试
5. 记录 metrics、history、predictions

## 6. 三个基线模型的实现思路

## 6.1 E1：CGCNN baseline

代码在 `faithful_gnn/cgcnn_runner.py`。

这部分尽量贴官方 CGCNN 的用法：

- 数据集类直接用官方 `CIFData`
- 批处理直接用官方 `collate_pool`
- 模型直接用官方 `CrystalGraphConvNet`

训练逻辑上做了两件比较关键的事：

- 用训练集随机抽样构建 `Normalizer`
- 模型回归的是归一化后的 target，最终评估时再反归一化

也就是说，CGCNN 的主线是：

1. 读取 `cgcnn/train|val|test`
2. 从 CIF 生成原子特征和邻接特征
3. 用卷积层做局部原子环境聚合
4. 输出 band gap
5. 损失函数是归一化目标上的 MSE

它的特点是结构清楚、计算轻，但建模能力主要停留在局部邻域。

## 6.2 E2：ALIGNN baseline

代码主体在 `faithful_gnn/alignn_runner.py`，结构定义在 `faithful_gnn/alignn_variants.py`。

这里的 faithful 思路是直接走 ALIGNN 的核心范式：

- 原子图 `graph`
- 线图 `line_graph`
- 在线图里显式建模键角信息

具体流程是：

1. 从 `manifest.jsonl` 读结构
2. 通过 `jarvis` 和 ALIGNN 官方数据管线转成图
3. 用 `get_train_val_loaders(...)` 构建 dataloader
4. 模型部分使用 `OfficialALIGNN`
5. 输出单头 `band_gap`

ALIGNN baseline 比 CGCNN 强的核心原因是：

- 不只看“原子和原子之间连不连”
- 还看“键与键之间的角度关系”
- 因此能显式编码局部几何结构

训练上有一个重要细节：

- 不做早停截断训练
- 所有实验都强制跑满 50 个 epoch
- 但最终测试时会加载“验证集 MAE 最优”的 checkpoint

所以：

- `epochs_ran = 50`
- `best_epoch` 只是说明“最佳模型出现在第几轮”
- 不是提前停掉了

## 6.3 E3：M3GNet baseline

代码在 `faithful_gnn/m3gnet_runner.py`。

这部分是用 `matgl` 实现 M3GNet 风格的三体几何建模。

主线流程是：

1. 从 `manifest.jsonl` 读所有结构
2. 用 `Structure2Graph` 把结构转成图
3. 用 `MGLDataset` 缓存图数据
4. 构建 line graph，保留三体相互作用信息
5. 前向时补齐 lattice、PBC offset、原子坐标等几何量
6. 用 `M3GNet` 做 band gap 回归

这部分和 CGCNN、ALIGNN 的主要差异在于：

- 它更强调三体几何关系
- 不是简单的原子邻居聚合
- 对晶体几何的表达更强，但实现也更依赖图缓存和几何张量准备

## 7. 三个增强实验的实现思路

三组增强都在 `faithful_gnn/alignn_variants.py` 里切换，入口是：

```python
build_alignn_model(model_config)
```

它根据 `variant` 选择：

- `baseline`
- `virtualnode`
- `transformer`
- `physconstraint`

所以 E4/E5/E6 的本质都是“共享 ALIGNN 基础编码器，再在后面各自改一段”。

## 7.1 E4：ALIGNN + Virtual Node

对应类：`ALIGNNVirtualNodeRegressor`

设计思路是给每个图额外加一个“全局虚拟节点”，让模型除了局部消息传递之外，还能维护一份图级别的全局状态。

它的机制可以通俗理解成：

- 每个晶体图都多了一个“全局记事本”
- 每一层前，记事本内容广播给所有节点
- 然后再把所有节点当前的平均信息写回记事本
- 最后一层时，把“局部图表示”和“全局记事本表示”拼起来预测

对应实现动作是：

- `virtual_node_embedding`：初始化全局节点
- `_update_virtual_node(...)`：
  - 把虚拟节点向量 broadcast 给该图所有原子节点
  - 对节点特征按图做 `scatter_mean`
  - 再更新虚拟节点
- 每个 ALIGNN 层和 GCN 层前都做一次这个更新
- 最终 `pooled + virtual_node` 拼接后过 MLP 输出

这组实验的目标很明确：

- 补足纯局部消息传递对全局上下文感知不足的问题

## 7.2 E5：ALIGNN + Transformer

对应类：`ALIGNNTransformerRegressor`

设计思路是：

- 先用 ALIGNN 做局部几何编码
- 再把节点序列送进 TransformerEncoder
- 让节点之间通过全局自注意力进一步交互

实现上分成几步：

1. 先用 `ALIGNNEncoder` 得到节点特征
2. 读取 `frac_coords`
3. 对分数坐标做 `sinusoidal_encoding`
4. 位置编码经线性层投到 hidden space
5. 节点特征补齐成 batch 内统一长度
6. 送入 `nn.TransformerEncoder`
7. 同时保留：
   - `local_pooled`：ALIGNN 的局部池化结果
   - `global_pooled`：Transformer 后的全局池化结果
8. 最后把两者拼接回归 band gap

所以 E5 不是“纯 Transformer 模型”，而是：

- `ALIGNN 编码器 + 位置编码 + TransformerEncoder + 融合头`

这也是为什么它更像“后挂一个全局注意力模块”，而不是从底层彻底重写成晶体图 Transformer。

## 7.3 E6：ALIGNN + Physics Constraint

对应类：`ALIGNNPhysicsRegressor`，损失函数在 `faithful_gnn/alignn_runner.py` 的 `compute_loss(...)`。

它的设计思路是把 band gap 回归从单任务，改成“主任务 + 辅助任务 + 物理一致性约束”的联合训练。

模型结构上是：

- 一个共享 ALIGNN 编码器
- 三个输出头：
  - `band_gap_head`
  - `cbm_head`
  - `vbm_head`

损失函数是：

```text
L = L_main + lambda_aux * L_aux + lambda_phys * L_phys + lambda_nonneg * L_nonneg
```

四项分别表示：

- `L_main`：band gap 主任务误差
- `L_aux`：cbm 和 vbm 辅助回归误差
- `L_phys`：约束预测满足 `band_gap = cbm - vbm`
- `L_nonneg`：惩罚负带隙预测

这里最重要的代码细节是后面补上的有效性 mask：

```text
valid_edge_mask = (abs(cbm) > eps) or (abs(vbm) > eps)
```

含义是：

- 如果一条样本的 `cbm=0` 且 `vbm=0`
- 很可能只是“没有有效边带信息”或者“占位值”
- 那么这条样本不能拿去算辅助损失和物理一致性损失

所以 E6 实际采用的是：

- band gap 主损失对所有样本都算
- `cbm/vbm` 辅助损失只在有效样本上算
- 物理一致性损失也只在有效样本上算

这个设计比直接把所有样本都塞进去更稳，因为不会被无效零值污染。

## 8. 训练编排与自动化运行

## 8.1 Supervisor

代码在 `supervise_training.py`。

它的作用是把整套流程串成一个固定 phase 序列：

1. `preprocess`
2. `E1_CGCNN_baseline`
3. `E2_ALIGNN_baseline`
4. `E3_M3GNet_baseline`
5. `E4_ALIGNN_virtualnode`
6. `E5_ALIGNN_transformer`
7. `E6_ALIGNN_physconstraint`
8. `summarize`

每个 phase 都定义了：

- 启动命令
- 完成标志文件
- 对应配置
- 对应输出目录

Supervisor 主要做三件事：

- 顺序执行 phase
- 记录状态到 `logs/supervisor_state.json`
- 把事件写到 `logs/supervisor.log`

它本质上就是一个“非交互训练总控脚本”。

## 8.2 Watchdog

代码在 `training_watchdog.py`。

这是远端 Linux 上的看门狗，职责比 supervisor 更高一层，主要解决“训练过程中断了怎么办”。

它做的事包括：

- 保证只启动一个 watchdog 实例
- 检查当前 phase 进度
- 检查是否有重复 supervisor 进程
- 检查磁盘可用空间
- 清理已完成 ALIGNN 实验的本地缓存
- 把 ALIGNN 的 LMDB cache 放到数据盘 `/root/lanyun-tmp/gnn_alignn_cache`
- 对已知故障做 remediation
- 对未知故障打包成 `watchdog_unknown_failure.json`

也就是说，watchdog 不是训练模型本身，而是“盯训练的人”。

## 8.3 Remote Monitor

代码在 `remote_training_monitor.py`。

这是本地 Windows 侧每 30 分钟去 SSH 巡检一次远端状态的脚本。

它会通过 `paramiko` 抓取远端：

- watchdog 状态
- supervisor 状态
- 当前 runner 进程
- 当前 phase 的日志尾部
- GPU 使用情况
- 是否已经生成 `comparison_summary.csv`
- 是否出现未知错误 bundle

如果一切正常，它就只是继续等下一轮。

如果出现未知错误，它会：

- 把故障现场归档到本地 `monitoring/unknown_failures`
- 记录 incident
- 准备 Codex triage prompt 去做进一步 debug

所以从设计上说：

- 已知错误可以靠 watchdog 的硬编码策略处理
- 未知错误可以通过 monitor 抓现场，再交给 Codex 做调试

这也是整个“半小时巡检 + 报错后自动 debug 再重跑”的核心实现思路。

## 9. 输出产物怎么组织

最终每个实验目录下都会有一套固定产物：

- `train.log`
- `history.json`
- `metrics.json`
- `predictions.csv`
- `best_model.pt`

总汇总文件是：

- `results/comparison_summary.csv`

所以整个产物层次是：

```text
data/
  official_dataset/
    manifest.jsonl
    metadata.json
    splits.json
    structures/
    cgcnn/

results/
  E1_CGCNN_baseline/
  E2_ALIGNN_baseline/
  E3_M3GNet_baseline/
  E4_ALIGNN_virtualnode/
  E5_ALIGNN_transformer/
  E6_ALIGNN_physconstraint/
  comparison_summary.csv

logs/
  supervisor.log
  supervisor_state.json
  watchdog.log
  watchdog_state.json
```

## 10. 六组实验最终结果

最终结果已经整理进最终汇报文件里，数值如下：

| 实验 | 模型 | MAE | RMSE | R2 | Metal F1 | Best Epoch |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| E1 | CGCNN | 0.302169 | 0.530287 | 0.884731 | 0.806929 | 48 |
| E2 | ALIGNN | 0.231432 | 0.471325 | 0.908939 | 0.862117 | 50 |
| E3 | M3GNet | 0.268041 | 0.553444 | 0.874444 | 0.871087 | 42 |
| E4 | ALIGNN + Virtual Node | 0.220077 | 0.474616 | 0.907663 | 0.883811 | 48 |
| E5 | ALIGNN + Transformer | 0.237450 | 0.491768 | 0.900869 | 0.876873 | 46 |
| E6 | ALIGNN + Physics Constraint | 0.221208 | 0.453924 | 0.915539 | 0.864048 | 50 |

可以直接这样解读：

- 最强 baseline 是 ALIGNN
- E4 在 MAE 和 Metal F1 上最好
- E6 在 RMSE 和 R2 上最好
- E5 没有超过 E2，说明当前这版 Transformer 接法没有带来稳定收益

再强调一次：

- E4 = `ALIGNN + Virtual Node`
- E5 = `ALIGNN + Transformer`
- E6 = `ALIGNN + Physics Constraint`
- 三者是平行 ablation，不是叠加模型

## 11. 给师兄讲时最推荐的表达方式

可以直接按下面这个逻辑讲：

第一层，先讲工程主线：

- 我先把鸿之微原始 JSONL 导成统一 manifest 和固定划分
- 然后做了一个统一实验入口
- 同一份数据分别喂给 CGCNN、ALIGNN、M3GNet
- 所有实验统一产出 metrics、prediction 和日志

第二层，再讲模型主线：

- CGCNN 是最传统的原子图卷积 baseline
- ALIGNN 在此基础上加入了 line graph，显式建模键角信息，所以是最强 baseline
- M3GNet 走的是三体几何建模路线

第三层，讲三个增强点：

- E4 是给 ALIGNN 加全局虚拟节点，补足全局信息汇聚
- E5 是在 ALIGNN 后面叠一个 TransformerEncoder 做全局注意力
- E6 是把任务改成 band gap + cbm/vbm 多头预测，并加入物理一致性约束

第四层，讲工程保障：

- 用 supervisor 串起来全部 phase
- 用 watchdog 做远端自愈和缓存/磁盘管理
- 用 remote monitor 每半小时 SSH 看一眼状态
- 遇到未知错误时抓现场并升级到 Codex 调试

最后再落到结论：

- ALIGNN 是最强 baseline
- Virtual Node 和 Physics Constraint 都有效
- 当前这版 Transformer 没带来提升，后面还需要继续改

## 12. 当前代码的优点与局限

### 优点

- 数据划分统一，横向对比公平
- 三类模型共用统一入口，实验管理清楚
- E4/E5/E6 都是在同一个 ALIGNN 基线上做单因素改动，ablation 比较干净
- 有 supervisor、watchdog、monitor 三层自动化，远程训练不容易失控
- E6 已经补上有效性 mask，避免无效 `cbm/vbm=0` 样本污染 physics loss

### 局限

- 项目里同时保留了旧原型代码和最终 faithful 代码，第一次看代码的人容易混淆
- E5 目前是“ALIGNN 后挂 Transformer”，还不是完整的晶体图 Transformer 重设计
- 当前主任务仍然只有 band gap，`cbm/vbm` 只在 E6 里用作辅助监督
- 本轮结果说明 E5 方案暂时无效，后续需要继续做结构和训练策略上的针对性改进

## 13. 一句话版本

一句话概括这套代码：

> 这套代码先把鸿之微晶体带隙数据导成统一官方划分，再用统一入口分别跑 CGCNN、ALIGNN、M3GNet 和三个基于 ALIGNN 的独立增强实验，同时用 supervisor/watchdog/remote monitor 保证远端训练可以顺序执行、定时巡检、异常自愈和结果汇总。
