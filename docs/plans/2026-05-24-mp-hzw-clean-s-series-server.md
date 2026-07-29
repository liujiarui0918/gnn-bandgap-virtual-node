# 2026-05-24 MP-HZW Clean Matched 数据清洗与 S 系列重训计划

## 0. 目标

本轮目标不是继续在不确定数据上盲训，而是先把 MP 与鸿之微 HZW 的同源样本做成可解释、可复现、可审计的 clean matched 数据集，再在该数据集上从头复现实验矩阵。

本轮输出分三类：

1. **远程训练环境**：把本地项目、HZW 原始数据、必要 vendor/runtime 文件同步到新服务器 `/data/home/tg100025/gnn`，所有缓存和中间文件放到 `/data/home/tg100025/gnn_work` 或 `/tmp`，避免写爆系统盘。
2. **完整垃圾数据表**：对 HZW 与 MP 可匹配样本逐条比较结构和标签，输出每条垃圾数据的 `reason_code`、结构比较摘要、band gap 冲突摘要。
3. **S 系列重训**：在 clean matched 数据上从头跑 `S1 S2 S3 S4 S5 S7 S8 S10 S11 S12`，训练队列顺序执行，数据审计、MP 下载、监控并行执行。

## 1. 当前事实

- 新服务器 SSH 已可达，登录节点为 `login`，用户为 `tg100025`。
- 数据盘 `/data` 约 20T，剩余约 19T；`/tmp` 约 98G，剩余约 93G。
- 登录节点可用 `python3 3.10` 和 `pip`，但登录节点没有直接看到 `nvidia-smi`。
- 服务器有 Slurm，`sinfo` 显示 `tx4090` 分区存在 4090 GPU 节点。因此训练应通过 Slurm 申请 GPU 节点，而不是直接在登录节点跑。
- 本地 HZW 原始文件存在：`api_downloads/api_downloads/crystal_band_100000.jsonl/crystal_band_100000.jsonl.gz.tmp`。
- 该 HZW 文件虽然后缀包含 `.gz.tmp`，实际内容是普通 JSONL；项目的 `iter_json_records()` 会自动识别 gzip magic bytes，因此能正确读取。
- HZW 原始记录包含完整晶体结构字段：`structure.lattice` 与 `structure.sites`。
- 现有 `export_mp_hzw_matched_dataset.py` 会从 MP summary 拉取 `structure/band_gap/cbm/vbm/is_metal/formula_pretty/nsites/density`，但当前 manifest 训练目标使用 MP `band_gap`，只保留 HZW label 和元数据，没有完整保留 HZW 结构。

## 2. 远程目录规划

服务器目录：

| 路径 | 用途 |
| --- | --- |
| `/data/home/tg100025/gnn` | 项目代码主目录 |
| `/data/home/tg100025/gnn_work/data` | 训练数据、MP fetch cache、clean manifest |
| `/data/home/tg100025/gnn_work/cache` | ALIGNN/M3GNet/CGCNN 图缓存 |
| `/data/home/tg100025/gnn_work/results` | 所有 S 系列训练输出 |
| `/data/home/tg100025/gnn_work/logs` | Slurm、worker、monitor 日志 |
| `/data/home/tg100025/gnn_work/envs` | Conda/venv 环境 |
| `/tmp/gnn_$USER` | 临时 scratch，可用于安装/解压/短期中间文件 |

原则：

- 不把大缓存写入 `/`。
- 不在登录节点长期训练。
- 每个后台任务写独立日志，所有 PID/job id 写入 `gnn_work/run`。
- MP API key 只通过环境变量或权限受限的本地文件传入，不写进代码、文档、公开日志。

## 3. 同步策略

必须同步：

- `crystal_gnn/`
- `faithful_gnn/`
- `configs/`
- `scripts/`
- `tests/`
- `*.py` 队列、导出、监控脚本
- `.vendor/`
- `vendor_runtime_bundle.tar.gz`
- `requirements.txt`
- `bootstrap*.sh`
- `api_downloads/api_downloads/crystal_band_100000.jsonl/crystal_band_100000.jsonl.gz.tmp`
- `api_downloads/api_downloads/crystal_band_100000_meta.json`
- `docs/plans/2026-05-24-mp-hzw-clean-s-series-server.md`

可以不同步或低优先级同步：

- `.git/`
- `__pycache__/`
- `.pytest_cache/`
- 旧 PDF/PPT/LaTeX 中间文件
- 旧 `results/` 下的大模型 checkpoint 和旧缓存

同步方式优先级：

1. 如果本地有 `rsync`：用 `rsync -avz --delete`，配合 exclude。
2. 如果没有 `rsync`：用 `tar` 打包后 `scp`，远程解压到 `/data/home/tg100025/gnn`。

## 4. MP-HZW 完整垃圾数据表定义

### 4.1 审计输入

输入文件：

- HZW 原始 JSONL：包含 `source_id=mp-*`、`hc_id`、`formula`、`num_atoms`、`crystal_band[0].band_gap`、`structure`。
- MP summary fetch cache：从 Materials Project summary API 拉取 `material_id, band_gap, cbm, vbm, is_metal, formula_pretty, nsites, density, structure`。

审计对象：

- HZW 中 `source_id` 以 `mp-` 开头且 HZW band gap 合法的记录。
- MP summary 中能查到对应 `material_id` 的记录。

### 4.2 每条审计记录字段

完整审计表 `full_audit_table.csv` 字段：

| 字段 | 含义 |
| --- | --- |
| `sample_id` | 统一样本 id |
| `source_id` | MP material id |
| `hzw_hc_id` | 鸿之微 id |
| `split` | 后续固定 split，可在 clean export 后补齐 |
| `mp_formula` | MP formula |
| `hzw_formula` | HZW formula |
| `mp_reduced_formula` | MP reduced formula |
| `hzw_reduced_formula` | HZW reduced formula |
| `mp_nsites` | MP structure site 数 |
| `hzw_nsites` | HZW structure site 数 |
| `atom_count_relation` | `equal` / `mp_multiple` / `hzw_multiple` / `mismatch` |
| `mp_band_gap` | MP summary band gap |
| `hzw_band_gap` | HZW band gap |
| `abs_gap_diff` | 两者 band gap 绝对差 |
| `mp_is_metal` | MP 金属判断 |
| `hzw_is_metal` | HZW 金属判断 |
| `mp_cbm` | MP CBM |
| `mp_vbm` | MP VBM |
| `mp_density` | MP density |
| `hzw_density` | HZW structure density |
| `mp_volume_per_atom` | MP 体积/原子 |
| `hzw_volume_per_atom` | HZW 体积/原子 |
| `volume_per_atom_rel_diff` | 体积/原子相对差 |
| `structure_match_level` | `exact_like` / `compatible` / `supercell_like` / `composition_only` / `mismatch` / `not_tested` |
| `structure_matcher_fit` | pymatgen StructureMatcher 是否匹配 |
| `is_garbage` | 是否剔除 |
| `reason_code` | 机器可读垃圾原因，可多值用 `;` 分隔 |
| `reason_text` | 人可读解释 |

垃圾表 `full_garbage_table.csv` 是 `full_audit_table.csv` 中 `is_garbage=true` 的子集。

### 4.3 垃圾原因分类

结构类：

- `missing_mp_record`：MP 查不到该 `source_id`。
- `missing_hzw_structure`：HZW 结构缺失。
- `missing_mp_structure`：MP 结构缺失。
- `bad_hzw_structure`：HZW 结构无法被 pymatgen 解析。
- `bad_mp_structure`：MP 结构无法被 pymatgen 解析。
- `composition_mismatch`：MP 与 HZW reduced composition 不一致。
- `atom_count_mismatch`：原子数不相等，且不是清晰整数倍超胞关系。
- `structure_match_failed`：composition 可比但 StructureMatcher 不匹配。
- `volume_per_atom_mismatch`：体积/原子差异过大，说明结构不是同一套晶体表示。

标签类：

- `mp_metal_hzw_nonmetal`：MP 判为金属或 gap 近 0，但 HZW gap 明显大于阈值。
- `mp_nonmetal_hzw_metal`：MP gap 明显非零，但 HZW gap 近 0。
- `same_class_gap_diff_gt_1ev`：金属/非金属类别一致，但 band gap 差值超过 1 eV。
- `invalid_mp_gap`：MP gap 缺失、负数、非有限数或超过合理上限。
- `invalid_hzw_gap`：HZW gap 缺失、负数、非有限数或超过合理上限。

### 4.4 结构比较策略

分层比较，避免对 9 万条样本都做最重匹配：

1. 解析 MP/HZW `Structure`。
2. 计算 composition、reduced formula、site 数、密度、体积/原子。
3. 如果 composition 不同，直接判 `composition_mismatch`。
4. 如果 site 数相同，优先用 `StructureMatcher(primitive_cell=True, scale=True)` 判断是否为同一结构。
5. 如果 site 数为整数倍且 composition reduced formula 相同，用 `StructureMatcher(primitive_cell=True, attempt_supercell=True, scale=True)` 尝试判断是否为 primitive/supercell 表示差异。
6. 如果 StructureMatcher 因大结构耗时或失败，退化为 `composition + volume_per_atom + atom_count_relation` 的保守判断，并把 `structure_match_level` 标为 `composition_only` 或 `not_tested`。

### 4.5 Clean 数据纳入标准

样本进入 clean matched 训练集需要同时满足：

- HZW 与 MP 都有合法结构。
- HZW 与 MP reduced composition 一致。
- 结构比较为 `exact_like`、`compatible` 或 `supercell_like`。
- 不存在金属/非金属冲突。
- `abs_gap_diff <= 1.0 eV`，或者两者都为金属且 gap 都接近 0。
- MP band gap 合法。

训练目标默认使用 MP summary `band_gap`，因为本轮目标是“从 MP 中还原 HZW 来源样本并在 MP 标签体系下重训”。同时在 manifest 中保留 `hzw_band_gap`，用于后续解释。

## 5. 输出文件

数据审计输出：

- `results/data_quality/mp_hzw_full_audit_table.csv`
- `results/data_quality/mp_hzw_full_garbage_table.csv`
- `results/data_quality/mp_hzw_clean_source_ids.csv`
- `results/data_quality/mp_hzw_garbage_summary.json`
- `results/data_quality/mp_hzw_garbage_summary.md`

Clean 数据输出：

- `data/mp_hzw_clean_matched_s/manifest.jsonl`
- `data/mp_hzw_clean_matched_s/metadata.json`
- `data/mp_hzw_clean_matched_s/splits.json`
- `data/mp_hzw_clean_matched_s/cgcnn/{train,val,test}/`

训练输出：

- `results/mp_hzw_clean_matched_s/S1_CGCNN_baseline/`
- `results/mp_hzw_clean_matched_s/S2_ALIGNN_baseline/`
- `results/mp_hzw_clean_matched_s/S3_M3GNet_baseline/`
- `results/mp_hzw_clean_matched_s/S4_ALIGNN_virtualnode/`
- `results/mp_hzw_clean_matched_s/S5_ALIGNN_transformer/`
- `results/mp_hzw_clean_matched_s/S7_ALIGNN_attentive_virtualnode/`
- `results/mp_hzw_clean_matched_s/S8_ALIGNN_multi_virtualnode/`
- `results/mp_hzw_clean_matched_s/S10_ALIGNN_sizeaware_virtualnode_active/`
- `results/mp_hzw_clean_matched_s/S11_ALIGNN_valrouted_e4e8/`
- `results/mp_hzw_clean_matched_s/S12_ALIGNN_sizesupervised_virtualnode/`
- `results/mp_hzw_clean_matched_s/comparison_summary.csv`

## 6. S 系列实验映射

| S 实验 | 对应旧实验 | 模型含义 | 配置来源 |
| --- | --- | --- | --- |
| `S1` | `E1` | CGCNN baseline | `configs/mp/MP_E1_cgcnn_faithful.yaml` |
| `S2` | `E2` | ALIGNN baseline | `configs/mp/MP_E2_alignn_faithful.yaml` |
| `S3` | `E3` | M3GNet baseline | `configs/mp/MP_E3_m3gnet_faithful.yaml` |
| `S4` | `E4` | ALIGNN + 朴素超节点 | `configs/mp/MP_E4_alignn_virtualnode.yaml` |
| `S5` | `E5` | ALIGNN + Transformer | `configs/E5_alignn_transformer_faithful.yaml` |
| `S7` | `E7` | ALIGNN + 单 attentive 超节点 | `configs/mp/MP_E7_alignn_attentive_virtualnode.yaml` |
| `S8` | `E8` | ALIGNN + 双 attentive 超节点 | `configs/mp/MP_E8_alignn_multi_virtualnode.yaml` |
| `S10` | `E10 active` | 尺度感知 active gate 双槽位 | `configs/E10_alignn_sizeaware_virtualnode_active.yaml` |
| `S11` | `E11 router` | 验证集阈值路由 E4/E8 | `configs/E11_alignn_valrouted_e4e8.yaml` |
| `S12` | `E12` | size-supervised dual slot，size 只进 loss | `configs/mp/MP_E12_alignn_sizesupervised_virtualnode.yaml` |

注意：

- `S11` 依赖 `S4` 和 `S8` 的 checkpoint，必须排在 `S4/S8` 完成之后。
- `S10` 和 `S12` 是独立可训练模型，不依赖其他 checkpoint。
- 单卡 GPU 不并行跑多个训练进程；训练顺序执行，数据审计、下载、监控并行。

## 7. 后台 worker 分工

### Worker A: remote-env

任务：

- 创建 `/data/home/tg100025/gnn_work/{data,cache,results,logs,run,envs}`。
- 检查 Slurm GPU 分区。
- 建立 Python/Conda 环境。
- 验证 `torch.cuda.is_available()` 必须在 GPU 作业内为 true。

成功标志：

- `gnn_work/logs/env_verify.log` 中能看到 CUDA 可用。

### Worker B: data-sync

任务：

- 同步项目代码与 HZW 原始数据到 `/data/home/tg100025/gnn`。
- 同步完成后在远程检查 HZW 文件大小和首行 JSON 可解析。

成功标志：

- 远程 HZW 原始文件存在，大小约 648 MB。
- 远程 `python3 - <<PY` 能读出第一条记录，并确认有 `structure.sites`。

### Worker C: mp-fetch-and-audit

任务：

- 用 MP API 拉取同源 `source_id` 对应的 MP summary。
- 生成完整审计表、垃圾表、clean source id。
- 导出 clean matched S 数据集。

成功标志：

- `results/data_quality/mp_hzw_full_garbage_table.csv` 存在且非空。
- `data/mp_hzw_clean_matched_s/manifest.jsonl` 存在。
- `metadata.json` 中记录 accepted/garbage/clean 计数。

### Worker D: train-queue

任务：

- 在 Slurm GPU 节点按顺序跑 `S1 S2 S3 S4 S5 S7 S8 S10 S11 S12`。
- 每个实验写独立日志。
- OOM 自动降低 batch size 重试。
- 非 OOM 错误写 `failure.json` 并停止队列，等待主 agent debug。

成功标志：

- 每个 S 实验输出 `metrics.json`。
- 最终生成 `comparison_summary.csv`。

### Worker E: monitor

任务：

- 每 30 分钟检查 Slurm job、GPU、磁盘、当前日志尾部、metrics 数量。
- 发现报错时抓取 traceback 和最近 200 行日志到 `logs/last_error_report.md`。
- 如果是已知 OOM，允许训练队列自降 batch retry。
- 如果是未知错误，不写死规则，交给主 agent 根据错误信息 debug 后重跑。

成功标志：

- `gnn_work/logs/monitor.log` 持续更新。
- 队列完成后自动退出，不再每半小时轮询。

## 8. 执行顺序

1. 本地补齐脚本：完整审计脚本、S 系列队列脚本、远程 Slurm 启动脚本。
2. 同步项目和 HZW 原始数据到新服务器。
3. 远程创建工作目录和 Python 环境。
4. Slurm GPU 作业内验证 CUDA。
5. 启动 MP fetch + full audit。
6. 审计通过后导出 clean matched S 数据。
7. 启动 S 系列训练队列。
8. 启动 monitor，每半小时检查，直到队列完成或未知错误阻塞。
9. 训练完成后同步 `results/mp_hzw_clean_matched_s/comparison_summary.csv` 和垃圾表回本地。

## 9. 验证命令

本地验证：

```bash
python -m pytest tests/test_e9_sizeaware_virtualnode.py tests/test_e11_router_and_e12_size_loss.py
python export_mp_hzw_matched_dataset.py --help
python run_mp_hzw_matched_experiments.py --help
```

远程验证：

```bash
ssh -i tg100025.pem tg100025@124.220.81.245 'bash -lc "df -h / /data /tmp; sinfo -o \"%P %D %t %G %c %m %N\"; python3 --version"'
```

训练监控：

```bash
ssh -i tg100025.pem tg100025@124.220.81.245 'bash -lc "tail -n 80 /data/home/tg100025/gnn_work/logs/monitor.log; squeue -u $USER"'
```

## 10. 风险与处理

| 风险 | 处理 |
| --- | --- |
| 登录节点无 GPU | 使用 Slurm `tx4090` 分区申请 GPU 节点 |
| MP API 下载慢或失败 | 使用 fetch cache，失败 chunk 重试，不重复下载已缓存记录 |
| StructureMatcher 过慢 | 对标签冲突和结构可疑样本做重匹配，其他样本做轻量结构摘要；超时降级并标记 |
| CGCNN 小文件过多导致慢 | CGCNN CIF 写到数据盘或本地 scratch，必要时减少 worker |
| ALIGNN OOM | 队列脚本已有 batch divisor retry |
| S11 缺 checkpoint | 队列强制 S11 等 S4/S8 完成后再运行 |
| 数据清洗阈值争议 | 审计表保留原始数值和 reason_code，阈值可复算 |

## 11. 本轮不做的事

- 不把 MP API key 写入代码或文档。
- 不在登录节点直接训练。
- 不并行启动多个 GPU 训练抢同一张卡。
- 不删除本地旧结果和旧报告。
- 不把结构不一致但标签看起来正常的样本硬塞进 clean 数据集。
