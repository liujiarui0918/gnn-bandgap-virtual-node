# Relaxed MP-HZW T-Series Plan

## Objective

Run a new MP-HZW matched bandgap experiment series using a less aggressive data-cleaning policy and a clean single-node vs dual-node virtual-node ablation.

The goal is to test whether the current conclusion still holds after restoring samples with large MP/HZW bandgap disagreement and large per-atom-volume disagreement:

- T4: single plain virtual-node ALIGNN is the strongest simple baseline.
- T5: dual plain virtual-node ALIGNN helps larger structures.
- T6/T7: size-aware routing or adaptive gating should preserve T4 on small systems and recover T5 on large systems.

## Dataset Policy

Input table:

`results/data_quality/structure_first_strict_180/mp_hzw_structure_matched_all.csv`

Remote expected table:

`/data/home/tg100025/gnn_work/results/data_quality/structure_first_strict_180/mp_hzw_structure_matched_all.csv`

Use MP bandgap as the training label, keeping the existing project convention for MP-label matched datasets.

### Keep

- Structure-matched MP-HZW rows.
- Rows with large `abs(mp_band_gap - hzw_band_gap)`.
- Rows with large `volume_per_atom_rel_diff`.

### Drop

1. Metal/non-metal contradiction by strict zero-gap status.

   Definition:

   ```text
   mp_zero = mp_band_gap <= 1e-6
   hzw_zero = hzw_band_gap <= 1e-6
   drop if mp_zero != hzw_zero
   ```

   Measured on the current matched table:

   ```text
   total rows                  92787
   MP zero / HZW nonzero        2935
   HZW zero / MP nonzero        5049
   zero/nonzero conflicts       7984
   ```

2. Atom-count relation mismatch.

   Keep only exact equal atom-count records:

   ```text
   atom_count_relation == equal
   hzw_num_atoms == mp_nsites
   ```

   Current matched table:

   ```text
   equal          92507
   mp_multiple      195
   hzw_multiple      85
   ```

Expected relaxed-clean size before final export safeguards:

```text
92787 - union(zero/nonzero conflict, atom-count non-equal) = 84551
```

The exact exported count may differ by rows missing from MP cache or failing structure reconstruction, and must be reported from `metadata.json`.

## Experiment Series

All models train for 50 epochs unless a smoke-test override is explicitly passed.

| Key | Name | Definition | Training |
| --- | --- | --- | --- |
| T1 | CGCNN baseline | Faithful CGCNN baseline | Train |
| T2 | ALIGNN baseline | Faithful ALIGNN baseline | Train |
| T3 | M3GNet baseline | Faithful M3GNet baseline | Train |
| T4 | ALIGNN single plain virtual node | ALIGNN + one plain virtual node | Train |
| T5 | ALIGNN dual plain virtual node | ALIGNN + two plain virtual nodes, no attention/gate/size loss | Train |
| T6 | Val-routed T4/T5 | Use T4 for small structures and T5 for large structures; threshold selected on validation set | No training |
| T7 | Adaptive gated dual plain virtual node | First plain virtual node always active; second plain virtual node controlled by size-aware gate | Train |

## T6 Routing Definition

T6 is not trained. It loads finished T4 and T5 checkpoints.

1. Run T4 and T5 on the validation split.
2. Sweep candidate `num_atoms` thresholds from validation `num_atoms`.
3. For each threshold:

   ```text
   if num_atoms <= threshold: use T4 prediction
   else: use T5 prediction
   ```

4. Select the threshold with the lowest validation MAE.
5. Apply that fixed threshold to test predictions.

Important: do not select the threshold on test.

## T7 Model Definition

T7 must be a new clean architecture, not the old S12 attention-based size-supervised model.

Required behavior:

- It is based on the plain dual virtual-node idea.
- Virtual node 1 is always active.
- Virtual node 2 is multiplied by a learned `slot2_enable` gate.
- The gate can explicitly see size features such as:

  ```text
  log1p(num_atoms)
  clipped_num_atoms / scale
  graph pooled feature
  ```

- Model outputs include:

  ```text
  band_gap
  slot2_enable
  slot2_gate_logit
  ```

This keeps compatibility with the existing `compute_loss` size supervision:

```text
size_target = sigmoid((num_atoms - threshold) / temperature)
L = L_gap + lambda_size * BCEWithLogits(slot2_gate_logit, size_target)
```

Recommended T7 loss:

```text
L = L_gap(pred, y)
  + lambda_size * BCE(gate, sigmoid((num_atoms - N*) / tau))
```

Where `N*` should be initialized from the T6 validation-selected threshold when running a final tuned T7. If T7 must run before T6, use a conservative default threshold such as 80 or 100 and report that it is a pre-routing prior.

## File Plan

Expected code/document changes:

- `docs/plans/2026-05-30-relaxed-mp-hzw-t-series.md`
- `export_mp_hzw_final_clean_dataset.py`
- `faithful_gnn/alignn_variants.py`
- `configs/mp/MP_T1_cgcnn_faithful.yaml`
- `configs/mp/MP_T2_alignn_faithful.yaml`
- `configs/mp/MP_T3_m3gnet_faithful.yaml`
- `configs/mp/MP_T4_alignn_virtualnode.yaml`
- `configs/mp/MP_T5_alignn_plain_dual_virtualnode.yaml`
- `configs/mp/MP_T6_alignn_valrouted_t4t5.yaml`
- `configs/mp/MP_T7_alignn_sizegated_plain_dual_virtualnode.yaml`
- `run_t_relaxed_matched_experiments.py`
- `scripts/slurm_t_relaxed_mp_queue.sh`
- optional start/monitor helper scripts
- focused tests for export flags, router dependencies, and T7 gate outputs

## Remote Paths

Recommended remote paths:

```text
PROJECT_ROOT=/data/home/tg100025/gnn
WORK_ROOT=/data/home/tg100025/gnn_work
MP_DATASET_DIR=/data/home/tg100025/gnn_work/data/mp_hzw_relaxed_clean_equal_atoms_mp
MP_RESULTS_DIR=/data/home/tg100025/gnn_work/results/mp_hzw_relaxed_clean_equal_atoms_mp_t
MP_CACHE_ROOT=/data/home/tg100025/gnn_work/cache/mp_hzw_relaxed_clean_equal_atoms_mp_t
MP_CGCNN_ROOT=/data/home/tg100025/gnn_work/data/mp_hzw_relaxed_clean_equal_atoms_mp_cgcnn
HZW_DATASET_DIR=/data/home/tg100025/gnn_work/data/mp_hzw_relaxed_clean_equal_atoms_hzw
HZW_RESULTS_DIR=/data/home/tg100025/gnn_work/results/mp_hzw_relaxed_clean_equal_atoms_hzw_t
HZW_CACHE_ROOT=/data/home/tg100025/gnn_work/cache/mp_hzw_relaxed_clean_equal_atoms_hzw_t
HZW_CGCNN_ROOT=/data/home/tg100025/gnn_work/data/mp_hzw_relaxed_clean_equal_atoms_hzw_cgcnn
```

## Slurm Rules

Before submitting:

```bash
/opt/slurm-22.05.2/bin/squeue -a
/opt/slurm-22.05.2/bin/sinfo -p tx4090
/opt/slurm-22.05.2/bin/scontrol show nodes | egrep "NodeName=|State=|AllocTRES="
```

Rationale: this Slurm cluster has `PrivateData=jobs,usage`, so `squeue -a` can hide other users' running jobs. Node allocation must be checked via `sinfo` and `scontrol show nodes`.

Recommended request:

```text
partition: tx4090
gres: gpu:1
cpus-per-task: 10
mem: 90G
time: 7 days
```

## Verification

Local verification:

```bash
python -m pytest tests/test_e11_router_and_e12_size_loss.py
python -m pytest tests/test_t7_sizegated_plain_dual_virtualnode.py
python run_t_relaxed_matched_experiments.py --help
```

Remote smoke verification before full queue:

```bash
python run_t_relaxed_matched_experiments.py \
  --skip_audit \
  --force_export \
  --epochs 1 \
  --experiments T4 T5 T6 T7
```

Full queue, 14 experiments in one Slurm job:

```bash
sbatch scripts/slurm_t_relaxed_mp_queue.sh
```

The Slurm script runs in this order:

```text
MP-label:  T1 T2 T3 T4 T5 T6 T7
HZW-label: T1 T2 T3 T4 T5 T6 T7
```

## Result Analysis

After completion:

1. Compare overall MAE/RMSE/R2/metal F1 for T1-T7.
2. Compare T4 vs T5 by `num_atoms` buckets:

   ```text
   <=10, 11-20, 21-40, 41-80, 81-160, >160
   ```

3. Report T6 selected validation threshold.
4. Report T7 `slot2_enable_mean` and size-bucketed `slot2_enable`.
5. Verify whether:

   ```text
   T6 or T7 <= T4 overall
   T5 improves over T4 on large structures
   T7 uses higher slot2_enable on larger structures
   ```

## Risks

- Restoring large bandgap-disagreement rows may increase label noise and weaken all models.
- Strict zero/nonzero conflict removes more rows than the old `mp_is_metal != hzw_is_metal` flag because it treats tiny nonzero gaps as nonmetal.
- T7 could collapse the second slot if `lambda_size` is too weak, or over-open it if `lambda_size` is too strong.
- T6 is an ensemble/router result, not a single trained model. It should be reported separately from trainable architectures.
