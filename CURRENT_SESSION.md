# Current Session Handoff

Date: 2026-05-24

This file preserves the current project state so a new agent/session can continue without relying on chat history.

## Current Objective

Finish MP-HZW data cleaning and train the S-series experiments on the final MP-label dataset:

`S1 S2 S3 S4 S5 S7 S8 S10 S11 S12`

The final training dataset is based on strict MP-HZW structure matching, then filtered by:

- `mp_is_metal == hzw_is_metal`
- `abs(mp_band_gap - hzw_band_gap) <= 0.5 eV`
- `volume_per_atom_rel_diff <= 0.08`

## Important Security Note

Do not write API keys, passwords, or private key contents into project files. Remote commands in this file assume the local SSH key file already exists.

## Remote Server

Remote project:

```bash
/data/home/tg100025/gnn
```

Remote work root:

```bash
/data/home/tg100025/gnn_work
```

SSH pattern:

```bash
ssh -i tg100025.pem tg100025@124.220.81.245
```

## Final Clean Dataset

Remote dataset:

```bash
/data/home/tg100025/gnn_work/data/mp_hzw_final_clean_matched_mp
```

Remote CGCNN CIF root:

```bash
/data/home/tg100025/gnn_work/data/mp_hzw_final_clean_matched_mp_cgcnn
```

Local metadata mirror:

```text
results/data_quality/final_clean_mp/metadata.json
```

Dataset metadata:

- Strict structure-matched rows: `92,787`
- Dropped metal/nonmetal mismatches: `7,252`
- Dropped bandgap difference `> 0.5 eV`: `2,631`
- Dropped volume-per-atom relative difference `> 0.08`: `1,153`
- Final accepted records: `81,751`
- Split sizes: `65,400 / 8,175 / 8,176`
- Label source: `Materials Project band_gap`
- Mean MP-HZW absolute gap difference after filtering: `0.048 eV`
- Max MP-HZW absolute gap difference after filtering: `0.4996 eV`
- Max volume-per-atom relative difference after filtering: `0.079995`

## Structure-First Audit

Strict 180-second StructureMatcher output:

```text
results/data_quality/structure_first_strict_180/
```

Remote strict audit:

```bash
/data/home/tg100025/gnn_work/results/data_quality/structure_first_strict_180
```

Final strict structure audit counts:

- Total auditable MP-HZW pairs: `95,462`
- Strictly structure-matched: `92,787`
- Structure-unmatched: `39`
- Missing MP records: `2,636`
- StructureMatcher timeout: `0`

Important evidence tables:

- `mp_hzw_structure_matched_all.csv`
- `mp_hzw_structure_unmatched_all.csv`
- `mp_hzw_matched_mp_metal_hzw_nonmetal.csv`
- `mp_hzw_matched_mp_nonmetal_hzw_metal.csv`
- `mp_hzw_matched_same_class_gap_diff_gt_1ev.csv`
- `mp_hzw_matched_volume_per_atom_rel_diff_gt_0p08.csv`
- `mp_hzw_matched_bandgap_consistent.csv`

Interpretation:

Structure correspondence and label consistency are separate. The structure-matched table proves MP and HZW structures correspond; label conflict tables prove that the two data sources can still disagree on bandgap/metallicity for corresponding structures.

## Training Status

Old queue on earlier clean dataset:

- `S1_CGCNN_baseline` completed.
- `S3_M3GNet_baseline` completed.
- ALIGNN-family jobs failed originally because `sklearn` was not visible under `PYTHONNOUSERSITE=1`.

Dependency issue was fixed by linking/importing `scikit-learn` into the active environment. Verified command:

```bash
PYTHONNOUSERSITE=1 /data/home/tg100025/gnn_work/envs/gnn-faithful/bin/python -c "import sklearn; from jarvis.core.atoms import pmg_to_atoms; print('dependency_ok', sklearn.__version__)"
```

Expected output includes:

```text
dependency_ok 1.7.2
```

## Current Blocker

Slurm currently refuses any GPU job for this account.

Rechecked on 2026-05-25: the blocker is still active. A minimal 4090 test job with
`--gres=gpu:1 --cpus-per-task=1 --mem=4G -t 00:05:00` was rejected before entering
the queue.

Observed error:

```text
AssocMaxNodePerJobLimit
Batch job submission failed: Job violates accounting/QOS policy
```

Account association showed:

```text
MaxTRESPerJob=gres/gpu=0,node=0
```

Full association line from the 2026-05-25 recheck:

```text
tg100025|tg100025|tx4090|normal|6|0|50|gres/gpu=2,node=1|gres/gpu=0,node=0
```

Even a minimal GPU test job fails:

```bash
sbatch -p tx4090 --gres=gpu:1 --cpus-per-task=1 --mem=4G -t 00:10:00 --wrap=hostname
```

This is a platform/account/QOS issue, not a code or data issue.

## Final Training Scripts

Local scripts:

```text
scripts/slurm_s_final_mp_queue.sh
scripts/start_s_final_mp_queue_remote.sh
```

Remote scripts are synced to:

```bash
/data/home/tg100025/gnn/scripts/slurm_s_final_mp_queue.sh
/data/home/tg100025/gnn/scripts/start_s_final_mp_queue_remote.sh
```

When GPU permission is restored, start final training with:

```bash
cd /data/home/tg100025/gnn
scripts/start_s_final_mp_queue_remote.sh
```

This uses:

```bash
DATASET_DIR=/data/home/tg100025/gnn_work/data/mp_hzw_final_clean_matched_mp
RESULTS_DIR=/data/home/tg100025/gnn_work/results/mp_hzw_final_clean_matched_mp_s
CACHE_ROOT=/data/home/tg100025/gnn_work/cache/mp_hzw_final_clean_matched_mp_s
CGCNN_ROOT=/data/home/tg100025/gnn_work/data/mp_hzw_final_clean_matched_mp_cgcnn
```

It runs:

```text
S1 S2 S3 S4 S5 S7 S8 S10 S11 S12
```

## Monitoring Commands

After a job starts:

```bash
cat /data/home/tg100025/gnn_work/run/s_final_mp_queue.jobid
squeue -u tg100025
```

GPU:

```bash
JOB=$(cat /data/home/tg100025/gnn_work/run/s_final_mp_queue.jobid)
srun --jobid=$JOB --overlap nvidia-smi
```

Logs:

```bash
JOB=$(cat /data/home/tg100025/gnn_work/run/s_final_mp_queue.jobid)
tail -f /data/home/tg100025/gnn_work/logs/slurm-gnn_s_final_mp-${JOB}.out
tail -f /data/home/tg100025/gnn_work/logs/s_final_mp_queue_monitor.log
ls -lah /data/home/tg100025/gnn_work/results/mp_hzw_final_clean_matched_mp_s/queue_logs
```

System:

```bash
free -h
df -h
htop
```

## Key Local Files Added/Changed In This Session

- `export_mp_hzw_structure_first_tables.py`
- `export_mp_hzw_final_clean_dataset.py`
- `scripts/slurm_s_final_mp_queue.sh`
- `scripts/start_s_final_mp_queue_remote.sh`
- `docs/plans/2026-05-24-mp-hzw-structure-first-audit.md`
- `results/data_quality/structure_first_strict_180/`
- `results/data_quality/final_clean_mp/metadata.json`

## Recommended Next Step

1. Check platform/account GPU permission.
2. Once `sbatch` accepts GPU jobs, run:

```bash
cd /data/home/tg100025/gnn
scripts/start_s_final_mp_queue_remote.sh
```

3. Monitor until all S-series experiments complete.
