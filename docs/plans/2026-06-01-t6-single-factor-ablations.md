# T6 Single-factor Ablation Plan

## Objective

The current T6 combines two changes over T4:

```text
T4: ALIGNN + plain single virtual node + MSE
T6: ALIGNN + top-k single virtual node + Huber loss
```

To identify which component contributes the gain, add two single-factor ablations:

| Key | Model | Loss | Purpose |
| --- | --- | --- | --- |
| T4 | Plain single virtual node | MSE | Reference |
| T7 | Plain single virtual node | Huber | Huber-only ablation |
| T8 | Top-k single virtual node | MSE | Top-k-only ablation |
| T6 | Top-k single virtual node | Huber | Combined model |

Run T7 and T8 on both label sources:

- MP label dataset
- HZW label dataset

This adds four training runs total.

## T7: Huber-only

T7 changes only the final regression loss:

```yaml
model.variant: virtualnode
training.final_gap_loss: huber
training.final_gap_huber_beta: 0.5
```

Expected interpretation:

- If T7 improves over T4, label-noise robustness from Huber is useful.
- If T7 is close to T6, top-k contributes little.

## T8: Top-k-only

T8 changes only the global token update:

```yaml
model.variant: topkvirtualnode
topk_fraction: 0.2
topk_min_nodes: 4
topk_max_nodes: 32
```

It intentionally keeps the default final gap loss, which is MSE.

Expected interpretation:

- If T8 improves over T4, sparse global-token message collection is useful.
- If T8 is close to T6, Huber contributes little.

## Queue

Default T-series queue:

```text
T1 T2 T3 T4 T6 T7 T8
```

Existing completed metrics for T1/T2/T3/T4/T6 will be skipped. The new submitted run should only train T7 and T8 for MP, then T7 and T8 for HZW.

## Verification

Local:

```bash
python -m pytest tests/test_t_relaxed_runner_config.py tests/test_t6_topk_huber_virtualnode.py -q -p no:cacheprovider
python -m py_compile run_t_relaxed_matched_experiments.py faithful_gnn/alignn_variants.py summarize.py
```

Remote:

```bash
python -m py_compile run_t_relaxed_matched_experiments.py faithful_gnn/alignn_variants.py summarize.py
bash -n scripts/slurm_t_relaxed_mp_queue.sh
squeue -u tg100025
sinfo -p tx4090
```
