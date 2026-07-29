# T6 Top-k Huber Single Virtual Node Plan

## Objective

Replace the old T6/T7 dual-node routing/adaptive experiments with one cleaner single-virtual-node experiment:

- Keep the ALIGNN + plain single virtual node backbone from T4.
- Replace the final bandgap regression loss from MSE to Huber loss.
- Let the global token receive information only from top-k selected atom nodes when the structure size is in a configured atom-count interval.
- Stop using dual-node models in the default T-series queue.

## Experiment Set

Default queue after this change:

| Key | Experiment | Purpose |
| --- | --- | --- |
| T1 | CGCNN baseline | Historical baseline |
| T2 | ALIGNN baseline | Main faithful baseline |
| T3 | M3GNet baseline | Message-passing baseline |
| T4 | ALIGNN + plain single virtual node | Strong single-global-token baseline |
| T6 | ALIGNN + top-k single virtual node + Huber loss | Noise-robust global-token variant |

T5 dual plain virtual node remains selectable for historical comparison, but is not in the default queue.
Old T6/T7 are removed from the default runner mapping.

## T6 Design

### Backbone

T6 reuses the T4 path:

```text
ALIGNN encoder -> single virtual node message injection/update -> graph readout -> bandgap head
```

The difference is only in how the virtual node is updated and how the final regression loss is computed.

### Top-k Global Token Update

Original T4 virtual node update:

```text
global_token_update = mean(all node features)
```

New T6 virtual node update:

```text
k = clamp(ceil(0.2 * num_nodes), min=4, max=32)
scores = learnable_score(node_features)
selected_nodes = top_k(nodes by scores)
global_token_update = softmax_weighted_mean(selected_nodes)
```

Reasoning:

- Top-k is enabled for all structure sizes by default.
- Very small structures naturally fall back to nearly full selection because `k` is clipped by the actual atom count.
- Large structures still keep an upper bound through `topk_max_nodes: 32`, so the global token is not flooded by every atom.

### Huber Loss

Original T4 final loss:

```text
loss = MSE(predicted_bandgap, target_bandgap)
```

New T6 final loss:

```text
loss = SmoothL1Loss(predicted_bandgap, target_bandgap, beta=0.5)
```

Reasoning:

- The relaxed MP-HZW clean set intentionally keeps bandgap disagreements and volume disagreements.
- MSE over-weights large-label-disagreement samples.
- Huber keeps quadratic behavior near normal errors but becomes linear for large outliers, which should be more robust under noisy labels.

## Files

Implementation:

- `faithful_gnn/alignn_variants.py`
- `run_t_relaxed_matched_experiments.py`
- `configs/mp/MP_T6_alignn_topk_huber_virtualnode.yaml`
- `scripts/slurm_t_relaxed_mp_queue.sh`
- `summarize.py`

Tests:

- `tests/test_t6_topk_huber_virtualnode.py`
- `tests/test_t_relaxed_runner_config.py`

## Queue Handling

The currently running Slurm job was launched with the old in-memory T6/T7 mapping. Editing files on disk cannot change that running Python process. Therefore the safe procedure is:

1. Sync the patched code to the server.
2. Cancel the old Slurm job before it reaches old T6/T7.
3. Archive old MP result directories for old T6/T7 so summaries do not confuse new results.
4. Submit the updated queue.
5. Rely on resume behavior to skip already completed experiments.

## Verification

Local checks:

```bash
python -m pytest tests/test_t_relaxed_runner_config.py tests/test_t6_topk_huber_virtualnode.py -q -p no:cacheprovider
python -m py_compile run_t_relaxed_matched_experiments.py faithful_gnn/alignn_variants.py summarize.py
```

Remote checks:

```bash
python -m py_compile run_t_relaxed_matched_experiments.py faithful_gnn/alignn_variants.py summarize.py
bash -n scripts/slurm_t_relaxed_mp_queue.sh
squeue -u tg100025
sinfo -p tx4090
```

Success criteria:

- New generated T6 config uses `variant: topkvirtualnode`.
- New generated T6 config uses `final_gap_loss: huber`.
- Default queue does not include old T7.
- Default queue does not include dual-node T5 unless explicitly requested.
- Summary includes only selected experiments from the current queue.
