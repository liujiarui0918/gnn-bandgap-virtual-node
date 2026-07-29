# Top-k Warmup and Global Broadcast Decay Plan

## Judgment

Both proposed experiments are reasonable.

The current best model is T7:

```text
ALIGNN + single plain virtual node + Huber loss
```

The two new experiments keep this strong baseline and change only one information-flow mechanism at a time:

| New key | Change | Direction |
| --- | --- | --- |
| T9 | Add warmup/soft mixing to top-k node-to-global collection | node -> global |
| T10 | Add decay to global-to-node broadcast | global -> node |

This is a clean ablation because both use the same Huber loss and the same single-global-token backbone.

## T9: Top-k Warmup With Soft Merge

### Motivation

Hard top-k from epoch 1 is unstable because the node representations and the top-k scoring layer are not trained yet. Early top-k selection can pick random nodes and poison the global token.

### Design

Let:

```text
mean_pool = mean(all atom node features)
topk_pool = softmax-weighted mean(top-k scored atom node features)
alpha(epoch) in [0, 1]
```

Use:

```text
global_update_pool = (1 - alpha) * mean_pool + alpha * topk_pool
```

Schedule:

```text
epoch 1-10:  alpha = 0
epoch 11-20: alpha linearly increases from 0 to 1
epoch 21-50: alpha = 1
```

Config:

```yaml
training:
  final_gap_loss: huber
  final_gap_huber_beta: 0.5
model:
  variant: warmuptopkvirtualnode
  topk_fraction: 0.2
  topk_min_nodes: 4
  topk_max_nodes: 32
  topk_warmup_epochs: 10
  topk_ramp_epochs: 10
```

### Interpretation

Compare:

- T7 vs T9: does warmup top-k improve over Huber-only?
- T6 vs T9: does warmup fix hard top-k hurting the combined model?
- T8 vs T9: does Huber + warmup top-k beat MSE + hard top-k?

## T10: Global Broadcast Decay

### Motivation

The plain virtual node broadcasts global context to every atom node at every layer:

```text
node = node + global_token
```

This helps early convergence by providing global information, but late in training it may inject global noise into local atom representations.

### Design

Scale the broadcast:

```text
node = node + beta(epoch) * global_token
```

Schedule:

```text
epoch 1-10:  beta = 1.0
epoch 11-50: beta linearly decays from 1.0 to 0.25
```

Config:

```yaml
training:
  final_gap_loss: huber
  final_gap_huber_beta: 0.5
model:
  variant: decayvirtualnode
  broadcast_decay_start_epoch: 10
  broadcast_decay_end_epoch: 50
  broadcast_min_scale: 0.25
```

### Interpretation

Compare:

- T7 vs T10: does reducing late global broadcast noise help?
- T9 vs T10: is the better fix on node-to-global collection or global-to-node broadcast?

## Implementation Plan

### Model changes

File:

```text
faithful_gnn/alignn_variants.py
```

Add model state support:

- Add `set_training_epoch(epoch, total_epochs)` methods for relevant virtual-node variants.
- Add `ALIGNNWarmupTopKVirtualNodeRegressor`.
- Add `ALIGNNDecayVirtualNodeRegressor`.
- Register variants:
  - `warmuptopkvirtualnode`
  - `decayvirtualnode`

### Training loop changes

File:

```text
faithful_gnn/alignn_runner.py
```

Before each train/val epoch, if the model supports it:

```python
model.set_training_epoch(epoch, total_epochs)
```

For test, set it to final epoch behavior:

```python
model.set_training_epoch(total_epochs, total_epochs)
```

### Config changes

Add:

```text
configs/mp/MP_T9_alignn_warmup_topk_huber_virtualnode.yaml
configs/mp/MP_T10_alignn_decay_huber_virtualnode.yaml
```

### Runner changes

File:

```text
run_t_relaxed_matched_experiments.py
```

Add:

```text
T9  -> T9_ALIGNN_warmup_topk_huber_virtualnode
T10 -> T10_ALIGNN_decay_huber_virtualnode
```

Default queue becomes:

```text
T1 T2 T3 T4 T6 T7 T8 T9 T10
```

Existing completed metrics skip automatically, so the submitted run only trains T9/T10 on MP and HZW.

### Slurm changes

File:

```text
scripts/slurm_t_relaxed_mp_queue.sh
```

Update default `T_EXPERIMENTS`.

### Tests

Add tests covering:

- warmup top-k alpha schedule
- warmup top-k returns mean pool at alpha 0
- warmup top-k differs from pure mean when alpha 1 and top-k selection is active
- decay broadcast scale schedule
- generated configs for T9/T10 use Huber and correct variants

## Remote Execution

After local verification:

1. Sync changed files to server.
2. Run remote `py_compile` and `bash -n`.
3. Check `squeue`, `sinfo`, and node allocation.
4. Submit Slurm job.
5. Confirm it skips completed T1/T2/T3/T4/T6/T7/T8 and starts MP T9.
