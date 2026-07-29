# E9 Size-Aware Virtual Node Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and train an `E9` ALIGNN variant that combines the stability of `E4` with a size-aware gated second virtual node.

**Architecture:** Implement a new `ALIGNNSizeAwareVirtualNodeRegressor` in the existing ALIGNN variant file. The primary virtual node follows the stable `E4` path; the secondary virtual node is gated by graph size and learned pooled features so it can help large structures without dominating small structures.

**Tech Stack:** Python, PyTorch, DGL, ALIGNN faithful runner, YAML experiment configs.

---

### Task 1: Add E9 Model Variant

**Files:**

- Modify: `faithful_gnn/alignn_variants.py`

- [ ] **Step 1: Add helper logic inside a new `ALIGNNSizeAwareVirtualNodeRegressor` class**

Implement a class that reuses `ALIGNNEncoder`, keeps an E4-style primary virtual node, and adds a gated secondary virtual node.

Required public behavior:

```python
outputs = model([graph, line_graph, lattice])
assert "band_gap" in outputs
assert "slot2_enable" in outputs
assert "global_fusion_strength" in outputs
```

- [ ] **Step 2: Register the new variant**

Add this branch to `build_alignn_model`:

```python
if variant == "sizeawarevirtualnode":
    return ALIGNNSizeAwareVirtualNodeRegressor(config, model_config)
```

- [ ] **Step 3: Keep existing variants untouched**

Do not change the behavior of `baseline`, `virtualnode`, `attentivevirtualnode`, `multivirtualnode`, `transformer`, or `physconstraint`.

### Task 2: Add Gate Diagnostics to Training Loop

**Files:**

- Modify: `faithful_gnn/alignn_runner.py`

- [ ] **Step 1: Collect diagnostic tensors from model outputs**

During `run_epoch`, if model outputs contain `slot2_enable` or `global_fusion_strength`, collect their detached means.

- [ ] **Step 2: Add diagnostics to returned metrics**

Add these optional keys to the epoch metrics:

```python
{
    "slot2_enable_mean": 0.0,
    "global_fusion_strength_mean": 0.0,
}
```

Only include keys when the model actually returns them.

- [ ] **Step 3: Persist diagnostics in history and final metrics**

Add train / val gate means into `history.json`, and add test gate means into `metrics.json` when present.

### Task 3: Add E9 Experiment Config

**Files:**

- Create: `configs/E9_alignn_sizeaware_virtualnode.yaml`

- [ ] **Step 1: Start from E4-like training defaults**

Use:

```yaml
epochs: 50
optimizer: adamw
lr: 0.001
weight_decay: 0.00001
scheduler: onecycle
onecycle_max_lr: 0.001
```

- [ ] **Step 2: Add E9 model parameters**

Use:

```yaml
variant: sizeawarevirtualnode
num_virtual_nodes: 2
attention_heads: 4
dropout: 0.1
slot2_residual_init: 0.15
fusion_residual_init: 0.15
gate_bias_init: -2.0
```

### Task 4: Verify Locally

**Files:**

- Read: `faithful_gnn/alignn_variants.py`
- Read: `faithful_gnn/alignn_runner.py`
- Read: `configs/E9_alignn_sizeaware_virtualnode.yaml`

- [ ] **Step 1: Run Python compile checks**

Run:

```powershell
python -m py_compile faithful_gnn/alignn_variants.py faithful_gnn/alignn_runner.py run_faithful_experiment.py
```

Expected:

```text
no output and exit code 0
```

- [ ] **Step 2: Run config parse check**

Run:

```powershell
python - <<'PY'
from crystal_gnn.utils import load_yaml
cfg = load_yaml("configs/E9_alignn_sizeaware_virtualnode.yaml")
assert cfg["model"]["variant"] == "sizeawarevirtualnode"
print(cfg["experiment_name"])
PY
```

Expected:

```text
E9_ALIGNN_sizeaware_virtualnode
```

### Task 5: Remote Training Handoff

**Files:**

- Use: `configs/E9_alignn_sizeaware_virtualnode.yaml`
- Use: `run_faithful_experiment.py`
- Use: `faithful_gnn/alignn_variants.py`
- Use: `faithful_gnn/alignn_runner.py`

- [ ] **Step 1: Sync changed files to the remote server**

Copy the changed files to `/root/lanyun-fs/gnn`.

- [ ] **Step 2: Launch E9 training**

Run remotely:

```bash
cd /root/lanyun-fs/gnn
nohup /root/miniconda/envs/gnn-faithful/bin/python3.11 run_faithful_experiment.py \
  --config configs/E9_alignn_sizeaware_virtualnode.yaml \
  --output_dir results/E9_ALIGNN_sizeaware_virtualnode \
  > logs/E9_ALIGNN_sizeaware_virtualnode.nohup.log 2>&1 &
```

- [ ] **Step 3: Verify it started**

Check:

```bash
ps -ef | grep E9_ALIGNN_sizeaware_virtualnode | grep -v grep
tail -n 40 logs/E9_ALIGNN_sizeaware_virtualnode.nohup.log
```

Expected:

```text
process exists and logs show dataloader/model startup
```
