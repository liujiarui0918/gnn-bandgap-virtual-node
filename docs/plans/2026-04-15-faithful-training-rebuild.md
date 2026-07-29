# Faithful Training Rebuild Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the first-generation lightweight experiment stack with paper-faithful baselines and ALIGNN-based improvement variants that always train for 50 epochs on fixed splits.

**Architecture:** Export one canonical fixed-split dataset from the raw JSONL once, then train each experiment through dedicated official-model runners instead of the old shared homemade trainer. CGCNN uses the official CGCNN model, ALIGNN and its three variants use the official ALIGNN backbone/components, and M3GNet uses the official MatGL M3GNet implementation in its own compatible environment.

**Tech Stack:** PyTorch, DGL, jarvis-tools, MatGL, pymatgen, YAML, CSV/JSON manifests, existing supervisor/summary scripts.

---

### Task 1: Write The Rebuild Design Into The Repo

**Files:**
- Create: `docs/plans/2026-04-15-faithful-training-rebuild.md`

**Step 1:** Capture the rebuild scope, architecture boundary, and experiment matrix.

**Step 2:** Make the design explicit about the non-goal:

`Cannot promise metric ordering before experiments finish.`

### Task 2: Export A Canonical Faithful Dataset

**Files:**
- Create: `export_official_dataset.py`

**Step 1:** Read the raw crystal-band JSONL with malformed-line tolerance.

**Step 2:** Filter invalid targets and build deterministic `train/val/test = 80/10/10` splits with seed `42`.

**Step 3:** Write a canonical manifest with:
- sample id
- split
- `band_gap`
- `cbm`
- `vbm`
- formula / source id
- serialized structure

**Step 4:** Write split-local CGCNN folders with:
- `id_prop.csv`
- `atom_init.json`
- `*.cif`

**Step 5:** Write export metadata summarizing accepted / skipped counts.

### Task 3: Add Shared Faithful Training Utilities

**Files:**
- Create: `faithful_gnn/__init__.py`
- Create: `faithful_gnn/vendor.py`
- Create: `faithful_gnn/common.py`

**Step 1:** Centralize vendor path bootstrapping for `.vendor/cgcnn`, `.vendor/alignn`, and `.vendor/matgl/src`.

**Step 2:** Centralize manifest loading, split helpers, logging helpers, checkpoint helpers, and metric aggregation.

### Task 4: Add The Official CGCNN Runner

**Files:**
- Create: `faithful_gnn/cgcnn_runner.py`

**Step 1:** Use official `cgcnn.model.CrystalGraphConvNet` and official `cgcnn.data.CIFData/collate_pool`.

**Step 2:** Train on explicit split folders, not on the official random split helper.

**Step 3:** Match the baseline architecture and save:
- `history.json`
- `metrics.json`
- `predictions.csv`
- `best_model.pt`

**Step 4:** Remove early stopping from this path and always run `50` epochs.

### Task 5: Add The ALIGNN Backbone And Three ALIGNN-Based Variants

**Files:**
- Create: `faithful_gnn/alignn_variants.py`
- Create: `faithful_gnn/alignn_runner.py`

**Step 1:** Reuse official ALIGNN components for the faithful baseline.

**Step 2:** Implement the three user-requested variants on the ALIGNN backbone:
- virtual node
- transformer/global attention
- physics-constraint multi-head

**Step 3:** Keep the training loop fixed at `50` epochs and save the same artifacts as the CGCNN runner.

**Step 4:** Make the physics-constraint loss explicit in code and metrics output.

### Task 6: Add The Official M3GNet Runner

**Files:**
- Create: `faithful_gnn/m3gnet_runner.py`

**Step 1:** Use official MatGL `M3GNet` and MatGL DGL graph conversion.

**Step 2:** Build deterministic splits from the canonical manifest rather than MatGL’s internal random splitter.

**Step 3:** Train for `50` epochs and write the same artifact contract.

### Task 7: Add A Unified Experiment Entry Point And Config Matrix

**Files:**
- Create: `run_faithful_experiment.py`
- Create: `configs/E1_cgcnn_faithful.yaml`
- Create: `configs/E2_alignn_faithful.yaml`
- Create: `configs/E3_m3gnet_faithful.yaml`
- Create: `configs/E4_alignn_virtualnode.yaml`
- Create: `configs/E5_alignn_transformer_faithful.yaml`
- Create: `configs/E6_alignn_physconstraint.yaml`

**Step 1:** Dispatch each experiment to the correct runner.

**Step 2:** Encode the new experiment family:
- faithful CGCNN baseline
- faithful ALIGNN baseline
- faithful M3GNet baseline
- ALIGNN + virtual node
- ALIGNN + transformer
- ALIGNN + physics constraint

### Task 8: Rewire Orchestration

**Files:**
- Modify: `supervise_training.py`
- Modify: `run_all_experiments.sh`

**Step 1:** Replace the old preprocess/train commands with:
- canonical export
- faithful experiment runner
- summarize

**Step 2:** Allow per-phase Python executables so ALIGNN/CGCNN and M3GNet can use different compatible environments if required.

### Task 9: Verify And Prepare Remote Relaunch

**Files:**
- Modify as needed: `bootstrap_env.sh`

**Step 1:** Run syntax checks locally.

**Step 2:** Redeploy to remote.

**Step 3:** Create the compatible remote environment(s), launch the new run, and monitor logs for failures.
