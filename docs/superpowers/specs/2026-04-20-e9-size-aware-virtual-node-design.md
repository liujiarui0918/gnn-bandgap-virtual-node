# E9 Size-Aware Dual-Slot Virtual Node Design

**Status:** Approved design draft pending final user review

**Goal**

Build a new `E9` ALIGNN-family model that beats `E4_ALIGNN_virtualnode` on overall test-set `MAE`, not just on large-structure buckets.

The design must preserve `E4`'s advantage on small and medium structures while selectively activating stronger global modeling on large and complex structures.

## 1. Problem Statement

Current experimental evidence shows a split regime:

- `E4` is strongest on small and medium structures, especially `num_atoms <= 80`
- `E8` starts to outperform `E4` on larger structures, especially `num_atoms > 80`
- `E5` and `E7` show that simply adding global attention or a single attentive virtual node is not enough

This means the current bottleneck is not "insufficient model size" in the general sense. The bottleneck is mismatch: one fixed global aggregation pattern is being forced onto samples with very different structural scales.

If the objective is to beat `E4` on overall `MAE`, then the model should:

- behave like `E4` on the majority regime where `E4` is already strong
- behave more like `E8` only when the sample appears to need richer global capacity

## 2. Approaches Considered

### Option A: Make `E8` larger and stronger

Increase slot count, attention depth, or hidden size in the existing dual-slot attentive design.

Pros:

- easy to implement
- likely improves very large structures

Cons:

- high risk of damaging the dominant small/medium regime
- repeats the same failure mode seen in `E5`, where more global machinery does not automatically improve overall `MAE`
- weak alignment with the observed split-regime statistics

### Option B: Add masked physics heads to `E4`

Attach `cbm`, `vbm`, and masked physics consistency losses to the `E4` backbone.

Pros:

- stable and incremental
- likely to improve tail behavior and calibration

Cons:

- may improve `RMSE` or robustness more than headline `MAE`
- does not directly address the observed advantage of multi-slot global modeling on large structures

### Option C: Size-aware hybrid between `E4` and `E8` (Recommended)

Keep the stable `E4` path as the default behavior. Add a second virtual slot and gated attentive fusion that turns on more strongly only when the graph appears large or globally complex.

Pros:

- directly matches the measured bucket-level behavior
- gives the model a path to recover `E4`-like behavior on the dominant regime
- gives the model access to `E8`-like capacity only when needed

Cons:

- more moving pieces than `E4`
- gate design must be conservative to avoid destabilizing training

## 3. Recommended Design

The recommended model is:

`E9 = ALIGNN backbone + always-on primary virtual slot + conditionally amplified secondary slot + size-aware gated fusion`

The core principle is asymmetric global capacity:

- slot 1 is the stable default global summary path
- slot 2 is a supplemental global memory path
- a learned gate controls how much the model uses slot 2 and how strongly global information is fused into the graph-level representation

This lets the model interpolate between:

- `E4`-like behavior on easier and smaller structures
- `E8`-like behavior on larger and harder structures

## 4. Architecture

### 4.1 Backbone

Reuse the existing ALIGNN encoder and keep the same fundamental depth used by the current faithful experiments:

- `alignn_layers = 4`
- `gcn_layers = 4`
- hidden size unchanged from the current ALIGNN-family variants

No changes should be made to the underlying ALIGNN message-passing pipeline in the first `E9` iteration. This keeps the experiment attributable to the new global modeling logic instead of a backbone rewrite.

### 4.2 Global Slots

Use two learnable virtual slots:

- `slot_1`: primary slot, always active
- `slot_2`: supplemental slot, modulated by the size-aware gate

Both slots are updated through the existing attentive virtual-node block pattern, but `slot_2` contribution is explicitly scaled before node injection and final fusion.

This differs from `E8`, where both slots participate symmetrically. In `E9`, the slots have different jobs:

- `slot_1` protects baseline stability
- `slot_2` provides extra capacity only when useful

### 4.3 Size-Aware Gate

Add a small gate network that predicts graph-level control values from lightweight structural cues and learned graph features.

Gate inputs for each graph:

- `num_atoms`
- `log1p(num_atoms)`
- graph pooled node feature
- mean node feature
- optional normalized average degree if easy to obtain from the batched graph

The gate produces at least two outputs:

- `slot2_enable` in `[0, 1]`
- `global_fusion_strength` in `[0, 1]`

Interpretation:

- `slot2_enable` determines how strongly the second slot participates in node updates and readout
- `global_fusion_strength` determines how much global slot context is allowed to modify the final graph representation

Initialization requirement:

- the gate must be biased toward conservative behavior at startup
- early training should therefore look much closer to `E4` than to `E8`

This is critical. The model should earn its way away from the `E4` regime instead of being born too far from it.

### 4.4 Node Injection

During message passing:

- `slot_1` is always used in the node update path
- `slot_2` is multiplied by `slot2_enable` before it contributes to node context

This keeps the early and default path stable while still allowing the second slot to amplify global capacity when the model decides the graph needs it.

### 4.5 Readout and Fusion

At graph readout:

- obtain the local pooled graph representation from node features
- derive slot-aware global context from the two slots
- use a fusion gate to combine local and global features

The final fused graph representation should look like:

- predominantly local plus primary-slot context by default
- progressively stronger slot-2 and attention-assisted context on harder graphs

Residual scaling should be preserved on new paths, with small initial values, to avoid the training instability seen in more aggressive attention-heavy variants.

## 5. Data Flow

Per graph, the forward path is:

1. Encode graph with the existing ALIGNN encoder
2. Initialize two virtual slots
3. For each ALIGNN / GCN stage:
   - inject primary slot into nodes
   - inject scaled secondary slot into nodes
   - update both slots from node features
4. Pool node features into a graph-level local representation
5. Build graph-level gate features from graph statistics and pooled embeddings
6. Predict `slot2_enable` and `global_fusion_strength`
7. Build final global context from slots, with slot 2 scaled by the predicted gate
8. Fuse local and global representations through conservative gated residual fusion
9. Predict `band_gap`

## 6. Training Strategy

### 6.1 Phase 1: Pure band-gap objective

The first `E9` experiment should remain single-task and optimize only `band_gap`.

Reason:

- the first question is whether the architecture alone can beat `E4`
- adding masked physics heads immediately would confound interpretation

Training defaults should stay close to `E4`:

- same optimizer family
- same base learning rate unless memory or stability forces change
- same epoch budget
- same batch size if memory allows

### 6.2 Phase 2: Optional physics-enhanced follow-up

If phase 1 gets close to `E4` but does not clearly exceed it, the next step is:

- `E9 + masked cbm/vbm auxiliary heads + masked physics consistency loss`

This follow-up should reuse the current valid-mask logic so that metallic samples with invalid `cbm/vbm` values do not pollute the auxiliary objective.

## 7. Failure Modes and Safeguards

### Failure Mode 1: Gate collapses to always-open

Risk:

- the model becomes effectively `E8` everywhere
- small and medium structures degrade

Safeguards:

- conservative gate bias initialization
- small residual scale on slot-2 and global fusion paths
- logging the mean and distribution of gate values per epoch

### Failure Mode 2: Gate collapses to always-closed

Risk:

- the model never uses the new capacity
- it degenerates to a slightly noisier `E4`

Safeguards:

- monitor gate activation by structure-size bucket
- inspect whether larger graphs produce consistently larger slot-2 activation

### Failure Mode 3: Global path destabilizes training

Risk:

- validation `MAE` oscillates or regresses early

Safeguards:

- keep added modules shallow
- retain residual scaling and modest dropout
- avoid stacking extra transformer blocks in the first `E9` iteration

## 8. Logging and Evaluation Requirements

Beyond the standard experiment outputs, `E9` should log:

- mean `slot2_enable` over train / val epochs
- mean `global_fusion_strength` over train / val epochs
- optional gate statistics bucketed by `num_atoms`

Evaluation should answer three questions:

1. Does `E9` beat `E4` on overall test-set `MAE`?
2. Does `E9` preserve `E4`'s strength on `num_atoms <= 80`?
3. Does `E9` retain or improve the large-structure gain observed in `E8`?

Success is not "beating `E4` in one bucket." Success is lowering overall `MAE` without a major collapse on the dominant sample regime.

## 9. Acceptance Criteria

The first `E9` implementation is considered successful if:

- it trains end-to-end in the faithful pipeline without breaking existing variants
- it produces standard outputs compatible with current result summarization
- it logs gate-related diagnostics for later analysis
- it beats `E4_ALIGNN_virtualnode` on overall test-set `MAE`

Secondary success criteria:

- it remains close to `E4` on small and medium structures
- it preserves or improves the large-structure advantage currently seen in `E8`

## 10. Implementation Scope

The first implementation should touch only the minimum surface required:

- add one new ALIGNN variant class for `E9`
- add one new experiment config
- add gate diagnostics to history / metrics if needed
- do not refactor unrelated training infrastructure in the same change

This keeps the experiment interpretable and reduces the risk of contaminating the result with unrelated engineering changes.

## 11. Summary

`E9` is not a bigger `E8`, and it is not `E4 + random attention`.

It is a deliberately asymmetric design:

- one stable primary global slot
- one selectively used supplemental global slot
- one conservative graph-level gate that decides how much extra global capacity the sample deserves

This is the most direct architectural response to the empirical result we already have:

- `E4` wins on the dominant regime
- `E8` wins on the large-structure regime

Therefore the most rational next step is to make one model that can behave like both, depending on the structure it sees.
