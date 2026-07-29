# Global Token and Over-Smoothing Diagnostic

## Status

This diagnostic has been run with hook-captured ALIGNN/GatedGCN atom embeddings from the saved T-series checkpoints.

Because the available local environment did not expose a GPU/Slurm runtime for a full rerun, the diagnostic used 240 stratified test samples per label source. The result is suitable for a cautious mechanism discussion, not for an over-claimed universal proof.

## Compared Models

- `T2`: ALIGNN baseline.
- `T4`: ALIGNN + one plain virtual node.
- `T7`: T4 + Huber loss.
- `T10`: T7 + global broadcast decay.

## Metrics

For each graph and captured layer, the script computed:

- Mean pairwise cosine similarity among atom embeddings. Lower values indicate less embedding collapse.
- Node-feature variance across atoms. Higher values indicate more node-state diversity.
- Graph Dirichlet energy over atom-graph edges. Higher values indicate less local embedding homogenization.

The expected over-smoothing relief signature is therefore:

```text
cosine decreases, variance increases, Dirichlet energy increases
```

## Main Result

The all-sample result is mixed. Global-token models increase node-feature variance and Dirichlet energy relative to T2, but they also increase mean pairwise cosine. Therefore, the current data do not support the strong statement that global tokens universally solve over-smoothing.

The large-structure subset is more coherent. For structures with more than 80 atoms, T4 and T7 show the expected relief signature on both MP and HZW label sources.

## Final-Layer Large-Structure Deltas vs T2

| Label source | Model | cosine delta | variance delta | Dirichlet delta | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| MP | T4 | -0.086 | +0.279 | +111.6 | Strongest clean evidence |
| MP | T7 | -0.084 | +0.251 | +91.6 | Supports same mechanism |
| MP | T10 | -0.081 | +0.178 | +57.4 | Weaker but still positive |
| HZW | T4 | -0.091 | +0.261 | +97.6 | Strongest clean evidence |
| HZW | T7 | -0.091 | +0.131 | +31.1 | Partial support |
| HZW | T10 | -0.102 | +0.047 | -13.3 | Not robust |

## Defensible Wording

Use:

> Hook-based diagnostics provide size-conditional evidence that a single global token can alleviate final-layer embedding homogenization for large crystal structures. The evidence is strongest for T4 and remains partially visible for T7.

Avoid:

> The global token solves over-smoothing.

The stronger claim is not supported because the all-sample cosine metric is not improved and T10 is not robust across label sources.

## Files

- MP diagnostic: `results/analysis/oversmoothing_hooks_mp/`.
- HZW diagnostic: `results/analysis/oversmoothing_hooks_hzw/`.
- Main figure: `figures/story_slides/oversmoothing_large_structure_evidence.png`.
- Script: `scripts/analyze_oversmoothing_hooks.py`.
