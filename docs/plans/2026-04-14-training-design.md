# 2026-04-14 Training Design

## Goal

Turn the PRD into a runnable training workspace fast enough to occupy the remote RTX 3090 tonight.

## Chosen Approach

- Use a single pure `PyTorch + pymatgen` code path to avoid `PyG/DGL` install friction on the rented server.
- Precompute crystal graphs once into shard files and reuse the cache across all six experiments.
- Implement four reusable model families:
  - `cgcnn`
  - `cgcnn_virtualnode`
  - `alignn` / `alignn_transformer`
  - `m3gnet` / `m3gnet_phys`
- Run experiments sequentially through `run_all_experiments.sh`.

## Tradeoffs

- This is a pragmatic ALIGNN-like / M3GNet-like implementation, not a paper-faithful reproduction of every internal block.
- Shared graph preprocessing uses one 8A neighbor cache for operational simplicity.
- The payoff is lower environment risk and a much faster path to first training run.
