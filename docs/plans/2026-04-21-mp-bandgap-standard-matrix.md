# Materials Project Standard Bandgap Matrix

Date: 2026-04-21

Dataset scope:
- Materials Project crystal structures only
- Standard bandgap benchmark

Experiment matrix:
- E1: CGCNN baseline
- E2: ALIGNN baseline
- E3: M3GNet baseline
- E4: ALIGNN + virtual node
- E7: ALIGNN + 1 attentive virtual node
- E8: ALIGNN + 2 attentive virtual nodes
- E12: ALIGNN + size-supervised dual-slot virtual node

Excluded for the standard MP run:
- E6: physics-constrained loss
- E13: size-supervised + physics-constrained loss

Reason for excluding E6/E13 on MP:
- The standard MP-wide bandgap task is clean enough for `band_gap`.
- `cbm`/`vbm` are not consistently usable across the full crystal benchmark, especially for metallic samples.
- To keep the MP study standard and comparable, use the seven experiments above only.
