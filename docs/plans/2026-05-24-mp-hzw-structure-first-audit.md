# MP-HZW structure-first audit and evidence tables

Date: 2026-05-24

## Objective

Build a reproducible MP-HZW audit that answers a stricter question than the earlier
clean-dataset filter:

1. Start from the original HZW records that carry an MP `material_id` and a valid HZW band gap.
2. Fetch or reuse the MP summary cache for the same `material_id`.
3. First split records only by whether the MP and HZW crystal structures correspond.
4. Only inside the structurally matched table, diagnose label/value conflicts:
   MP metal vs HZW nonmetal, MP nonmetal vs HZW metal, same-class band gap difference greater than 1 eV, etc.

This keeps the core evidence clean: structure mismatch is a data-linking/structure problem,
while metal/nonmetal and band-gap disagreements are label or calculation-result problems.

## Inputs

- HZW raw data:
  `/data/home/tg100025/gnn_work/data/raw/hzw/crystal_band_100000.jsonl.gz.tmp`
- MP summary cache:
  `/data/home/tg100025/gnn_work/results/data_quality/mp_hzw_summary_fetch_cache.jsonl`
- Existing audit table, used as a fast bootstrap and consistency reference:
  `/data/home/tg100025/gnn_work/results/data_quality/mp_hzw_full_audit_table.csv`
- Local mirrored reference tables:
  `results/data_quality/mp_hzw_full_audit_table.remote.csv`
  `results/data_quality/mp_hzw_full_garbage_table.remote.csv`
  `results/data_quality/mp_hzw_garbage_summary.remote.json`

## Key Distinction

The earlier `clean_rows = 85478` table intentionally removed both structure problems and
band-gap/metallicity conflicts.

The new audit must not do that. It should produce:

- A structurally matched table, even if MP and HZW disagree on band gap.
- A structurally unmatched table, containing only records whose structures cannot be
  trusted as the same material representation.
- Derived conflict tables from the structurally matched table.

## Structure-Match Policy

For each MP-HZW pair:

1. Parse both structures with `pymatgen.Structure.from_dict`.
2. Reject as structurally unmatched if either structure is missing, malformed,
   disordered, or non-periodic.
3. Compare reduced composition.
4. Compare atom-count relation:
   `equal`, `mp_multiple`, `hzw_multiple`, or `mismatch`.
5. Compare volume per atom using `volume_rel_tol = 0.08`.
6. Run `pymatgen.analysis.structure_matcher.StructureMatcher`.

Recommended strict mode:

- Reuse the previous exact `StructureMatcher` result for records already marked
  `exact_like` or `structure_matcher_fit=True`.
- Re-run `StructureMatcher` on records previously marked `compatible` or
  `supercell_like` because the old audit skipped exact matching for structures
  larger than 80 atoms.
- Mark timeout/error rows separately instead of silently treating them as clean.

Structure table labels:

- `structure_matched_exact`: equal atom count and `StructureMatcher.fit=True`.
- `structure_matched_supercell`: integer-multiple cell relation and `StructureMatcher.fit=True`.
- `structure_compatible_unverified`: composition/atom/volume checks pass but full matcher
  timed out or was intentionally skipped. This should not be mixed with strictly proven rows
  in final claims.
- `structure_unmatched`: missing/bad structure, composition mismatch, atom-count mismatch,
  volume-per-atom mismatch, or StructureMatcher failure.
- `missing_mp_record`: HZW points to an MP id but the MP cache/API did not return the record.

## Required Output Tables

All outputs go under:

`results/data_quality/structure_first/`

Primary tables:

- `mp_hzw_structure_matched_all.csv`
  Structurally corresponding records. Includes label conflicts.
- `mp_hzw_structure_unmatched_all.csv`
  Structurally non-corresponding records and missing/bad structure rows.
- `mp_hzw_missing_mp_record.csv`
  HZW rows whose MP id is absent from the MP summary cache.
- `mp_hzw_structure_compatible_unverified.csv`
  Optional evidence table for rows that pass cheap structure checks but are not proven
  by StructureMatcher due to timeout or explicit skip.

Derived tables from `mp_hzw_structure_matched_all.csv`:

- `mp_hzw_matched_mp_metal_hzw_nonmetal.csv`
  MP is metal-like, HZW band gap is nonmetal-like.
- `mp_hzw_matched_mp_nonmetal_hzw_metal.csv`
  HZW is metal-like, MP band gap is nonmetal-like.
- `mp_hzw_matched_same_class_gap_diff_gt_1ev.csv`
  Both sides are same metal/nonmetal class, but absolute band-gap difference is greater than 1 eV.
- `mp_hzw_matched_gap_diff_gt_0p5ev.csv`
  Broader evidence table for nontrivial numerical disagreement.
- `mp_hzw_matched_gap_diff_gt_0p1ev.csv`
  Sensitive evidence table for small but visible numerical disagreement.
- `mp_hzw_matched_mp_gap_zero_hzw_positive.csv`
  MP gap is metal-like/zero but HZW gap is positive.
- `mp_hzw_matched_hzw_gap_zero_mp_positive.csv`
  HZW gap is metal-like/zero but MP gap is positive.
- `mp_hzw_matched_bandgap_consistent.csv`
  Matched structures with no metal/nonmetal conflict and `abs_gap_diff <= 1 eV`.

Derived tables from `mp_hzw_structure_unmatched_all.csv`:

- `mp_hzw_unmatched_composition_mismatch.csv`
- `mp_hzw_unmatched_atom_count_mismatch.csv`
- `mp_hzw_unmatched_volume_per_atom_mismatch.csv`
- `mp_hzw_unmatched_structure_match_failed.csv`
- `mp_hzw_unmatched_bad_or_missing_structure.csv`

Summary artifacts:

- `mp_hzw_structure_first_summary.json`
- `mp_hzw_structure_first_summary.md`
- `mp_hzw_structure_first_counts.csv`

## Core Columns

Every evidence table should include:

- `sample_id`
- `source_id`
- `hzw_hc_id`
- `mp_formula`
- `hzw_formula`
- `mp_reduced_formula`
- `hzw_reduced_formula`
- `mp_nsites`
- `hzw_nsites`
- `atom_count_relation`
- `structure_status`
- `structure_match_level`
- `structure_matcher_fit`
- `structure_reason_code`
- `mp_band_gap`
- `hzw_band_gap`
- `abs_gap_diff`
- `mp_is_metal`
- `hzw_is_metal`
- `label_conflict_code`
- `mp_cbm`
- `mp_vbm`
- `mp_density`
- `hzw_density`
- `mp_volume_per_atom`
- `hzw_volume_per_atom`
- `volume_per_atom_rel_diff`
- `reason_text`

## Label Conflict Policy

Definitions should be explicit and reproducible:

- `metal_gap_threshold = 0.05 eV`
- `nonmetal_gap_threshold = 0.10 eV`
- `gap_diff_threshold = 1.0 eV`

MP metal-like:

- MP `is_metal == True`, or
- MP `band_gap <= metal_gap_threshold`.

HZW metal-like:

- HZW `band_gap <= metal_gap_threshold`.

Conflict categories:

- `mp_metal_hzw_nonmetal`:
  MP metal-like and HZW band gap is at least `0.10 eV`.
- `mp_nonmetal_hzw_metal`:
  HZW metal-like and MP band gap is at least `0.10 eV`.
- `same_class_gap_diff_gt_1ev`:
  no metal/nonmetal conflict, but `abs(mp_gap - hzw_gap) > 1.0 eV`.

## Execution Plan

1. Add a new script `export_mp_hzw_structure_first_tables.py`.
2. The script should support two modes:
   - `--from_audit_csv`: fast mode using the existing audit CSV.
   - `--rerun_unverified_matcher`: strict mode that re-runs `StructureMatcher` for
     rows previously marked `compatible` or `supercell_like` without exact proof.
3. Run strict mode on the server using the existing Python environment.
4. Do not interrupt the current S-series training job unless the audit job causes resource
   contention. Prefer low worker count for structure matching if training is still in CPU-heavy
   preprocessing.
5. Copy the final `results/data_quality/structure_first/` directory back to local.
6. Report:
   - total records with MP id and valid HZW gap,
   - MP cache hits and misses,
   - strict structure-matched count,
   - structure-unmatched count by reason,
   - label conflicts inside structure-matched rows.

## Verification

Run locally:

```bash
python -m py_compile export_mp_hzw_structure_first_tables.py
```

Run on server:

```bash
cd /data/home/tg100025/gnn
/data/home/tg100025/gnn_work/envs/gnn-faithful/bin/python export_mp_hzw_structure_first_tables.py \
  --audit_csv /data/home/tg100025/gnn_work/results/data_quality/mp_hzw_full_audit_table.csv \
  --output_dir /data/home/tg100025/gnn_work/results/data_quality/structure_first \
  --from_audit_csv
```

Strict rerun command:

```bash
cd /data/home/tg100025/gnn
/data/home/tg100025/gnn_work/envs/gnn-faithful/bin/python export_mp_hzw_structure_first_tables.py \
  --audit_csv /data/home/tg100025/gnn_work/results/data_quality/mp_hzw_full_audit_table.csv \
  --hzw_raw_path /data/home/tg100025/gnn_work/data/raw/hzw/crystal_band_100000.jsonl.gz.tmp \
  --mp_cache_path /data/home/tg100025/gnn_work/results/data_quality/mp_hzw_summary_fetch_cache.jsonl \
  --output_dir /data/home/tg100025/gnn_work/results/data_quality/structure_first \
  --rerun_unverified_matcher \
  --matcher_timeout_seconds 20
```

## Risk Notes

- Full StructureMatcher on large structures can be slow. The old audit skipped exact matching
  above 80 atoms for this reason.
- Rows with MP missing from cache cannot be structure-compared and should stay in a separate
  missing-MP table, not in the structure-unmatched table.
- A structurally matched row can still have conflicting band gaps. That is not a structure
  failure; it is precisely the evidence needed to show data-label inconsistency between MP and HZW.
