from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

from pymatgen.core import Structure

from audit_mp_hzw_matched_garbage import hzw_band_gap, load_fetch_cache
from crystal_gnn.data import iter_json_records
from faithful_gnn.vendor import project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the final MP-label dataset from strict MP-HZW structure matches, "
            "dropping configurable label conflicts, band-gap differences, and volume anomalies."
        )
    )
    parser.add_argument("--hzw_raw_path", required=True)
    parser.add_argument("--structure_matched_csv", required=True)
    parser.add_argument("--mp_cache_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cgcnn_root", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument(
        "--label_source",
        choices=("mp", "hzw"),
        default="mp",
        help="Which matched bandgap label to use as the supervised target.",
    )
    parser.add_argument(
        "--metal_conflict_mode",
        choices=("stored", "zero_vs_nonzero"),
        default="stored",
        help=(
            "stored keeps the legacy mp_is_metal != hzw_is_metal filter; "
            "zero_vs_nonzero drops only rows where one band_gap is <= zero_gap_threshold "
            "and the other is > zero_gap_threshold."
        ),
    )
    parser.add_argument(
        "--zero_gap_threshold",
        type=float,
        default=1e-6,
        help="Band-gap threshold used by --metal_conflict_mode zero_vs_nonzero.",
    )
    parser.add_argument("--max_abs_gap_diff", type=float, default=0.5)
    parser.add_argument(
        "--disable_gap_diff_filter",
        action="store_true",
        help="Do not drop rows by abs(mp_band_gap - hzw_band_gap).",
    )
    parser.add_argument("--max_volume_per_atom_rel_diff", type=float, default=0.08)
    parser.add_argument(
        "--disable_volume_filter",
        action="store_true",
        help="Do not drop rows by volume_per_atom_rel_diff.",
    )
    parser.add_argument(
        "--require_equal_atom_count",
        action="store_true",
        help="Drop primitive/supercell-equivalent pairs whose MP and HZW atom counts differ.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--write_structures_dir", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def safe_reset_dir(output_dir: Path, force: bool) -> None:
    workspace_root = project_root().resolve()
    resolved_output = output_dir.resolve()
    work_root = Path(os.environ.get("WORK_ROOT", "")).expanduser()
    allowed_roots = [workspace_root / "data"]
    if str(work_root):
        allowed_roots.append(work_root / "data")
    resolved_allowed_roots = [root.resolve() for root in allowed_roots]
    is_allowed_root = any(
        resolved_output == root or root in resolved_output.parents for root in resolved_allowed_roots
    )
    safe_name = "clean" in resolved_output.name and (
        "matched" in resolved_output.name or "final" in resolved_output.name or "equal_atoms" in resolved_output.name
    )
    if not is_allowed_root or not safe_name:
        raise ValueError(f"refusing to reset unsafe output_dir: {resolved_output}")
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def safe_remove_path(path: Path, allowed_roots: list[Path]) -> None:
    resolved_path = path.resolve()
    resolved_roots = [root.resolve() for root in allowed_roots]
    if not any(resolved_path == root or root in resolved_path.parents for root in resolved_roots):
        raise ValueError(f"refusing to remove unsafe path: {resolved_path}")
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def prepare_cgcnn_root(output_dir: Path, requested_root: str | None) -> Path:
    link_path = output_dir / "cgcnn"
    if requested_root is None:
        safe_remove_path(link_path, allowed_roots=[output_dir])
        link_path.mkdir(parents=True, exist_ok=True)
        return link_path

    cgcnn_root = Path(requested_root)
    if not cgcnn_root.is_absolute():
        cgcnn_root = (output_dir / cgcnn_root).resolve()
    allowed_roots = [output_dir, cgcnn_root.parent]
    safe_remove_path(cgcnn_root, allowed_roots=allowed_roots)
    cgcnn_root.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        safe_remove_path(link_path, allowed_roots=[output_dir])
    os.symlink(str(cgcnn_root.resolve()), str(link_path), target_is_directory=True)
    return cgcnn_root


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def is_zero_vs_nonzero_conflict(mp_gap: float, hzw_gap: float, threshold: float) -> bool:
    return (mp_gap <= threshold) != (hzw_gap <= threshold)


def get_filter_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "metal_conflict": {
            "enabled": True,
            "mode": args.metal_conflict_mode,
            "zero_gap_threshold": args.zero_gap_threshold,
        },
        "gap_diff": {
            "enabled": not args.disable_gap_diff_filter,
            "max_abs_gap_diff": args.max_abs_gap_diff,
        },
        "volume_per_atom_rel_diff": {
            "enabled": not args.disable_volume_filter,
            "max_volume_per_atom_rel_diff": args.max_volume_per_atom_rel_diff,
        },
        "atom_count_relation": {
            "enabled": args.require_equal_atom_count,
            "required_relation": "equal" if args.require_equal_atom_count else None,
        },
        "hzw_num_atoms_vs_mp_nsites": {
            "enabled": args.require_equal_atom_count,
            "required_relation": "equal" if args.require_equal_atom_count else None,
        },
    }


def load_final_audit_rows(path: Path, args: argparse.Namespace) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    rows: dict[str, dict[str, str]] = {}
    counts = {
        "structure_matched_rows": 0,
        "drop_metal_class_mismatch": 0,
        "drop_zero_vs_nonzero_metal_conflict": 0,
        "drop_abs_gap_diff_gt_threshold": 0,
        "drop_volume_per_atom_rel_diff_gt_threshold": 0,
        "drop_atom_count_relation_not_equal": 0,
        "accepted_rows": 0,
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            counts["structure_matched_rows"] += 1
            mp_is_metal = parse_bool(row.get("mp_is_metal"))
            hzw_is_metal = parse_bool(row.get("hzw_is_metal"))
            mp_gap = parse_float(row.get("mp_band_gap"))
            hzw_gap = parse_float(row.get("hzw_band_gap"))
            abs_gap_diff = parse_float(row.get("abs_gap_diff"))
            volume_diff = parse_float(row.get("volume_per_atom_rel_diff"))
            atom_count_relation = str(row.get("atom_count_relation") or "").strip()
            if args.metal_conflict_mode == "stored" and mp_is_metal != hzw_is_metal:
                counts["drop_metal_class_mismatch"] += 1
                continue
            if args.metal_conflict_mode == "zero_vs_nonzero" and is_zero_vs_nonzero_conflict(
                mp_gap, hzw_gap, args.zero_gap_threshold
            ):
                counts["drop_zero_vs_nonzero_metal_conflict"] += 1
                continue
            if not args.disable_gap_diff_filter and abs_gap_diff > args.max_abs_gap_diff:
                counts["drop_abs_gap_diff_gt_threshold"] += 1
                continue
            if not args.disable_volume_filter and volume_diff > args.max_volume_per_atom_rel_diff:
                counts["drop_volume_per_atom_rel_diff_gt_threshold"] += 1
                continue
            if args.require_equal_atom_count and atom_count_relation != "equal":
                counts["drop_atom_count_relation_not_equal"] += 1
                continue
            sample_id = str(row.get("sample_id") or "").strip()
            if sample_id:
                rows[sample_id] = row
                counts["accepted_rows"] += 1
    return rows, counts


def collect_hzw_records(raw_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in iter_json_records(raw_path):
        gap = hzw_band_gap(record)
        if gap is None:
            continue
        source_id = str(record.get("source_id") or "").strip()
        if not source_id.startswith("mp-"):
            continue
        index = len(records)
        sample_id = f"{source_id.replace('-', '_')}__{index:06d}"
        records.append(
            {
                "sample_id": sample_id,
                "source_id": source_id,
                "hzw_hc_id": str(record.get("hc_id") or ""),
                "hzw_formula": str(record.get("formula") or ""),
                "hzw_band_gap": float(gap),
                "hzw_num_atoms": int(record.get("num_atoms") or 0),
                "hzw_crystal_band": (record.get("crystal_band") or [{}])[0],
            }
        )
    return records


def build_manifest_record(
    hzw_record: dict[str, Any],
    audit_row: dict[str, str],
    mp_record: dict[str, Any],
    label_source: str = "mp",
) -> dict[str, Any] | None:
    structure_dict = mp_record.get("structure")
    if not structure_dict:
        return None
    structure = Structure.from_dict(structure_dict)
    mp_band_gap = float(mp_record["band_gap"])
    mp_cbm = float(mp_record.get("cbm") or 0.0)
    mp_vbm = float(mp_record.get("vbm") or 0.0)
    formula = str(mp_record.get("formula_pretty") or structure.composition.reduced_formula)
    density = float(mp_record.get("density") or structure.density)
    hzw_band = hzw_record.get("hzw_crystal_band") or {}
    hzw_band_gap = float(hzw_record["hzw_band_gap"])
    hzw_cbm = parse_float(hzw_band.get("cbm"))
    hzw_vbm = parse_float(hzw_band.get("vbm"))
    if label_source == "mp":
        band_gap = mp_band_gap
        cbm = mp_cbm
        vbm = mp_vbm
    elif label_source == "hzw":
        band_gap = hzw_band_gap
        cbm = hzw_cbm
        vbm = hzw_vbm
    else:
        raise ValueError(f"unknown label_source: {label_source}")
    return {
        "sample_id": hzw_record["sample_id"],
        "source_id": hzw_record["source_id"],
        "hzw_hc_id": hzw_record.get("hzw_hc_id", ""),
        "formula": formula,
        "mp_formula": formula,
        "hzw_formula": hzw_record.get("hzw_formula", ""),
        "density": density,
        "band_gap": band_gap,
        "mp_band_gap": mp_band_gap,
        "hzw_band_gap": hzw_band_gap,
        "abs_gap_diff": parse_float(audit_row.get("abs_gap_diff")),
        "cbm": cbm,
        "vbm": vbm,
        "mp_cbm": mp_cbm,
        "mp_vbm": mp_vbm,
        "hzw_cbm": hzw_cbm,
        "hzw_vbm": hzw_vbm,
        "hzw_gap_type": hzw_band.get("gap_type"),
        "hzw_num_atoms": int(hzw_record.get("hzw_num_atoms") or 0),
        "mp_nsites": int(len(structure)),
        "atom_count_relation": audit_row.get("atom_count_relation", ""),
        "structure_status": audit_row.get("structure_status", ""),
        "structure_match_level": audit_row.get("structure_match_level", ""),
        "structure_matcher_fit": audit_row.get("structure_matcher_fit", ""),
        "volume_per_atom_rel_diff": parse_float(audit_row.get("volume_per_atom_rel_diff")),
        "mp_is_metal": parse_bool(audit_row.get("mp_is_metal")),
        "hzw_is_metal": parse_bool(audit_row.get("hzw_is_metal")),
        "structure": structure.as_dict(),
    }


def write_manifest(records: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_cgcnn_split(records: list[dict[str, Any]], split_root: Path, atom_init_path: Path) -> None:
    split_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(atom_init_path, split_root / "atom_init.json")
    with (split_root / "id_prop.csv").open("w", encoding="utf-8", newline="") as handle:
        for record in records:
            structure = Structure.from_dict(record["structure"])
            structure.to(filename=str(split_root / f"{record['sample_id']}.cif"))
            handle.write(f"{record['sample_id']},{record['band_gap']}\n")


def main() -> None:
    args = parse_args()
    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and leave room for test split.")

    output_dir = Path(args.output_dir)
    safe_reset_dir(output_dir, force=args.force)
    cgcnn_root = prepare_cgcnn_root(output_dir, args.cgcnn_root)
    structures_dir = output_dir / "structures"
    if args.write_structures_dir:
        structures_dir.mkdir(parents=True, exist_ok=True)

    final_audit, filter_counts = load_final_audit_rows(Path(args.structure_matched_csv), args)
    mp_cache = load_fetch_cache(Path(args.mp_cache_path))
    hzw_records = collect_hzw_records(Path(args.hzw_raw_path))

    records: list[dict[str, Any]] = []
    skip_counts: dict[str, int] = {}
    for hzw_record in hzw_records:
        audit_row = final_audit.get(hzw_record["sample_id"])
        if audit_row is None:
            continue
        mp_record = mp_cache.get(hzw_record["source_id"])
        if mp_record is None:
            skip_counts["missing_mp_cache"] = skip_counts.get("missing_mp_cache", 0) + 1
            continue
        try:
            manifest_record = build_manifest_record(hzw_record, audit_row, mp_record, args.label_source)
        except Exception:
            skip_counts["build_record_error"] = skip_counts.get("build_record_error", 0) + 1
            continue
        if manifest_record is None:
            skip_counts["missing_mp_structure"] = skip_counts.get("missing_mp_structure", 0) + 1
            continue
        if args.require_equal_atom_count and manifest_record["hzw_num_atoms"] != manifest_record["mp_nsites"]:
            skip_counts["drop_hzw_raw_num_atoms_mismatch"] = (
                skip_counts.get("drop_hzw_raw_num_atoms_mismatch", 0) + 1
            )
            continue
        records.append(manifest_record)
        if args.limit is not None and len(records) >= args.limit:
            break

    if not records:
        raise RuntimeError("no final clean records were exported")

    rng = random.Random(args.seed)
    rng.shuffle(records)
    total = len(records)
    train_end = int(total * args.train_ratio)
    val_end = train_end + int(total * args.val_ratio)
    split_map = {
        "train": records[:train_end],
        "val": records[train_end:val_end],
        "test": records[val_end:],
    }

    manifest_records: list[dict[str, Any]] = []
    for split_name, split_records in split_map.items():
        for record in split_records:
            enriched = dict(record)
            enriched["split"] = split_name
            manifest_records.append(enriched)
            if args.write_structures_dir:
                structure = Structure.from_dict(record["structure"])
                structure.to(filename=str(structures_dir / f"{record['sample_id']}.cif"))

    write_manifest(manifest_records, output_dir / "manifest.jsonl")
    atom_init_path = project_root() / ".vendor" / "cgcnn" / "data" / "sample-regression" / "atom_init.json"
    for split_name, split_records in split_map.items():
        write_cgcnn_split(split_records, cgcnn_root / split_name, atom_init_path)

    abs_diffs = [abs(float(record["mp_band_gap"]) - float(record["hzw_band_gap"])) for record in manifest_records]
    volume_diffs = [float(record["volume_per_atom_rel_diff"]) for record in manifest_records]
    metadata = {
        "source": f"materials_project_hongzhiwei_final_clean_{args.label_source}_label",
        "label_source": "materials_project.band_gap"
        if args.label_source == "mp"
        else "hongzhiwei.crystal_band.band_gap",
        "hzw_raw_path": str(Path(args.hzw_raw_path).resolve()),
        "structure_matched_csv": str(Path(args.structure_matched_csv).resolve()),
        "mp_cache_path": str(Path(args.mp_cache_path).resolve()),
        "output_dir": str(output_dir.resolve()),
        "seed": args.seed,
        "thresholds": {
            "metal_conflict_mode": args.metal_conflict_mode,
            "metal_class_match": "mp_is_metal == hzw_is_metal"
            if args.metal_conflict_mode == "stored"
            else None,
            "zero_gap_threshold": args.zero_gap_threshold,
            "zero_vs_nonzero_conflict": "one band_gap <= zero_gap_threshold and the other > zero_gap_threshold"
            if args.metal_conflict_mode == "zero_vs_nonzero"
            else None,
            "gap_diff_filter_enabled": not args.disable_gap_diff_filter,
            "max_abs_gap_diff": args.max_abs_gap_diff if not args.disable_gap_diff_filter else None,
            "volume_filter_enabled": not args.disable_volume_filter,
            "max_volume_per_atom_rel_diff": args.max_volume_per_atom_rel_diff
            if not args.disable_volume_filter
            else None,
            "atom_count_relation": "equal" if args.require_equal_atom_count else "any structure-compatible relation",
        },
        "filter_settings": get_filter_settings(args),
        "filter_counts": filter_counts,
        "accepted_records": total,
        "skip_counts": skip_counts,
        "split_sizes": {name: len(items) for name, items in split_map.items()},
        "mp_vs_hzw_band_gap_abs_diff_mean": round(sum(abs_diffs) / len(abs_diffs), 6),
        "mp_vs_hzw_band_gap_abs_diff_max": round(max(abs_diffs), 6),
        "volume_per_atom_rel_diff_mean": round(sum(volume_diffs) / len(volume_diffs), 6),
        "volume_per_atom_rel_diff_max": round(max(volume_diffs), 6),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "splits.json").write_text(
        json.dumps({name: [record["sample_id"] for record in items] for name, items in split_map.items()}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
