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
        description="Export MP-HZW clean matched dataset from a full audit table and MP fetch cache."
    )
    parser.add_argument(
        "--hzw_raw_path",
        default="api_downloads/api_downloads/crystal_band_100000.jsonl/crystal_band_100000.jsonl.gz.tmp",
    )
    parser.add_argument("--audit_csv", default="results/data_quality/mp_hzw_full_audit_table.csv")
    parser.add_argument("--mp_cache_path", default="results/data_quality/mp_hzw_summary_fetch_cache.jsonl")
    parser.add_argument("--output_dir", default="data/mp_hzw_clean_matched_s")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--cgcnn_root",
        default=None,
        help="Optional physical root for CGCNN CIF files. output_dir/cgcnn will symlink to it.",
    )
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
    looks_like_clean_dataset = "clean" in resolved_output.name and "matched" in resolved_output.name
    if not is_allowed_root or not looks_like_clean_dataset:
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


def load_clean_audit_rows(audit_csv: Path) -> dict[str, dict[str, str]]:
    clean: dict[str, dict[str, str]] = {}
    with audit_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            is_garbage = str(row.get("is_garbage", "")).strip().lower() in {"true", "1", "yes"}
            if is_garbage:
                continue
            sample_id = str(row.get("sample_id") or "").strip()
            if sample_id:
                clean[sample_id] = row
    return clean


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
                "hzw_density": record.get("density"),
                "hzw_structure": record.get("structure"),
                "hzw_crystal_band": (record.get("crystal_band") or [{}])[0],
            }
        )
    return records


def build_manifest_record(
    hzw_record: dict[str, Any],
    audit_row: dict[str, str],
    mp_record: dict[str, Any],
) -> dict[str, Any] | None:
    structure_dict = mp_record.get("structure")
    if not structure_dict:
        return None
    structure = Structure.from_dict(structure_dict)
    band_gap = float(mp_record["band_gap"])
    cbm = float(mp_record.get("cbm") or 0.0)
    vbm = float(mp_record.get("vbm") or 0.0)
    formula = str(mp_record.get("formula_pretty") or structure.composition.reduced_formula)
    density = float(mp_record.get("density") or structure.density)
    hzw_band = hzw_record.get("hzw_crystal_band") or {}
    return {
        "sample_id": hzw_record["sample_id"],
        "source_id": hzw_record["source_id"],
        "hzw_hc_id": hzw_record.get("hzw_hc_id", ""),
        "formula": formula,
        "mp_formula": formula,
        "hzw_formula": hzw_record.get("hzw_formula", ""),
        "density": density,
        "band_gap": band_gap,
        "mp_band_gap": band_gap,
        "hzw_band_gap": float(hzw_record["hzw_band_gap"]),
        "cbm": cbm,
        "vbm": vbm,
        "mp_cbm": cbm,
        "mp_vbm": vbm,
        "hzw_cbm": hzw_band.get("cbm"),
        "hzw_vbm": hzw_band.get("vbm"),
        "hzw_gap_type": hzw_band.get("gap_type"),
        "hzw_num_atoms": int(hzw_record.get("hzw_num_atoms") or 0),
        "mp_nsites": int(len(structure)),
        "structure_match_level": audit_row.get("structure_match_level", ""),
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

    clean_audit = load_clean_audit_rows(Path(args.audit_csv))
    mp_cache = load_fetch_cache(Path(args.mp_cache_path))
    hzw_records = collect_hzw_records(Path(args.hzw_raw_path))

    records: list[dict[str, Any]] = []
    skip_counts: dict[str, int] = {}
    for hzw_record in hzw_records:
        audit_row = clean_audit.get(hzw_record["sample_id"])
        if audit_row is None:
            continue
        mp_record = mp_cache.get(hzw_record["source_id"])
        if mp_record is None:
            skip_counts["missing_mp_cache"] = skip_counts.get("missing_mp_cache", 0) + 1
            continue
        try:
            manifest_record = build_manifest_record(hzw_record, audit_row, mp_record)
        except Exception:
            skip_counts["build_record_error"] = skip_counts.get("build_record_error", 0) + 1
            continue
        if manifest_record is None:
            skip_counts["missing_mp_structure"] = skip_counts.get("missing_mp_structure", 0) + 1
            continue
        records.append(manifest_record)
        if args.limit is not None and len(records) >= args.limit:
            break

    if not records:
        raise RuntimeError("no clean matched records were exported")

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
    metadata = {
        "source": "materials_project_hongzhiwei_clean_matched",
        "hzw_raw_path": str(Path(args.hzw_raw_path).resolve()),
        "audit_csv": str(Path(args.audit_csv).resolve()),
        "mp_cache_path": str(Path(args.mp_cache_path).resolve()),
        "output_dir": str(output_dir.resolve()),
        "seed": args.seed,
        "accepted_records": total,
        "clean_audit_rows": len(clean_audit),
        "skip_counts": skip_counts,
        "split_sizes": {name: len(items) for name, items in split_map.items()},
        "mp_vs_hzw_band_gap_abs_diff_mean": round(sum(abs_diffs) / len(abs_diffs), 6),
        "mp_vs_hzw_band_gap_abs_diff_max": round(max(abs_diffs), 6),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "splits.json").write_text(
        json.dumps({name: [record["sample_id"] for record in items] for name, items in split_map.items()}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
