from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

from pymatgen.core import Structure

from crystal_gnn.data import iter_json_records
from faithful_gnn.vendor import project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the Hongzhiwei crystal-band dataset into fixed official splits.")
    parser.add_argument("--data_path", required=True, help="Path to the raw JSONL or JSONL.GZ file.")
    parser.add_argument("--output_dir", required=True, help="Directory where the faithful dataset export will be written.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def safe_reset_dir(output_dir: Path) -> None:
    workspace_root = project_root().resolve()
    resolved_output = output_dir.resolve()
    if resolved_output == workspace_root or workspace_root not in resolved_output.parents:
        raise ValueError(f"refusing to delete unsafe output_dir: {resolved_output}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def extract_target(record: dict[str, Any]) -> dict[str, float] | None:
    crystal_band = record.get("crystal_band") or []
    if not crystal_band:
        return None
    band_info = crystal_band[0]
    band_gap = band_info.get("band_gap")
    if band_gap is None:
        return None
    band_gap = float(band_gap)
    if band_gap < 0 or band_gap > 15:
        return None
    return {
        "band_gap": band_gap,
        "cbm": float(band_info.get("cbm", 0.0) or 0.0),
        "vbm": float(band_info.get("vbm", 0.0) or 0.0),
    }


def build_manifest_record(record: dict[str, Any], accepted_index: int) -> dict[str, Any] | None:
    targets = extract_target(record)
    if targets is None:
        return None
    try:
        structure = Structure.from_dict(record["structure"])
    except Exception:
        return None
    formula = str(record.get("formula") or structure.composition.reduced_formula)
    return {
        "sample_id": f"cb_{accepted_index:06d}",
        "source_id": str(record.get("source_id") or record.get("hc_id") or ""),
        "formula": formula,
        "density": float(record.get("density", structure.density)),
        "band_gap": targets["band_gap"],
        "cbm": targets["cbm"],
        "vbm": targets["vbm"],
        "structure": structure.as_dict(),
    }


def write_manifest(records: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_cgcnn_split(
    split_name: str,
    records: list[dict[str, Any]],
    split_root: Path,
    atom_init_path: Path,
) -> None:
    split_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(atom_init_path, split_root / "atom_init.json")
    with (split_root / "id_prop.csv").open("w", encoding="utf-8", newline="") as handle:
        for record in records:
            structure = Structure.from_dict(record["structure"])
            structure.to(filename=str(split_root / f"{record['sample_id']}.cif"))
            # Official CGCNN appends ".cif" internally when loading the ID column.
            handle.write(f"{record['sample_id']},{record['band_gap']}\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    safe_reset_dir(output_dir)

    structures_dir = output_dir / "structures"
    cgcnn_root = output_dir / "cgcnn"
    structures_dir.mkdir(parents=True, exist_ok=True)
    cgcnn_root.mkdir(parents=True, exist_ok=True)

    accepted_records: list[dict[str, Any]] = []
    malformed_json = 0
    skipped_invalid_target = 0
    skipped_bad_structure = 0

    def on_json_error(_: int, __: Any, ___: str) -> None:
        nonlocal malformed_json
        malformed_json += 1

    iterator = iter_json_records(args.data_path, on_error=on_json_error)
    for raw_record in iterator:
        manifest_record = build_manifest_record(raw_record, len(accepted_records))
        if manifest_record is None:
            if extract_target(raw_record) is None:
                skipped_invalid_target += 1
            else:
                skipped_bad_structure += 1
            continue
        accepted_records.append(manifest_record)
        if args.limit is not None and len(accepted_records) >= args.limit:
            break

    rng = random.Random(args.seed)
    rng.shuffle(accepted_records)

    total = len(accepted_records)
    train_end = int(total * args.train_ratio)
    val_end = train_end + int(total * args.val_ratio)
    split_map = {
        "train": accepted_records[:train_end],
        "val": accepted_records[train_end:val_end],
        "test": accepted_records[val_end:],
    }

    manifest_records: list[dict[str, Any]] = []
    for split_name, records in split_map.items():
        for record in records:
            enriched = dict(record)
            enriched["split"] = split_name
            manifest_records.append(enriched)
            structure = Structure.from_dict(record["structure"])
            structure.to(filename=str(structures_dir / f"{record['sample_id']}.cif"))

    write_manifest(manifest_records, output_dir / "manifest.jsonl")
    atom_init_path = project_root() / ".vendor" / "cgcnn" / "data" / "sample-regression" / "atom_init.json"
    for split_name, records in split_map.items():
        write_cgcnn_split(split_name, records, cgcnn_root / split_name, atom_init_path)

    metadata = {
        "raw_path": str(Path(args.data_path).resolve()),
        "output_dir": str(output_dir.resolve()),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": round(1.0 - args.train_ratio - args.val_ratio, 6),
        "accepted_records": total,
        "malformed_json": malformed_json,
        "skipped_invalid_target": skipped_invalid_target,
        "skipped_bad_structure": skipped_bad_structure,
        "split_sizes": {name: len(records) for name, records in split_map.items()},
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    with (output_dir / "splits.json").open("w", encoding="utf-8") as handle:
        json.dump({name: [record["sample_id"] for record in records] for name, records in split_map.items()}, handle, indent=2)

    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
