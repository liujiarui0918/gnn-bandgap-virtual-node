from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import requests
from pymatgen.core import Structure

from faithful_gnn.vendor import project_root


DEFAULT_ENDPOINT = "https://api.materialsproject.org/materials/summary/"
DEFAULT_FIELDS = (
    "material_id,band_gap,cbm,vbm,is_metal,formula_pretty,nsites,density,structure"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a Materials Project crystal band-gap benchmark into the local faithful dataset layout."
    )
    parser.add_argument("--api_key", default=os.environ.get("MP_API_KEY"), help="Materials Project API key.")
    parser.add_argument("--output_dir", required=True, help="Directory where the faithful dataset export will be written.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None, help="Optional accepted-sample limit for smoke tests.")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Number of MP documents requested per page.")
    parser.add_argument("--request_timeout", type=float, default=120.0)
    parser.add_argument("--request_pause", type=float, default=0.05)
    parser.add_argument("--max_retries", type=int, default=6)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    return parser.parse_args()


def safe_reset_dir(output_dir: Path) -> None:
    workspace_root = project_root().resolve()
    resolved_output = output_dir.resolve()
    if resolved_output == workspace_root or workspace_root not in resolved_output.parents:
        raise ValueError(f"refusing to delete unsafe output_dir: {resolved_output}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def coerce_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def build_manifest_record(record: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    band_gap = coerce_float(record.get("band_gap"))
    if band_gap is None or band_gap < 0.0 or band_gap > 15.0:
        return None, "invalid_target"

    structure_dict = record.get("structure")
    if not structure_dict:
        return None, "bad_structure"

    try:
        structure = Structure.from_dict(structure_dict)
    except Exception:
        return None, "bad_structure"

    if not structure.is_ordered:
        return None, "disordered_structure"
    if not all(bool(flag) for flag in structure.lattice.pbc):
        return None, "nonperiodic_structure"

    material_id = str(record.get("material_id") or "").strip()
    if not material_id:
        return None, "missing_material_id"

    formula = str(record.get("formula_pretty") or structure.composition.reduced_formula)
    density = coerce_float(record.get("density"), default=float(structure.density))
    cbm = coerce_float(record.get("cbm"), default=0.0) or 0.0
    vbm = coerce_float(record.get("vbm"), default=0.0) or 0.0

    return {
        "sample_id": material_id.replace("/", "_"),
        "source_id": material_id,
        "formula": formula,
        "density": float(density),
        "band_gap": float(band_gap),
        "cbm": float(cbm),
        "vbm": float(vbm),
        "structure": structure.as_dict(),
    }, None


def fetch_page(
    session: requests.Session,
    endpoint: str,
    api_key: str,
    skip: int,
    limit: int,
    timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    params = {
        "_skip": skip,
        "_limit": limit,
        "_fields": DEFAULT_FIELDS,
    }
    headers = {"X-API-KEY": api_key}

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = session.get(endpoint, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or "data" not in payload:
                raise ValueError(f"unexpected MP payload: {type(payload)!r}")
            return payload
        except Exception as exc:  # pragma: no cover - network error path
            last_error = exc
            sleep_seconds = min(30.0, 2 ** attempt)
            print(
                json.dumps(
                    {
                        "event": "retry",
                        "skip": skip,
                        "attempt": attempt + 1,
                        "sleep_seconds": sleep_seconds,
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                )
            )
            time.sleep(sleep_seconds)
    assert last_error is not None
    raise RuntimeError(f"failed to fetch Materials Project page after {max_retries} retries") from last_error


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
            handle.write(f"{record['sample_id']},{record['band_gap']}\n")


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise ValueError("Materials Project API key is required. Pass --api_key or set MP_API_KEY.")
    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and leave room for a non-empty test split.")
    if args.chunk_size <= 0 or args.chunk_size > 1000:
        raise ValueError("chunk_size must be in the range [1, 1000].")

    output_dir = Path(args.output_dir)
    safe_reset_dir(output_dir)

    structures_dir = output_dir / "structures"
    cgcnn_root = output_dir / "cgcnn"
    structures_dir.mkdir(parents=True, exist_ok=True)
    cgcnn_root.mkdir(parents=True, exist_ok=True)

    accepted_records: list[dict[str, Any]] = []
    skipped_invalid_target = 0
    skipped_bad_structure = 0
    skipped_disordered_structure = 0
    skipped_nonperiodic_structure = 0
    skipped_missing_material_id = 0
    pages_fetched = 0
    total_docs_reported: int | None = None
    total_docs_scanned = 0

    session = requests.Session()
    skip = 0
    started_at = time.time()

    while True:
        payload = fetch_page(
            session=session,
            endpoint=args.endpoint,
            api_key=args.api_key,
            skip=skip,
            limit=args.chunk_size,
            timeout=args.request_timeout,
            max_retries=args.max_retries,
        )
        pages_fetched += 1
        total_docs_reported = int(payload.get("meta", {}).get("total_doc") or 0)
        batch = payload.get("data") or []
        if not batch:
            break

        total_docs_scanned += len(batch)
        for raw_record in batch:
            manifest_record, skip_reason = build_manifest_record(raw_record)
            if manifest_record is None:
                if skip_reason == "invalid_target":
                    skipped_invalid_target += 1
                elif skip_reason == "bad_structure":
                    skipped_bad_structure += 1
                elif skip_reason == "disordered_structure":
                    skipped_disordered_structure += 1
                elif skip_reason == "nonperiodic_structure":
                    skipped_nonperiodic_structure += 1
                elif skip_reason == "missing_material_id":
                    skipped_missing_material_id += 1
                continue

            accepted_records.append(manifest_record)
            if args.limit is not None and len(accepted_records) >= args.limit:
                break

        progress = {
            "event": "fetch_progress",
            "pages_fetched": pages_fetched,
            "docs_scanned": total_docs_scanned,
            "docs_reported": total_docs_reported,
            "accepted": len(accepted_records),
            "skipped_invalid_target": skipped_invalid_target,
            "skipped_bad_structure": skipped_bad_structure,
            "skipped_disordered_structure": skipped_disordered_structure,
            "skipped_nonperiodic_structure": skipped_nonperiodic_structure,
            "elapsed_seconds": round(time.time() - started_at, 2),
        }
        print(json.dumps(progress, ensure_ascii=False))

        if args.limit is not None and len(accepted_records) >= args.limit:
            break
        skip += len(batch)
        if total_docs_reported and skip >= total_docs_reported:
            break
        if args.request_pause > 0:
            time.sleep(args.request_pause)

    if not accepted_records:
        raise RuntimeError("no valid Materials Project records were exported")

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
        "source": "materials_project",
        "endpoint": args.endpoint,
        "output_dir": str(output_dir.resolve()),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": round(1.0 - args.train_ratio - args.val_ratio, 6),
        "accepted_records": total,
        "total_docs_reported": total_docs_reported,
        "total_docs_scanned": total_docs_scanned,
        "pages_fetched": pages_fetched,
        "chunk_size": args.chunk_size,
        "limit": args.limit,
        "skipped_invalid_target": skipped_invalid_target,
        "skipped_bad_structure": skipped_bad_structure,
        "skipped_disordered_structure": skipped_disordered_structure,
        "skipped_nonperiodic_structure": skipped_nonperiodic_structure,
        "skipped_missing_material_id": skipped_missing_material_id,
        "split_sizes": {name: len(records) for name, records in split_map.items()},
        "export_seconds": round(time.time() - started_at, 3),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    with (output_dir / "splits.json").open("w", encoding="utf-8") as handle:
        json.dump({name: [record["sample_id"] for record in records] for name, records in split_map.items()}, handle, indent=2)

    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
