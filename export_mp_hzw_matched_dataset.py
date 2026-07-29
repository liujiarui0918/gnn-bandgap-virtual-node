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

from crystal_gnn.data import iter_json_records
from faithful_gnn.vendor import project_root


DEFAULT_ENDPOINT = "https://api.materialsproject.org/materials/summary/"
DEFAULT_FIELDS = (
    "material_id,band_gap,cbm,vbm,is_metal,formula_pretty,nsites,density,structure"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an MP dataset matched to Hongzhiwei crystal_band records by source_id=mp-*. "
            "Targets and structures are fetched from Materials Project."
        )
    )
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--hzw_raw_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit after valid HZW source-id filtering.")
    parser.add_argument("--chunk_size", type=int, default=100, help="MP material_ids per request.")
    parser.add_argument("--request_timeout", type=float, default=120.0)
    parser.add_argument("--request_pause", type=float, default=0.15)
    parser.add_argument("--max_retries", type=int, default=6)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument(
        "--cgcnn_root",
        default=None,
        help=(
            "Optional physical root for CGCNN CIF split files. If set, output_dir/cgcnn "
            "is created as a symlink to this path to avoid slow network small-file IO."
        ),
    )
    parser.add_argument(
        "--write_structures_dir",
        action="store_true",
        help="Also write output_dir/structures/*.cif. This is not needed by the training queue.",
    )
    parser.add_argument("--force", action="store_true", help="Delete and rebuild output_dir.")
    return parser.parse_args()


def safe_reset_dir(output_dir: Path, force: bool) -> None:
    workspace_root = project_root().resolve()
    resolved_output = output_dir.resolve()
    if resolved_output == workspace_root or workspace_root not in resolved_output.parents:
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
    allowed_roots = [output_dir]
    tmp_root = Path("/root/lanyun-tmp")
    if tmp_root.exists():
        allowed_roots.append(tmp_root)
    safe_remove_path(cgcnn_root, allowed_roots=allowed_roots)
    cgcnn_root.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        safe_remove_path(link_path, allowed_roots=[output_dir])
    os.symlink(str(cgcnn_root.resolve()), str(link_path), target_is_directory=True)
    return cgcnn_root


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


def hzw_band_gap(record: dict[str, Any]) -> float | None:
    crystal_band = record.get("crystal_band") or []
    if not crystal_band:
        return None
    band_gap = coerce_float(crystal_band[0].get("band_gap"))
    if band_gap is None or band_gap < 0.0 or band_gap > 15.0:
        return None
    return band_gap


def collect_hzw_records(raw_path: Path, limit: int | None) -> tuple[list[dict[str, Any]], dict[str, int]]:
    matched: list[dict[str, Any]] = []
    stats = {
        "hzw_records_seen": 0,
        "hzw_valid_gap": 0,
        "hzw_missing_mp_source_id": 0,
        "hzw_invalid_target": 0,
    }
    for record in iter_json_records(raw_path):
        stats["hzw_records_seen"] += 1
        gap = hzw_band_gap(record)
        if gap is None:
            stats["hzw_invalid_target"] += 1
            continue
        stats["hzw_valid_gap"] += 1
        source_id = str(record.get("source_id") or "").strip()
        if not source_id.startswith("mp-"):
            stats["hzw_missing_mp_source_id"] += 1
            continue
        matched.append(
            {
                "source_id": source_id,
                "hc_id": str(record.get("hc_id") or ""),
                "hzw_formula": str(record.get("formula") or ""),
                "hzw_band_gap": gap,
                "hzw_num_atoms": int(record.get("num_atoms") or 0),
            }
        )
        if limit is not None and len(matched) >= limit:
            break
    stats["hzw_mp_source_records"] = len(matched)
    stats["hzw_unique_mp_ids"] = len({item["source_id"] for item in matched})
    return matched, stats


def load_fetch_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    if not cache_path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            material_id = str(record.get("material_id") or "").strip()
            if material_id:
                cache[material_id] = record
    return cache


def append_cache(cache_path: Path, records: list[dict[str, Any]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def fetch_mp_chunk(
    session: requests.Session,
    endpoint: str,
    api_key: str,
    material_ids: list[str],
    timeout: float,
    max_retries: int,
) -> list[dict[str, Any]]:
    params = {
        "_fields": DEFAULT_FIELDS,
        "_limit": len(material_ids),
        # The MP summary endpoint treats repeated material_ids parameters as
        # "last value wins"; comma-separated IDs are required for batch fetch.
        "material_ids": ",".join(material_ids),
    }
    headers = {"X-API-KEY": api_key}
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = session.get(endpoint, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data")
            if not isinstance(data, list):
                raise ValueError(f"unexpected MP payload keys: {list(payload) if isinstance(payload, dict) else type(payload)}")
            return data
        except Exception as exc:  # pragma: no cover - network path
            last_error = exc
            sleep_seconds = min(30.0, 2**attempt)
            print(
                json.dumps(
                    {
                        "event": "mp_fetch_retry",
                        "attempt": attempt + 1,
                        "chunk_first": material_ids[0] if material_ids else "",
                        "chunk_size": len(material_ids),
                        "sleep_seconds": sleep_seconds,
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            time.sleep(sleep_seconds)
    assert last_error is not None
    raise RuntimeError(f"failed to fetch MP chunk after {max_retries} retries") from last_error


def fetch_mp_records(
    material_ids: list[str],
    args: argparse.Namespace,
    cache_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    cache = load_fetch_cache(cache_path)
    missing = [material_id for material_id in material_ids if material_id not in cache]
    stats = {
        "mp_fetch_requested_unique": len(material_ids),
        "mp_fetch_cache_hits": len(material_ids) - len(missing),
        "mp_fetch_missing_initial": len(missing),
        "mp_fetch_returned": 0,
    }
    if not missing:
        return cache, stats

    session = requests.Session()
    started_at = time.time()
    for start in range(0, len(missing), args.chunk_size):
        chunk = missing[start : start + args.chunk_size]
        records = fetch_mp_chunk(
            session=session,
            endpoint=args.endpoint,
            api_key=args.api_key,
            material_ids=chunk,
            timeout=args.request_timeout,
            max_retries=args.max_retries,
        )
        append_cache(cache_path, records)
        for record in records:
            material_id = str(record.get("material_id") or "").strip()
            if material_id:
                cache[material_id] = record
        stats["mp_fetch_returned"] += len(records)
        print(
            json.dumps(
                {
                    "event": "mp_fetch_progress",
                    "chunks_done": (start // args.chunk_size) + 1,
                    "chunks_total": math.ceil(len(missing) / args.chunk_size),
                    "cache_size": len(cache),
                    "returned_total": stats["mp_fetch_returned"],
                    "elapsed_seconds": round(time.time() - started_at, 2),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if args.request_pause > 0:
            time.sleep(args.request_pause)
    return cache, stats


def build_mp_manifest_record(
    hzw_record: dict[str, Any],
    mp_record: dict[str, Any],
    sample_index: int,
) -> tuple[dict[str, Any] | None, str | None]:
    band_gap = coerce_float(mp_record.get("band_gap"))
    if band_gap is None or band_gap < 0.0 or band_gap > 15.0:
        return None, "invalid_mp_target"

    structure_dict = mp_record.get("structure")
    if not structure_dict:
        return None, "missing_mp_structure"
    try:
        structure = Structure.from_dict(structure_dict)
    except Exception:
        return None, "bad_mp_structure"
    if not structure.is_ordered:
        return None, "disordered_mp_structure"
    if not all(bool(flag) for flag in structure.lattice.pbc):
        return None, "nonperiodic_mp_structure"

    source_id = str(hzw_record["source_id"])
    formula = str(mp_record.get("formula_pretty") or structure.composition.reduced_formula)
    density = coerce_float(mp_record.get("density"), default=float(structure.density)) or float(structure.density)
    cbm = coerce_float(mp_record.get("cbm"), default=0.0) or 0.0
    vbm = coerce_float(mp_record.get("vbm"), default=0.0) or 0.0
    sample_id = f"{source_id.replace('-', '_')}__{sample_index:06d}"
    return {
        "sample_id": sample_id,
        "source_id": source_id,
        "hzw_hc_id": hzw_record.get("hc_id", ""),
        "formula": formula,
        "hzw_formula": hzw_record.get("hzw_formula", ""),
        "density": float(density),
        "band_gap": float(band_gap),
        "hzw_band_gap": float(hzw_record["hzw_band_gap"]),
        "cbm": float(cbm),
        "vbm": float(vbm),
        "structure": structure.as_dict(),
    }, None


def write_manifest(records: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_cgcnn_split(
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
    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and leave room for test split.")
    if args.chunk_size <= 0 or args.chunk_size > 500:
        raise ValueError("chunk_size must be in [1, 500].")

    raw_path = Path(args.hzw_raw_path)
    output_dir = Path(args.output_dir)
    safe_reset_dir(output_dir, force=args.force)
    structures_dir = output_dir / "structures"
    cgcnn_root = prepare_cgcnn_root(output_dir, args.cgcnn_root)
    if args.write_structures_dir:
        structures_dir.mkdir(parents=True, exist_ok=True)

    hzw_records, hzw_stats = collect_hzw_records(raw_path, args.limit)
    unique_mp_ids = sorted({record["source_id"] for record in hzw_records})
    mp_cache, fetch_stats = fetch_mp_records(unique_mp_ids, args, output_dir / "mp_fetch_cache.jsonl")

    records: list[dict[str, Any]] = []
    skip_counts: dict[str, int] = {}
    for hzw_record in hzw_records:
        source_id = hzw_record["source_id"]
        mp_record = mp_cache.get(source_id)
        if mp_record is None:
            skip_counts["mp_not_found"] = skip_counts.get("mp_not_found", 0) + 1
            continue
        manifest_record, skip_reason = build_mp_manifest_record(hzw_record, mp_record, len(records))
        if manifest_record is None:
            assert skip_reason is not None
            skip_counts[skip_reason] = skip_counts.get(skip_reason, 0) + 1
            continue
        records.append(manifest_record)

    if not records:
        raise RuntimeError("no matched MP records were exported")

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

    abs_diffs = [abs(float(record["band_gap"]) - float(record["hzw_band_gap"])) for record in manifest_records]
    metadata = {
        "source": "materials_project_matched_to_hongzhiwei_source_id",
        "hzw_raw_path": str(raw_path.resolve()),
        "endpoint": args.endpoint,
        "output_dir": str(output_dir.resolve()),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": round(1.0 - args.train_ratio - args.val_ratio, 6),
        "accepted_records": total,
        "split_sizes": {name: len(items) for name, items in split_map.items()},
        "skip_counts": skip_counts,
        "hzw_stats": hzw_stats,
        "mp_fetch_stats": fetch_stats,
        "mp_vs_hzw_band_gap_abs_diff_mean": round(sum(abs_diffs) / len(abs_diffs), 6),
        "mp_vs_hzw_band_gap_abs_diff_max": round(max(abs_diffs), 6),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    with (output_dir / "splits.json").open("w", encoding="utf-8") as handle:
        json.dump({name: [record["sample_id"] for record in items] for name, items in split_map.items()}, handle, indent=2)

    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
