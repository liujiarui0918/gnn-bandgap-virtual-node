from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import requests
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Structure

from crystal_gnn.data import iter_json_records


DEFAULT_ENDPOINT = "https://api.materialsproject.org/materials/summary/"
DEFAULT_FIELDS = (
    "material_id,band_gap,cbm,vbm,is_metal,formula_pretty,nsites,density,structure"
)

AUDIT_COLUMNS = [
    "sample_id",
    "source_id",
    "hzw_hc_id",
    "mp_formula",
    "hzw_formula",
    "mp_reduced_formula",
    "hzw_reduced_formula",
    "mp_nsites",
    "hzw_nsites",
    "atom_count_relation",
    "mp_band_gap",
    "hzw_band_gap",
    "abs_gap_diff",
    "mp_is_metal",
    "hzw_is_metal",
    "mp_cbm",
    "mp_vbm",
    "mp_density",
    "hzw_density",
    "mp_volume_per_atom",
    "hzw_volume_per_atom",
    "volume_per_atom_rel_diff",
    "structure_match_level",
    "structure_matcher_fit",
    "is_garbage",
    "reason_code",
    "reason_text",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit MP-Hongzhiwei matched records and write full garbage table."
    )
    parser.add_argument("--api_key", default=os.environ.get("MP_API_KEY"), help="Materials Project API key.")
    parser.add_argument(
        "--hzw_raw_path",
        default="api_downloads/api_downloads/crystal_band_100000.jsonl/crystal_band_100000.jsonl.gz.tmp",
    )
    parser.add_argument("--output_dir", default="results/data_quality")
    parser.add_argument("--mp_cache_path", default=None)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--request_timeout", type=float, default=120.0)
    parser.add_argument("--request_pause", type=float, default=0.15)
    parser.add_argument("--max_retries", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--offline_cache",
        action="store_true",
        help="Do not call MP API for cache misses; mark missing ids as missing_mp_record.",
    )
    parser.add_argument("--metal_gap_threshold", type=float, default=0.05)
    parser.add_argument("--nonmetal_gap_threshold", type=float, default=0.10)
    parser.add_argument("--gap_diff_threshold", type=float, default=1.0)
    parser.add_argument("--volume_rel_tol", type=float, default=0.08)
    parser.add_argument(
        "--max_structure_matcher_atoms",
        type=int,
        default=80,
        help="Skip expensive StructureMatcher above this max atom count and use composition/volume heuristics.",
    )
    parser.add_argument("--force_fetch", action="store_true")
    return parser.parse_args()


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


def valid_gap(value: Any) -> float | None:
    gap = coerce_float(value)
    if gap is None or gap < 0.0 or gap > 15.0:
        return None
    return gap


def hzw_band_gap(record: dict[str, Any]) -> float | None:
    crystal_band = record.get("crystal_band") or []
    if not crystal_band:
        return None
    return valid_gap(crystal_band[0].get("band_gap"))


def composition_formula(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return Composition(text).reduced_formula
    except Exception:
        return text


def structure_from_payload(payload: Any) -> tuple[Structure | None, str | None]:
    if not payload:
        return None, "missing_structure"
    try:
        structure = Structure.from_dict(payload)
    except Exception as exc:
        return None, f"bad_structure:{type(exc).__name__}"
    if not structure.is_ordered:
        return None, "disordered_structure"
    if not all(bool(flag) for flag in structure.lattice.pbc):
        return None, "nonperiodic_structure"
    return structure, None


def collect_hzw_records(raw_path: Path, limit: int | None) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    stats = {
        "hzw_records_seen": 0,
        "hzw_valid_gap": 0,
        "hzw_missing_mp_source_id": 0,
        "hzw_invalid_gap": 0,
    }
    for record in iter_json_records(raw_path):
        stats["hzw_records_seen"] += 1
        gap = hzw_band_gap(record)
        if gap is None:
            stats["hzw_invalid_gap"] += 1
            continue
        stats["hzw_valid_gap"] += 1
        source_id = str(record.get("source_id") or "").strip()
        if not source_id.startswith("mp-"):
            stats["hzw_missing_mp_source_id"] += 1
            continue
        records.append(
            {
                "source_id": source_id,
                "hzw_hc_id": str(record.get("hc_id") or ""),
                "hzw_formula": str(record.get("formula") or ""),
                "hzw_band_gap": gap,
                "hzw_num_atoms": int(record.get("num_atoms") or 0),
                "hzw_structure": record.get("structure"),
            }
        )
        if limit is not None and len(records) >= limit:
            break
    stats["hzw_mp_source_records"] = len(records)
    stats["hzw_unique_mp_ids"] = len({record["source_id"] for record in records})
    return records, stats


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
    cache = {} if args.force_fetch else load_fetch_cache(cache_path)
    missing = [material_id for material_id in material_ids if material_id not in cache]
    stats = {
        "mp_fetch_requested_unique": len(material_ids),
        "mp_fetch_cache_hits": len(material_ids) - len(missing),
        "mp_fetch_missing_initial": len(missing),
        "mp_fetch_returned": 0,
        "mp_fetch_offline_misses": 0,
    }
    if not missing:
        return cache, stats
    if args.offline_cache:
        stats["mp_fetch_offline_misses"] = len(missing)
        return cache, stats
    if not args.api_key:
        raise ValueError("MP API key required when MP cache is incomplete.")

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


def atom_count_relation(mp_nsites: int, hzw_nsites: int) -> str:
    if mp_nsites <= 0 or hzw_nsites <= 0:
        return "unknown"
    if mp_nsites == hzw_nsites:
        return "equal"
    if mp_nsites > hzw_nsites and mp_nsites % hzw_nsites == 0:
        return "mp_multiple"
    if hzw_nsites > mp_nsites and hzw_nsites % mp_nsites == 0:
        return "hzw_multiple"
    return "mismatch"


def rel_diff(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    denom = max(abs(left), abs(right), 1e-12)
    return abs(left - right) / denom


def compare_structures(
    mp_structure: Structure | None,
    hzw_structure: Structure | None,
    args: argparse.Namespace,
) -> tuple[str, str, float | None, bool | None, list[str]]:
    reasons: list[str] = []
    if mp_structure is None or hzw_structure is None:
        return "mismatch", "unknown", None, None, ["missing_structure_for_compare"]

    mp_nsites = len(mp_structure)
    hzw_nsites = len(hzw_structure)
    relation = atom_count_relation(mp_nsites, hzw_nsites)
    mp_reduced = mp_structure.composition.reduced_formula
    hzw_reduced = hzw_structure.composition.reduced_formula
    volume_per_atom_diff = rel_diff(
        float(mp_structure.volume) / max(mp_nsites, 1),
        float(hzw_structure.volume) / max(hzw_nsites, 1),
    )

    if mp_reduced != hzw_reduced:
        return "mismatch", relation, volume_per_atom_diff, False, ["composition_mismatch"]
    if relation == "mismatch":
        return "mismatch", relation, volume_per_atom_diff, False, ["atom_count_mismatch"]
    if volume_per_atom_diff is not None and volume_per_atom_diff > args.volume_rel_tol:
        reasons.append("volume_per_atom_mismatch")

    max_atoms = max(mp_nsites, hzw_nsites)
    if max_atoms > args.max_structure_matcher_atoms:
        if reasons:
            return "mismatch", relation, volume_per_atom_diff, False, reasons
        if relation == "equal":
            return "compatible", relation, volume_per_atom_diff, None, []
        return "supercell_like", relation, volume_per_atom_diff, None, []

    try:
        matcher = StructureMatcher(
            ltol=0.2,
            stol=0.3,
            angle_tol=5,
            primitive_cell=True,
            scale=True,
            attempt_supercell=relation != "equal",
        )
        matched = bool(matcher.fit(mp_structure, hzw_structure))
    except Exception:
        matched = False
        reasons.append("structure_matcher_error")

    if not matched:
        reasons.append("structure_match_failed")
        return "mismatch", relation, volume_per_atom_diff, False, reasons
    if relation == "equal":
        return "exact_like", relation, volume_per_atom_diff, True, []
    return "supercell_like", relation, volume_per_atom_diff, True, []


def classify_record(
    index: int,
    hzw_record: dict[str, Any],
    mp_record: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    sample_id = f"{hzw_record['source_id'].replace('-', '_')}__{index:06d}"
    reasons: list[str] = []
    reason_text: list[str] = []

    hzw_gap = valid_gap(hzw_record.get("hzw_band_gap"))
    if hzw_gap is None:
        reasons.append("invalid_hzw_gap")
        reason_text.append("HZW band gap is missing or outside valid range.")

    if mp_record is None:
        row = {column: "" for column in AUDIT_COLUMNS}
        row.update(
            {
                "sample_id": sample_id,
                "source_id": hzw_record["source_id"],
                "hzw_hc_id": hzw_record.get("hzw_hc_id", ""),
                "hzw_formula": hzw_record.get("hzw_formula", ""),
                "hzw_band_gap": hzw_gap if hzw_gap is not None else "",
                "is_garbage": True,
                "reason_code": "missing_mp_record",
                "reason_text": "MP summary endpoint did not return this material_id.",
            }
        )
        return row

    mp_gap = valid_gap(mp_record.get("band_gap"))
    if mp_gap is None:
        reasons.append("invalid_mp_gap")
        reason_text.append("MP band gap is missing or outside valid range.")

    mp_structure, mp_structure_error = structure_from_payload(mp_record.get("structure"))
    hzw_structure, hzw_structure_error = structure_from_payload(hzw_record.get("hzw_structure"))
    if mp_structure_error is not None:
        reasons.append("missing_mp_structure" if mp_structure_error == "missing_structure" else "bad_mp_structure")
        reason_text.append(f"MP structure invalid: {mp_structure_error}.")
    if hzw_structure_error is not None:
        reasons.append("missing_hzw_structure" if hzw_structure_error == "missing_structure" else "bad_hzw_structure")
        reason_text.append(f"HZW structure invalid: {hzw_structure_error}.")

    structure_match_level = "mismatch"
    structure_matcher_fit: bool | None = None
    relation = "unknown"
    volume_per_atom_diff: float | None = None
    structure_reasons: list[str] = []
    if mp_structure is not None and hzw_structure is not None:
        structure_match_level, relation, volume_per_atom_diff, structure_matcher_fit, structure_reasons = compare_structures(
            mp_structure,
            hzw_structure,
            args,
        )
        reasons.extend(structure_reasons)

    mp_nsites = int(len(mp_structure)) if mp_structure is not None else int(mp_record.get("nsites") or 0)
    hzw_nsites = int(len(hzw_structure)) if hzw_structure is not None else int(hzw_record.get("hzw_num_atoms") or 0)
    if relation == "unknown":
        relation = atom_count_relation(mp_nsites, hzw_nsites)
    mp_formula = str(mp_record.get("formula_pretty") or "")
    hzw_formula = str(hzw_record.get("hzw_formula") or "")
    mp_reduced = mp_structure.composition.reduced_formula if mp_structure is not None else composition_formula(mp_formula)
    hzw_reduced = hzw_structure.composition.reduced_formula if hzw_structure is not None else composition_formula(hzw_formula)
    mp_volume_per_atom = float(mp_structure.volume) / max(mp_nsites, 1) if mp_structure is not None else None
    hzw_volume_per_atom = float(hzw_structure.volume) / max(hzw_nsites, 1) if hzw_structure is not None else None

    mp_is_metal_raw = bool(mp_record.get("is_metal")) if mp_record.get("is_metal") is not None else False
    mp_is_metal = mp_is_metal_raw or (mp_gap is not None and mp_gap <= args.metal_gap_threshold)
    hzw_is_metal = hzw_gap is not None and hzw_gap <= args.metal_gap_threshold

    if mp_gap is not None and hzw_gap is not None:
        if mp_is_metal and hzw_gap >= args.nonmetal_gap_threshold:
            reasons.append("mp_metal_hzw_nonmetal")
            reason_text.append("MP is metal-like but HZW band gap is nonmetal-like.")
        elif hzw_is_metal and mp_gap >= args.nonmetal_gap_threshold:
            reasons.append("mp_nonmetal_hzw_metal")
            reason_text.append("HZW is metal-like but MP band gap is nonmetal-like.")
        elif abs(mp_gap - hzw_gap) > args.gap_diff_threshold:
            reasons.append("same_class_gap_diff_gt_1ev")
            reason_text.append(f"MP and HZW are same class but band gap differs by more than {args.gap_diff_threshold:g} eV.")

    structure_reason_text = {
        "composition_mismatch": "MP and HZW reduced compositions are different.",
        "atom_count_mismatch": "MP and HZW atom counts are not equal and not an integer-multiple cell relation.",
        "volume_per_atom_mismatch": "MP and HZW volume per atom differs beyond tolerance.",
        "structure_match_failed": "pymatgen StructureMatcher cannot match the two structures.",
        "structure_matcher_error": "StructureMatcher raised an exception.",
    }
    for reason in structure_reasons:
        if reason in structure_reason_text:
            reason_text.append(structure_reason_text[reason])

    dedup_reasons = []
    for reason in reasons:
        if reason not in dedup_reasons:
            dedup_reasons.append(reason)
    is_garbage = bool(dedup_reasons)
    abs_gap_diff = abs(mp_gap - hzw_gap) if mp_gap is not None and hzw_gap is not None else None

    return {
        "sample_id": sample_id,
        "source_id": hzw_record["source_id"],
        "hzw_hc_id": hzw_record.get("hzw_hc_id", ""),
        "mp_formula": mp_formula,
        "hzw_formula": hzw_formula,
        "mp_reduced_formula": mp_reduced,
        "hzw_reduced_formula": hzw_reduced,
        "mp_nsites": mp_nsites,
        "hzw_nsites": hzw_nsites,
        "atom_count_relation": relation,
        "mp_band_gap": mp_gap if mp_gap is not None else "",
        "hzw_band_gap": hzw_gap if hzw_gap is not None else "",
        "abs_gap_diff": abs_gap_diff if abs_gap_diff is not None else "",
        "mp_is_metal": bool(mp_is_metal),
        "hzw_is_metal": bool(hzw_is_metal),
        "mp_cbm": coerce_float(mp_record.get("cbm"), default=""),
        "mp_vbm": coerce_float(mp_record.get("vbm"), default=""),
        "mp_density": coerce_float(mp_record.get("density"), default=""),
        "hzw_density": float(hzw_structure.density) if hzw_structure is not None else "",
        "mp_volume_per_atom": mp_volume_per_atom if mp_volume_per_atom is not None else "",
        "hzw_volume_per_atom": hzw_volume_per_atom if hzw_volume_per_atom is not None else "",
        "volume_per_atom_rel_diff": volume_per_atom_diff if volume_per_atom_diff is not None else "",
        "structure_match_level": structure_match_level,
        "structure_matcher_fit": "" if structure_matcher_fit is None else bool(structure_matcher_fit),
        "is_garbage": bool(is_garbage),
        "reason_code": ";".join(dedup_reasons),
        "reason_text": " ".join(reason_text),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, ""))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def count_reason_codes(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for reason in str(row.get("reason_code", "")).split(";"):
            if not reason:
                continue
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    with (output_dir / "mp_hzw_garbage_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    lines = [
        "# MP-HZW full garbage audit summary",
        "",
        f"- Total audited: {summary['total_audited']}",
        f"- Garbage rows: {summary['garbage_rows']}",
        f"- Clean rows: {summary['clean_rows']}",
        f"- Garbage ratio: {summary['garbage_ratio_percent']:.3f}%",
        "",
        "## Reason counts",
        "",
    ]
    for reason, count in summary["reason_counts"].items():
        lines.append(f"- `{reason}`: {count}")
    lines.extend(["", "## Structure match levels", ""])
    for level, count in summary["structure_match_level_counts"].items():
        lines.append(f"- `{level}`: {count}")
    lines.extend(["", "## Atom count relation", ""])
    for relation, count in summary["atom_count_relation_counts"].items():
        lines.append(f"- `{relation}`: {count}")
    (output_dir / "mp_hzw_garbage_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0 or args.chunk_size > 500:
        raise ValueError("chunk_size must be in [1, 500].")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mp_cache_path = Path(args.mp_cache_path) if args.mp_cache_path else output_dir / "mp_hzw_summary_fetch_cache.jsonl"

    hzw_records, hzw_stats = collect_hzw_records(Path(args.hzw_raw_path), args.limit)
    unique_mp_ids = sorted({record["source_id"] for record in hzw_records})
    mp_cache, fetch_stats = fetch_mp_records(unique_mp_ids, args, mp_cache_path)

    rng = random.Random(args.seed)
    indexed_records = list(enumerate(hzw_records))
    rng.shuffle(indexed_records)

    audit_rows: list[dict[str, Any]] = []
    started_at = time.time()
    for position, (original_index, hzw_record) in enumerate(indexed_records, start=1):
        mp_record = mp_cache.get(hzw_record["source_id"])
        audit_rows.append(classify_record(original_index, hzw_record, mp_record, args))
        if position % 1000 == 0:
            print(
                json.dumps(
                    {
                        "event": "audit_progress",
                        "done": position,
                        "total": len(indexed_records),
                        "elapsed_seconds": round(time.time() - started_at, 2),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    audit_rows.sort(key=lambda row: (str(row["source_id"]), str(row["sample_id"])))
    garbage_rows = [row for row in audit_rows if bool(row["is_garbage"])]
    clean_rows = [row for row in audit_rows if not bool(row["is_garbage"])]

    write_csv(output_dir / "mp_hzw_full_audit_table.csv", audit_rows, AUDIT_COLUMNS)
    write_csv(output_dir / "mp_hzw_full_garbage_table.csv", garbage_rows, AUDIT_COLUMNS)
    write_csv(output_dir / "mp_hzw_clean_source_ids.csv", [{"source_id": row["source_id"]} for row in clean_rows], ["source_id"])

    summary = {
        "hzw_stats": hzw_stats,
        "mp_fetch_stats": fetch_stats,
        "mp_cache_path": str(mp_cache_path.resolve()),
        "total_audited": len(audit_rows),
        "garbage_rows": len(garbage_rows),
        "clean_rows": len(clean_rows),
        "garbage_ratio_percent": 100.0 * len(garbage_rows) / max(len(audit_rows), 1),
        "reason_counts": count_reason_codes(garbage_rows),
        "structure_match_level_counts": count_by(audit_rows, "structure_match_level"),
        "atom_count_relation_counts": count_by(audit_rows, "atom_count_relation"),
        "thresholds": {
            "metal_gap_threshold": args.metal_gap_threshold,
            "nonmetal_gap_threshold": args.nonmetal_gap_threshold,
            "gap_diff_threshold": args.gap_diff_threshold,
            "volume_rel_tol": args.volume_rel_tol,
            "max_structure_matcher_atoms": args.max_structure_matcher_atoms,
        },
    }
    write_summary(output_dir, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
