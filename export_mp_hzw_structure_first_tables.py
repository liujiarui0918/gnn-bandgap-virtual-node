from __future__ import annotations

import argparse
import csv
import json
import signal
from collections import Counter
from pathlib import Path
from typing import Any


STRUCTURE_REASON_CODES = {
    "composition_mismatch",
    "atom_count_mismatch",
    "volume_per_atom_mismatch",
    "structure_match_failed",
    "structure_matcher_error",
    "missing_structure_for_compare",
    "missing_mp_structure",
    "bad_mp_structure",
    "missing_hzw_structure",
    "bad_hzw_structure",
}

LABEL_REASON_CODES = {
    "mp_metal_hzw_nonmetal",
    "mp_nonmetal_hzw_metal",
    "same_class_gap_diff_gt_1ev",
}

EXTRA_COLUMNS = [
    "structure_status",
    "structure_reason_code",
    "label_conflict_code",
    "label_conflict_text",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split MP-HZW audit results by structure correspondence first, then export "
            "label-conflict evidence tables inside structurally corresponding records."
        )
    )
    parser.add_argument("--audit_csv", default="results/data_quality/mp_hzw_full_audit_table.csv")
    parser.add_argument("--output_dir", default="results/data_quality/structure_first")
    parser.add_argument("--metal_gap_threshold", type=float, default=0.05)
    parser.add_argument("--nonmetal_gap_threshold", type=float, default=0.10)
    parser.add_argument("--gap_diff_threshold", type=float, default=1.0)
    parser.add_argument("--volume_rel_tol", type=float, default=0.08)
    parser.add_argument("--hzw_raw_path", default=None)
    parser.add_argument("--mp_cache_path", default=None)
    parser.add_argument("--rerun_unverified_matcher", action="store_true")
    parser.add_argument("--matcher_timeout_seconds", type=float, default=20.0)
    parser.add_argument(
        "--strict_matched_only",
        action="store_true",
        help="If set, exclude compatible/supercell-like rows without StructureMatcher proof from matched_all.",
    )
    return parser.parse_args()


def parse_bool(value: Any) -> bool | None:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def split_codes(value: Any) -> list[str]:
    return [part for part in str(value or "").split(";") if part]


def atom_relation(mp_nsites: int, hzw_nsites: int) -> str:
    if mp_nsites == hzw_nsites:
        return "equal"
    if hzw_nsites > 0 and mp_nsites % hzw_nsites == 0:
        return "mp_multiple"
    if mp_nsites > 0 and hzw_nsites % mp_nsites == 0:
        return "hzw_multiple"
    return "mismatch"


def relative_difference(left: float, right: float) -> float:
    denom = max(abs(left), abs(right), 1e-12)
    return abs(left - right) / denom


def infer_structure_status(row: dict[str, Any]) -> tuple[str, list[str]]:
    reason_codes = split_codes(row.get("reason_code", ""))
    if "missing_mp_record" in reason_codes:
        return "missing_mp_record", ["missing_mp_record"]

    structure_reasons = [code for code in reason_codes if code in STRUCTURE_REASON_CODES]
    level = str(row.get("structure_match_level", "")).strip()
    fit = parse_bool(row.get("structure_matcher_fit", ""))

    if structure_reasons or level == "mismatch":
        return "structure_unmatched", structure_reasons or ["structure_match_failed"]
    if level == "exact_like" and fit is True:
        return "structure_matched_exact", []
    if level == "supercell_like" and fit is True:
        return "structure_matched_supercell", []
    if level in {"compatible", "supercell_like"} and fit is None:
        return "structure_compatible_unverified", []
    if level in {"exact_like", "supercell_like", "compatible"}:
        return f"structure_{level}", []
    return "structure_unknown", structure_reasons or ["unknown_structure_status"]


def matcher_timeout_handler(signum: int, frame: object) -> None:
    raise TimeoutError("StructureMatcher timed out")


def strict_match_structures(
    mp_payload: Any,
    hzw_payload: Any,
    volume_rel_tol: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    from pymatgen.analysis.structure_matcher import StructureMatcher

    from audit_mp_hzw_matched_garbage import structure_from_payload

    mp_structure, mp_error = structure_from_payload(mp_payload)
    hzw_structure, hzw_error = structure_from_payload(hzw_payload)
    if mp_error is not None or hzw_error is not None:
        reasons = []
        if mp_error is not None:
            reasons.append("missing_mp_structure" if mp_error == "missing_structure" else "bad_mp_structure")
        if hzw_error is not None:
            reasons.append("missing_hzw_structure" if hzw_error == "missing_structure" else "bad_hzw_structure")
        return {
            "structure_status": "structure_unmatched",
            "structure_match_level": "mismatch",
            "structure_matcher_fit": False,
            "structure_reason_code": ";".join(reasons),
        }

    mp_nsites = len(mp_structure)
    hzw_nsites = len(hzw_structure)
    relation = atom_relation(mp_nsites, hzw_nsites)
    volume_diff = relative_difference(
        float(mp_structure.volume) / max(mp_nsites, 1),
        float(hzw_structure.volume) / max(hzw_nsites, 1),
    )
    reasons: list[str] = []
    if mp_structure.composition.reduced_formula != hzw_structure.composition.reduced_formula:
        reasons.append("composition_mismatch")
    if relation == "mismatch":
        reasons.append("atom_count_mismatch")
    if volume_diff is not None and volume_diff > volume_rel_tol:
        reasons.append("volume_per_atom_mismatch")
    if reasons:
        return {
            "structure_status": "structure_unmatched",
            "structure_match_level": "mismatch",
            "structure_matcher_fit": False,
            "structure_reason_code": ";".join(reasons),
        }

    old_handler = None
    timer_supported = hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")
    try:
        if timer_supported and timeout_seconds > 0:
            old_handler = signal.signal(signal.SIGALRM, matcher_timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        matcher = StructureMatcher(
            ltol=0.2,
            stol=0.3,
            angle_tol=5,
            primitive_cell=True,
            scale=True,
            attempt_supercell=relation != "equal",
        )
        matched = bool(matcher.fit(mp_structure, hzw_structure))
    except TimeoutError:
        return {
            "structure_status": "structure_match_timeout",
            "structure_match_level": "timeout",
            "structure_matcher_fit": "",
            "structure_reason_code": "structure_match_timeout",
        }
    except Exception:
        return {
            "structure_status": "structure_unmatched",
            "structure_match_level": "mismatch",
            "structure_matcher_fit": False,
            "structure_reason_code": "structure_matcher_error",
        }
    finally:
        if timer_supported and timeout_seconds > 0:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)

    if not matched:
        return {
            "structure_status": "structure_unmatched",
            "structure_match_level": "mismatch",
            "structure_matcher_fit": False,
            "structure_reason_code": "structure_match_failed",
        }
    if relation == "equal":
        return {
            "structure_status": "structure_matched_exact",
            "structure_match_level": "exact_like",
            "structure_matcher_fit": True,
            "structure_reason_code": "",
        }
    return {
        "structure_status": "structure_matched_supercell",
        "structure_match_level": "supercell_like",
        "structure_matcher_fit": True,
        "structure_reason_code": "",
    }


def build_hzw_by_sample_id(hzw_raw_path: Path) -> dict[str, dict[str, Any]]:
    from audit_mp_hzw_matched_garbage import collect_hzw_records

    hzw_records, _stats = collect_hzw_records(hzw_raw_path, limit=None)
    return {
        f"{record['source_id'].replace('-', '_')}__{index:06d}": record
        for index, record in enumerate(hzw_records)
    }


def rerun_unverified_matcher(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, int]:
    from audit_mp_hzw_matched_garbage import load_fetch_cache

    if not args.hzw_raw_path or not args.mp_cache_path:
        raise ValueError("--hzw_raw_path and --mp_cache_path are required with --rerun_unverified_matcher")
    hzw_by_sample_id = build_hzw_by_sample_id(Path(args.hzw_raw_path))
    mp_cache = load_fetch_cache(Path(args.mp_cache_path))
    stats = Counter()
    for row in rows:
        if row.get("structure_status") != "structure_compatible_unverified":
            continue
        stats["attempted"] += 1
        hzw_record = hzw_by_sample_id.get(str(row.get("sample_id", "")))
        mp_record = mp_cache.get(str(row.get("source_id", "")))
        if hzw_record is None or mp_record is None:
            row["structure_status"] = "structure_unmatched"
            row["structure_match_level"] = "mismatch"
            row["structure_matcher_fit"] = False
            row["structure_reason_code"] = "missing_structure_for_compare"
            stats["missing_payload"] += 1
            continue
        result = strict_match_structures(
            mp_record.get("structure"),
            hzw_record.get("hzw_structure"),
            volume_rel_tol=float(args.volume_rel_tol),
            timeout_seconds=float(args.matcher_timeout_seconds),
        )
        row.update(result)
        stats[str(row["structure_status"])] += 1
        if stats["attempted"] == 1 or stats["attempted"] % 100 == 0:
            print(json.dumps({"event": "strict_match_progress", **stats}, ensure_ascii=False), flush=True)
    return dict(stats)


def infer_label_conflict(row: dict[str, Any], args: argparse.Namespace) -> tuple[str, str]:
    mp_gap = parse_float(row.get("mp_band_gap"))
    hzw_gap = parse_float(row.get("hzw_band_gap"))
    if mp_gap is None or hzw_gap is None:
        return "", ""

    mp_is_metal_raw = parse_bool(row.get("mp_is_metal")) is True
    hzw_is_metal_raw = parse_bool(row.get("hzw_is_metal")) is True
    mp_is_metal = mp_is_metal_raw or mp_gap <= args.metal_gap_threshold
    hzw_is_metal = hzw_is_metal_raw or hzw_gap <= args.metal_gap_threshold

    if mp_is_metal and hzw_gap >= args.nonmetal_gap_threshold:
        return "mp_metal_hzw_nonmetal", "MP is metal-like but HZW band gap is nonmetal-like."
    if hzw_is_metal and mp_gap >= args.nonmetal_gap_threshold:
        return "mp_nonmetal_hzw_metal", "HZW is metal-like but MP band gap is nonmetal-like."
    if abs(mp_gap - hzw_gap) > args.gap_diff_threshold:
        return (
            "same_class_gap_diff_gt_1ev",
            f"MP and HZW are same class but band gap differs by more than {args.gap_diff_threshold:g} eV.",
        )
    return "", ""


def is_structure_corresponding(status: str, strict_only: bool) -> bool:
    if status in {"structure_matched_exact", "structure_matched_supercell"}:
        return True
    if strict_only:
        return False
    return status in {"structure_compatible_unverified", "structure_exact_like", "structure_supercell_like", "structure_compatible"}


def read_rows(path: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"empty audit csv: {path}")
        columns = list(reader.fieldnames)
        rows: list[dict[str, Any]] = []
        for row in reader:
            status, structure_reasons = infer_structure_status(row)
            label_code, label_text = infer_label_conflict(row, args)
            enriched = dict(row)
            enriched["structure_status"] = status
            enriched["structure_reason_code"] = ";".join(structure_reasons)
            enriched["label_conflict_code"] = label_code
            enriched["label_conflict_text"] = label_text
            rows.append(enriched)
    for column in EXTRA_COLUMNS:
        if column not in columns:
            columns.append(column)
    return rows, columns


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_counts_csv(path: Path, sections: dict[str, Counter[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "key", "count"])
        writer.writeheader()
        for section, counter in sections.items():
            for key, count in counter.most_common():
                writer.writerow({"section": section, "key": key, "count": count})


def filter_gap(rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    return [row for row in rows if (parse_float(row.get("abs_gap_diff")) or 0.0) > threshold]


def main() -> None:
    args = parse_args()
    audit_csv = Path(args.audit_csv)
    output_dir = Path(args.output_dir)
    rows, columns = read_rows(audit_csv, args)
    strict_match_stats: dict[str, int] = {}
    if args.rerun_unverified_matcher:
        strict_match_stats = rerun_unverified_matcher(rows, args)

    missing_mp = [row for row in rows if row["structure_status"] == "missing_mp_record"]
    structure_unmatched = [row for row in rows if row["structure_status"] == "structure_unmatched"]
    compatible_unverified = [row for row in rows if row["structure_status"] == "structure_compatible_unverified"]
    structure_match_timeout = [row for row in rows if row["structure_status"] == "structure_match_timeout"]
    structure_matched = [
        row for row in rows if is_structure_corresponding(row["structure_status"], args.strict_matched_only)
    ]

    matched_by_label = {
        "mp_metal_hzw_nonmetal": [
            row for row in structure_matched if row["label_conflict_code"] == "mp_metal_hzw_nonmetal"
        ],
        "mp_nonmetal_hzw_metal": [
            row for row in structure_matched if row["label_conflict_code"] == "mp_nonmetal_hzw_metal"
        ],
        "same_class_gap_diff_gt_1ev": [
            row for row in structure_matched if row["label_conflict_code"] == "same_class_gap_diff_gt_1ev"
        ],
    }
    bandgap_consistent = [
        row
        for row in structure_matched
        if not row["label_conflict_code"] and (parse_float(row.get("abs_gap_diff")) or 0.0) <= args.gap_diff_threshold
    ]
    mp_gap_zero_hzw_positive = [
        row
        for row in structure_matched
        if (parse_float(row.get("mp_band_gap")) or 0.0) <= args.metal_gap_threshold
        and (parse_float(row.get("hzw_band_gap")) or 0.0) >= args.nonmetal_gap_threshold
    ]
    hzw_gap_zero_mp_positive = [
        row
        for row in structure_matched
        if (parse_float(row.get("hzw_band_gap")) or 0.0) <= args.metal_gap_threshold
        and (parse_float(row.get("mp_band_gap")) or 0.0) >= args.nonmetal_gap_threshold
    ]
    volume_anomaly = [
        row for row in structure_matched if (parse_float(row.get("volume_per_atom_rel_diff")) or 0.0) > args.volume_rel_tol
    ]

    unmatched_by_reason = {
        "composition_mismatch": [
            row for row in structure_unmatched if "composition_mismatch" in split_codes(row["structure_reason_code"])
        ],
        "atom_count_mismatch": [
            row for row in structure_unmatched if "atom_count_mismatch" in split_codes(row["structure_reason_code"])
        ],
        "volume_per_atom_mismatch": [
            row for row in structure_unmatched if "volume_per_atom_mismatch" in split_codes(row["structure_reason_code"])
        ],
        "structure_match_failed": [
            row for row in structure_unmatched if "structure_match_failed" in split_codes(row["structure_reason_code"])
        ],
        "bad_or_missing_structure": [
            row
            for row in structure_unmatched
            if any(
                code in split_codes(row["structure_reason_code"])
                for code in {
                    "missing_structure_for_compare",
                    "missing_mp_structure",
                    "bad_mp_structure",
                    "missing_hzw_structure",
                    "bad_hzw_structure",
                }
            )
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "mp_hzw_structure_matched_all.csv", structure_matched, columns)
    write_csv(output_dir / "mp_hzw_structure_unmatched_all.csv", structure_unmatched, columns)
    write_csv(output_dir / "mp_hzw_missing_mp_record.csv", missing_mp, columns)
    write_csv(output_dir / "mp_hzw_structure_compatible_unverified.csv", compatible_unverified, columns)
    write_csv(output_dir / "mp_hzw_structure_match_timeout.csv", structure_match_timeout, columns)

    write_csv(output_dir / "mp_hzw_matched_mp_metal_hzw_nonmetal.csv", matched_by_label["mp_metal_hzw_nonmetal"], columns)
    write_csv(output_dir / "mp_hzw_matched_mp_nonmetal_hzw_metal.csv", matched_by_label["mp_nonmetal_hzw_metal"], columns)
    write_csv(
        output_dir / "mp_hzw_matched_same_class_gap_diff_gt_1ev.csv",
        matched_by_label["same_class_gap_diff_gt_1ev"],
        columns,
    )
    write_csv(output_dir / "mp_hzw_matched_gap_diff_gt_0p5ev.csv", filter_gap(structure_matched, 0.5), columns)
    write_csv(output_dir / "mp_hzw_matched_gap_diff_gt_0p1ev.csv", filter_gap(structure_matched, 0.1), columns)
    write_csv(output_dir / "mp_hzw_matched_mp_gap_zero_hzw_positive.csv", mp_gap_zero_hzw_positive, columns)
    write_csv(output_dir / "mp_hzw_matched_hzw_gap_zero_mp_positive.csv", hzw_gap_zero_mp_positive, columns)
    write_csv(output_dir / "mp_hzw_matched_bandgap_consistent.csv", bandgap_consistent, columns)
    write_csv(output_dir / "mp_hzw_matched_volume_per_atom_rel_diff_gt_0p08.csv", volume_anomaly, columns)

    write_csv(output_dir / "mp_hzw_unmatched_composition_mismatch.csv", unmatched_by_reason["composition_mismatch"], columns)
    write_csv(output_dir / "mp_hzw_unmatched_atom_count_mismatch.csv", unmatched_by_reason["atom_count_mismatch"], columns)
    write_csv(
        output_dir / "mp_hzw_unmatched_volume_per_atom_mismatch.csv",
        unmatched_by_reason["volume_per_atom_mismatch"],
        columns,
    )
    write_csv(output_dir / "mp_hzw_unmatched_structure_match_failed.csv", unmatched_by_reason["structure_match_failed"], columns)
    write_csv(output_dir / "mp_hzw_unmatched_bad_or_missing_structure.csv", unmatched_by_reason["bad_or_missing_structure"], columns)

    status_counts = Counter(row["structure_status"] for row in rows)
    matched_label_counts = Counter(row["label_conflict_code"] or "no_label_conflict" for row in structure_matched)
    unmatched_reason_counts: Counter[str] = Counter()
    for row in structure_unmatched:
        for code in split_codes(row["structure_reason_code"]):
            unmatched_reason_counts[code] += 1

    summary = {
        "audit_csv": str(audit_csv),
        "output_dir": str(output_dir),
        "strict_matched_only": bool(args.strict_matched_only),
        "total_rows": len(rows),
        "structure_matched_rows": len(structure_matched),
        "structure_unmatched_rows": len(structure_unmatched),
        "structure_compatible_unverified_rows": len(compatible_unverified),
        "missing_mp_record_rows": len(missing_mp),
        "strict_match_stats": strict_match_stats,
        "structure_status_counts": dict(status_counts),
        "matched_label_conflict_counts": dict(matched_label_counts),
        "unmatched_structure_reason_counts": dict(unmatched_reason_counts),
        "matched_gap_diff_gt_0p1ev": len(filter_gap(structure_matched, 0.1)),
        "matched_gap_diff_gt_0p5ev": len(filter_gap(structure_matched, 0.5)),
        "matched_same_class_gap_diff_gt_1ev": len(matched_by_label["same_class_gap_diff_gt_1ev"]),
        "matched_mp_metal_hzw_nonmetal": len(matched_by_label["mp_metal_hzw_nonmetal"]),
        "matched_mp_nonmetal_hzw_metal": len(matched_by_label["mp_nonmetal_hzw_metal"]),
        "matched_bandgap_consistent": len(bandgap_consistent),
        "matched_volume_per_atom_rel_diff_gt_0p08": len(volume_anomaly),
    }
    with (output_dir / "mp_hzw_structure_first_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    write_counts_csv(
        output_dir / "mp_hzw_structure_first_counts.csv",
        {
            "structure_status": status_counts,
            "matched_label_conflict": matched_label_counts,
            "unmatched_structure_reason": unmatched_reason_counts,
        },
    )

    md_lines = [
        "# MP-HZW structure-first audit summary",
        "",
        f"- Total rows: {len(rows)}",
        f"- Structure matched rows: {len(structure_matched)}",
        f"- Structure unmatched rows: {len(structure_unmatched)}",
        f"- Compatible but unverified rows: {len(compatible_unverified)}",
        f"- Missing MP records: {len(missing_mp)}",
        "",
        "## Structure Status",
        "",
    ]
    for key, count in status_counts.most_common():
        md_lines.append(f"- `{key}`: {count}")
    md_lines.extend(["", "## Label Conflicts Inside Structure-Matched Rows", ""])
    for key, count in matched_label_counts.most_common():
        md_lines.append(f"- `{key}`: {count}")
    md_lines.extend(["", "## Structure-Unmatched Reasons", ""])
    for key, count in unmatched_reason_counts.most_common():
        md_lines.append(f"- `{key}`: {count}")
    (output_dir / "mp_hzw_structure_first_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
