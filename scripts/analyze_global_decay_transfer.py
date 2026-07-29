from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_CSV = ROOT / "results" / "data_quality" / "mp_hzw_full_audit_table.remote.csv"

RUNS = {
    "MP_T7_Huber": ROOT
    / "results"
    / "mp_hzw_relaxed_clean_equal_atoms_mp_t"
    / "T7_ALIGNN_huber_virtualnode"
    / "predictions.csv",
    "MP_T10_DecayHuber": ROOT
    / "results"
    / "mp_hzw_relaxed_clean_equal_atoms_mp_t"
    / "T10_ALIGNN_decay_huber_virtualnode"
    / "predictions.csv",
    "HZW_T7_Huber": ROOT
    / "results"
    / "mp_hzw_relaxed_clean_equal_atoms_hzw_t"
    / "T7_ALIGNN_huber_virtualnode"
    / "predictions.csv",
    "HZW_T10_DecayHuber": ROOT
    / "results"
    / "mp_hzw_relaxed_clean_equal_atoms_hzw_t"
    / "T10_ALIGNN_decay_huber_virtualnode"
    / "predictions.csv",
}


def load_audit() -> dict[str, dict[str, str]]:
    with AUDIT_CSV.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return {row["sample_id"]: row for row in reader}


def load_predictions(path: Path) -> dict[str, dict[str, float | str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows: dict[str, dict[str, float | str]] = {}
        for row in reader:
            if row.get("split") != "test":
                continue
            target = float(row["target"])
            pred = float(row["prediction"])
            rows[row["sample_id"]] = {
                "sample_id": row["sample_id"],
                "target": target,
                "prediction": pred,
                "error": pred - target,
                "abs_error": abs(pred - target),
            }
        return rows


def atom_bucket(n: int) -> str:
    if n <= 10:
        return "<=10"
    if n <= 20:
        return "11-20"
    if n <= 40:
        return "21-40"
    if n <= 80:
        return "41-80"
    if n <= 160:
        return "81-160"
    return ">160"


def gap_bucket(gap: float) -> str:
    if abs(gap) <= 1e-6:
        return "0"
    if gap < 0.5:
        return "(0,0.5)"
    if gap < 1.0:
        return "[0.5,1)"
    if gap < 2.0:
        return "[1,2)"
    if gap < 4.0:
        return "[2,4)"
    return ">=4"


def diff_bucket(diff: float) -> str:
    if diff < 0.1:
        return "<0.1"
    if diff < 0.5:
        return "[0.1,0.5)"
    if diff < 1.0:
        return "[0.5,1)"
    return ">=1"


def summarize(values: list[dict[str, float | str]]) -> dict[str, float | int]:
    if not values:
        return {
            "n": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "median_abs_error": float("nan"),
            "p90_abs_error": float("nan"),
            "error_gt_1ev_ratio": float("nan"),
        }
    abs_errors = sorted(float(v["abs_error"]) for v in values)
    errors = [float(v["error"]) for v in values]
    n = len(values)
    mse = sum(e * e for e in errors) / n

    def percentile(p: float) -> float:
        idx = min(n - 1, max(0, int(round((n - 1) * p))))
        return abs_errors[idx]

    return {
        "n": n,
        "mae": sum(abs_errors) / n,
        "rmse": mse**0.5,
        "bias": sum(errors) / n,
        "median_abs_error": percentile(0.5),
        "p90_abs_error": percentile(0.9),
        "error_gt_1ev_ratio": sum(1 for e in abs_errors if e > 1.0) / n,
    }


def build_records() -> list[dict[str, object]]:
    audit = load_audit()
    preds = {name: load_predictions(path) for name, path in RUNS.items()}
    records: list[dict[str, object]] = []

    for dataset in ["MP", "HZW"]:
        t7 = preds[f"{dataset}_T7_Huber"]
        t10 = preds[f"{dataset}_T10_DecayHuber"]
        shared = sorted(set(t7) & set(t10))
        for sample_id in shared:
            audit_row = audit.get(sample_id, {})
            mp_gap = float(audit_row.get("mp_band_gap") or 0.0)
            hzw_gap = float(audit_row.get("hzw_band_gap") or 0.0)
            num_atoms = int(float(audit_row.get("mp_nsites") or audit_row.get("num_atoms") or 0))
            target_gap = mp_gap if dataset == "MP" else hzw_gap
            label_diff = abs(mp_gap - hzw_gap)
            for model_name, pred_row in [("T7_Huber", t7[sample_id]), ("T10_DecayHuber", t10[sample_id])]:
                records.append(
                    {
                        "dataset": dataset,
                        "model": model_name,
                        "sample_id": sample_id,
                        "target": float(pred_row["target"]),
                        "prediction": float(pred_row["prediction"]),
                        "error": float(pred_row["error"]),
                        "abs_error": float(pred_row["abs_error"]),
                        "num_atoms": num_atoms,
                        "atom_bucket": atom_bucket(num_atoms),
                        "target_gap_bucket": gap_bucket(target_gap),
                        "mp_hzw_gap_diff": label_diff,
                        "mp_hzw_gap_diff_bucket": diff_bucket(label_diff),
                        "mp_is_metal": abs(mp_gap) <= 1e-6,
                        "hzw_is_metal": abs(hzw_gap) <= 1e-6,
                    }
                )
    return records


def grouped_summary(records: list[dict[str, object]], key: str) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, float | str]]] = defaultdict(list)
    for row in records:
        groups[(str(row["dataset"]), str(row["model"]), str(row[key]))].append(row)  # type: ignore[arg-type]

    out: list[dict[str, object]] = []
    for (dataset, model, bucket), rows in sorted(groups.items()):
        stats = summarize(rows)
        out.append({"dataset": dataset, "model": model, "group": key, "bucket": bucket, **stats})
    return out


def t10_minus_t7_by_group(records: list[dict[str, object]], key: str) -> list[dict[str, object]]:
    by_dataset_bucket: dict[tuple[str, str], dict[str, list[dict[str, float | str]]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        by_dataset_bucket[(str(row["dataset"]), str(row[key]))][str(row["model"])].append(row)  # type: ignore[arg-type]

    out: list[dict[str, object]] = []
    for (dataset, bucket), model_rows in sorted(by_dataset_bucket.items()):
        if "T7_Huber" not in model_rows or "T10_DecayHuber" not in model_rows:
            continue
        s7 = summarize(model_rows["T7_Huber"])
        s10 = summarize(model_rows["T10_DecayHuber"])
        out.append(
            {
                "dataset": dataset,
                "group": key,
                "bucket": bucket,
                "n": s7["n"],
                "t7_mae": s7["mae"],
                "t10_mae": s10["mae"],
                "delta_t10_minus_t7": float(s10["mae"]) - float(s7["mae"]),
                "t7_bias": s7["bias"],
                "t10_bias": s10["bias"],
                "delta_bias": float(s10["bias"]) - float(s7["bias"]),
                "t7_p90_abs_error": s7["p90_abs_error"],
                "t10_p90_abs_error": s10["p90_abs_error"],
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(overall: list[dict[str, object]], deltas: list[dict[str, object]]) -> None:
    path = OUT_DIR / "global_decay_transfer_explanation.md"
    mp_overall = [r for r in overall if r["dataset"] == "MP"]
    hzw_overall = [r for r in overall if r["dataset"] == "HZW"]

    def fmt(x: object) -> str:
        if isinstance(x, float):
            return f"{x:.6f}"
        return str(x)

    lines = [
        "# Why global broadcast decay helps MP but not HZW",
        "",
        "## Short answer",
        "",
        "Global broadcast decay is not a universally stronger architecture. It is a training-time information-flow schedule: early epochs use strong global-to-node broadcasting, while late epochs reduce the global broadcast scale. On the MP label, this reduces global-token over-correction and improves bias/tail errors. On the HZW label, however, the same late-stage reduction removes useful global calibration, so the Huber-only model remains better.",
        "",
        "## Overall T7 vs T10 comparison",
        "",
        "| Dataset | Model | MAE | RMSE | Bias | Median AE | P90 AE | >1 eV ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in mp_overall + hzw_overall:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["model"]),
                    fmt(row["mae"]),
                    fmt(row["rmse"]),
                    fmt(row["bias"]),
                    fmt(row["median_abs_error"]),
                    fmt(row["p90_abs_error"]),
                    fmt(row["error_gt_1ev_ratio"]),
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## Evidence-based interpretation",
        "",
        "1. MP label: T10 improves MAE from T7's 0.224931 to 0.223686 and reduces RMSE from 0.503546 to 0.497333. Its mean residual also moves closer to zero, from 0.025437 to 0.013963. This means decay mainly helps MP by reducing systematic over-prediction and slightly compressing the long-error tail.",
        "2. HZW label: T10 worsens MAE from T7's 0.194002 to 0.197034 and RMSE from 0.445837 to 0.453592. Although its mean residual is closer to zero, the median and p90 absolute errors become worse. This indicates that decay reduces global bias but hurts ordinary-sample accuracy under the HZW labels.",
        "3. Mechanistically, the HZW target appears more compatible with persistent global calibration in the virtual node. MP labels benefit from attenuating global broadcast late in training, whereas HZW labels still need that global correction.",
        "4. Therefore, the robust cross-source claim should remain T7: ALIGNN + single plain virtual node + Huber loss. T10 should be described as an MP-label-specific improvement, not as the final universal model.",
        "",
        "## Bucket deltas",
        "",
        "Positive delta means T10 is worse than T7; negative delta means T10 is better.",
        "",
        "| Dataset | Group | Bucket | N | T7 MAE | T10 MAE | Delta | T7 bias | T10 bias |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in deltas:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["group"]),
                    str(row["bucket"]),
                    str(row["n"]),
                    fmt(row["t7_mae"]),
                    fmt(row["t10_mae"]),
                    fmt(row["delta_t10_minus_t7"]),
                    fmt(row["t7_bias"]),
                    fmt(row["t10_bias"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    records = build_records()
    overall = grouped_summary(records, "dataset")
    overall = [row for row in overall if row["bucket"] == row["dataset"]]
    all_grouped: list[dict[str, object]] = []
    all_deltas: list[dict[str, object]] = []
    for key in ["atom_bucket", "target_gap_bucket", "mp_hzw_gap_diff_bucket"]:
        all_grouped.extend(grouped_summary(records, key))
        all_deltas.extend(t10_minus_t7_by_group(records, key))

    write_csv(OUT_DIR / "global_decay_transfer_records.csv", records)
    write_csv(OUT_DIR / "global_decay_transfer_grouped_summary.csv", all_grouped)
    write_csv(OUT_DIR / "global_decay_transfer_t10_minus_t7.csv", all_deltas)
    write_csv(OUT_DIR / "global_decay_transfer_overall.csv", overall)
    write_markdown(overall, all_deltas)

    print(json.dumps({"records": len(records), "overall": overall}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
