from __future__ import annotations

import argparse
import csv
import sys
import types
import uuid
from pathlib import Path


if "audit_mp_hzw_matched_garbage" not in sys.modules:
    audit_stub = types.ModuleType("audit_mp_hzw_matched_garbage")
    audit_stub.hzw_band_gap = lambda record: record.get("band_gap")
    audit_stub.load_fetch_cache = lambda _path: {}
    sys.modules["audit_mp_hzw_matched_garbage"] = audit_stub

import export_mp_hzw_final_clean_dataset as exporter
from export_mp_hzw_final_clean_dataset import build_manifest_record, get_filter_settings, load_final_audit_rows


def _make_csv_path():
    root = Path(__file__).resolve().parent / f"_tmp_export_clean_{uuid.uuid4().hex}"
    root.mkdir()
    return root / "audit.csv"


def _cleanup_csv_path(path):
    path.unlink(missing_ok=True)
    path.parent.rmdir()


def _write_audit_csv(path, rows):
    columns = [
        "sample_id",
        "mp_band_gap",
        "hzw_band_gap",
        "abs_gap_diff",
        "volume_per_atom_rel_diff",
        "atom_count_relation",
        "mp_is_metal",
        "hzw_is_metal",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _args(**overrides):
    values = {
        "metal_conflict_mode": "stored",
        "zero_gap_threshold": 1e-6,
        "max_abs_gap_diff": 0.5,
        "disable_gap_diff_filter": False,
        "max_volume_per_atom_rel_diff": 0.08,
        "disable_volume_filter": False,
        "require_equal_atom_count": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_load_final_audit_rows_keeps_legacy_strict_defaults():
    csv_path = _make_csv_path()
    _write_audit_csv(
        csv_path,
        [
            {
                "sample_id": "stored_conflict",
                "mp_band_gap": "1.0",
                "hzw_band_gap": "1.1",
                "abs_gap_diff": "0.1",
                "volume_per_atom_rel_diff": "0.01",
                "atom_count_relation": "equal",
                "mp_is_metal": "true",
                "hzw_is_metal": "false",
            },
            {
                "sample_id": "large_gap_diff",
                "mp_band_gap": "1.0",
                "hzw_band_gap": "2.0",
                "abs_gap_diff": "1.0",
                "volume_per_atom_rel_diff": "0.01",
                "atom_count_relation": "equal",
                "mp_is_metal": "false",
                "hzw_is_metal": "false",
            },
            {
                "sample_id": "large_volume_diff",
                "mp_band_gap": "1.0",
                "hzw_band_gap": "1.1",
                "abs_gap_diff": "0.1",
                "volume_per_atom_rel_diff": "0.10",
                "atom_count_relation": "equal",
                "mp_is_metal": "false",
                "hzw_is_metal": "false",
            },
            {
                "sample_id": "accepted",
                "mp_band_gap": "1.0",
                "hzw_band_gap": "1.1",
                "abs_gap_diff": "0.1",
                "volume_per_atom_rel_diff": "0.01",
                "atom_count_relation": "supercell",
                "mp_is_metal": "false",
                "hzw_is_metal": "false",
            },
        ],
    )

    try:
        rows, counts = load_final_audit_rows(csv_path, _args())
    finally:
        _cleanup_csv_path(csv_path)

    assert list(rows) == ["accepted"]
    assert counts["drop_metal_class_mismatch"] == 1
    assert counts["drop_zero_vs_nonzero_metal_conflict"] == 0
    assert counts["drop_abs_gap_diff_gt_threshold"] == 1
    assert counts["drop_volume_per_atom_rel_diff_gt_threshold"] == 1
    assert counts["drop_atom_count_relation_not_equal"] == 0
    assert counts["accepted_rows"] == 1


def test_load_final_audit_rows_supports_relaxed_t_series_filters():
    csv_path = _make_csv_path()
    _write_audit_csv(
        csv_path,
        [
            {
                "sample_id": "zero_vs_nonzero_conflict",
                "mp_band_gap": "0.0",
                "hzw_band_gap": "0.2",
                "abs_gap_diff": "0.2",
                "volume_per_atom_rel_diff": "0.01",
                "atom_count_relation": "equal",
                "mp_is_metal": "true",
                "hzw_is_metal": "false",
            },
            {
                "sample_id": "large_diffs_allowed",
                "mp_band_gap": "1.0",
                "hzw_band_gap": "2.2",
                "abs_gap_diff": "1.2",
                "volume_per_atom_rel_diff": "0.20",
                "atom_count_relation": "equal",
                "mp_is_metal": "true",
                "hzw_is_metal": "false",
            },
            {
                "sample_id": "atom_count_drop",
                "mp_band_gap": "1.0",
                "hzw_band_gap": "1.1",
                "abs_gap_diff": "0.1",
                "volume_per_atom_rel_diff": "0.01",
                "atom_count_relation": "supercell",
                "mp_is_metal": "false",
                "hzw_is_metal": "false",
            },
            {
                "sample_id": "threshold_zero_allowed",
                "mp_band_gap": "0.000001",
                "hzw_band_gap": "0.0",
                "abs_gap_diff": "0.000001",
                "volume_per_atom_rel_diff": "0.01",
                "atom_count_relation": "equal",
                "mp_is_metal": "false",
                "hzw_is_metal": "true",
            },
        ],
    )
    args = _args(
        metal_conflict_mode="zero_vs_nonzero",
        disable_gap_diff_filter=True,
        disable_volume_filter=True,
        require_equal_atom_count=True,
    )

    try:
        rows, counts = load_final_audit_rows(csv_path, args)
        filter_settings = get_filter_settings(args)
    finally:
        _cleanup_csv_path(csv_path)

    assert list(rows) == ["large_diffs_allowed", "threshold_zero_allowed"]
    assert counts["drop_metal_class_mismatch"] == 0
    assert counts["drop_zero_vs_nonzero_metal_conflict"] == 1
    assert counts["drop_abs_gap_diff_gt_threshold"] == 0
    assert counts["drop_volume_per_atom_rel_diff_gt_threshold"] == 0
    assert counts["drop_atom_count_relation_not_equal"] == 1
    assert counts["accepted_rows"] == 2
    assert filter_settings["metal_conflict"]["mode"] == "zero_vs_nonzero"
    assert filter_settings["gap_diff"]["enabled"] is False
    assert filter_settings["volume_per_atom_rel_diff"]["enabled"] is False
    assert filter_settings["atom_count_relation"]["enabled"] is True


def test_build_manifest_record_switches_supervised_label_source(monkeypatch):
    class FakeComposition:
        reduced_formula = "Si"

    class FakeStructure:
        composition = FakeComposition()
        density = 2.33

        def __len__(self):
            return 1

        def as_dict(self):
            return {"sites": [{"label": "Si"}]}

    class FakeStructureFactory:
        @staticmethod
        def from_dict(_payload):
            return FakeStructure()

    monkeypatch.setattr(exporter, "Structure", FakeStructureFactory)
    hzw_record = {
        "sample_id": "sample_1",
        "source_id": "mp-1",
        "hzw_hc_id": "hc-1",
        "hzw_formula": "Si",
        "hzw_band_gap": 1.25,
        "hzw_num_atoms": 1,
        "hzw_crystal_band": {"cbm": 2.0, "vbm": 0.75, "gap_type": "Direct"},
    }
    audit_row = {
        "abs_gap_diff": "0.75",
        "atom_count_relation": "equal",
        "volume_per_atom_rel_diff": "0.0",
        "mp_is_metal": "false",
        "hzw_is_metal": "false",
    }
    mp_record = {
        "structure": {"sites": [{"label": "Si"}]},
        "band_gap": 0.5,
        "cbm": 1.0,
        "vbm": 0.5,
        "formula_pretty": "Si",
        "density": 2.33,
    }

    mp_label_record = build_manifest_record(hzw_record, audit_row, mp_record, label_source="mp")
    hzw_label_record = build_manifest_record(hzw_record, audit_row, mp_record, label_source="hzw")

    assert mp_label_record["band_gap"] == 0.5
    assert mp_label_record["cbm"] == 1.0
    assert mp_label_record["vbm"] == 0.5
    assert hzw_label_record["band_gap"] == 1.25
    assert hzw_label_record["cbm"] == 2.0
    assert hzw_label_record["vbm"] == 0.75
    assert hzw_label_record["mp_band_gap"] == 0.5
    assert hzw_label_record["hzw_band_gap"] == 1.25
