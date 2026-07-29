from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from pymatgen.core import Element, Structure

GRAPH_EXTRA_FEATURE_NAMES: tuple[str, ...] = (
    "density_norm",
    "log_volume",
    "log_volume_per_atom",
    "lattice_length_cv",
    "lattice_angle_std",
    "log_num_atoms",
    "element_count_norm",
    "composition_entropy",
    "mean_atomic_number_norm",
    "std_atomic_number_norm",
    "mean_electronegativity_norm",
    "std_electronegativity_norm",
    "mean_atomic_radius_norm",
    "std_atomic_radius_norm",
    "mean_atomic_mass_norm",
    "std_atomic_mass_norm",
    "metal_fraction",
    "mean_group_norm",
    "std_group_norm",
    "mean_row_norm",
    "std_row_norm",
)

GRAPH_EXTRA_FEATURE_INDEX: dict[str, int] = {
    name: index for index, name in enumerate(GRAPH_EXTRA_FEATURE_NAMES)
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _weighted_mean_std(values: list[float], weights: list[float]) -> tuple[float, float]:
    if not values or not weights:
        return 0.0, 0.0
    total = sum(weights)
    if total <= 0.0:
        return 0.0, 0.0
    normalized = [weight / total for weight in weights]
    mean = sum(value * weight for value, weight in zip(values, normalized))
    variance = sum(weight * (value - mean) ** 2 for value, weight in zip(values, normalized))
    return mean, math.sqrt(max(variance, 0.0))


def _element_radius(element: Element) -> float:
    return _safe_float(
        element.atomic_radius,
        default=_safe_float(getattr(element, "atomic_radius_calculated", None), default=1.0),
    )


def _element_group(element: Element) -> float:
    return _safe_float(getattr(element, "group", None), default=0.0)


def _element_row(element: Element) -> float:
    return _safe_float(getattr(element, "row", None), default=0.0)


def build_graph_extra_features(record: dict[str, Any]) -> list[float]:
    structure = Structure.from_dict(record["structure"])
    composition = structure.composition.fractional_composition
    fraction_map = composition.get_el_amt_dict()

    element_names = sorted(fraction_map)
    fractions = [float(fraction_map[name]) for name in element_names]
    elements = [Element(name) for name in element_names]

    atomic_numbers = [float(element.Z) for element in elements]
    electronegativities = [_safe_float(element.X, default=0.0) for element in elements]
    atomic_radii = [_element_radius(element) for element in elements]
    atomic_masses = [_safe_float(element.atomic_mass, default=0.0) for element in elements]
    groups = [_element_group(element) for element in elements]
    rows = [_element_row(element) for element in elements]
    metal_fraction = sum(
        fraction for fraction, element in zip(fractions, elements) if bool(getattr(element, "is_metal", False))
    )

    lengths = list(structure.lattice.abc)
    angles = list(structure.lattice.angles)
    mean_length = sum(lengths) / max(1, len(lengths))
    length_cv = math.sqrt(sum((length - mean_length) ** 2 for length in lengths) / max(1, len(lengths))) / max(
        mean_length,
        1e-6,
    )
    angle_mean = sum(angles) / max(1, len(angles))
    angle_std = math.sqrt(sum((angle - angle_mean) ** 2 for angle in angles) / max(1, len(angles))) / 180.0

    num_atoms = max(len(structure), 1)
    volume = _safe_float(structure.lattice.volume, default=0.0)
    density = _safe_float(record.get("density"), default=_safe_float(structure.density, default=0.0))
    volume_per_atom = volume / float(num_atoms)
    element_count = len(element_names)
    if element_count > 1:
        entropy_raw = -sum(fraction * math.log(max(fraction, 1e-12)) for fraction in fractions)
        composition_entropy = entropy_raw / math.log(float(element_count))
    else:
        composition_entropy = 0.0

    mean_atomic_number, std_atomic_number = _weighted_mean_std(atomic_numbers, fractions)
    mean_electronegativity, std_electronegativity = _weighted_mean_std(electronegativities, fractions)
    mean_atomic_radius, std_atomic_radius = _weighted_mean_std(atomic_radii, fractions)
    mean_atomic_mass, std_atomic_mass = _weighted_mean_std(atomic_masses, fractions)
    mean_group, std_group = _weighted_mean_std(groups, fractions)
    mean_row, std_row = _weighted_mean_std(rows, fractions)

    features = [
        density / 20.0,
        math.log1p(max(volume, 0.0)) / 8.0,
        math.log1p(max(volume_per_atom, 0.0)) / 5.0,
        length_cv,
        angle_std,
        math.log1p(float(num_atoms)) / 5.0,
        min(float(element_count) / 10.0, 1.0),
        composition_entropy,
        mean_atomic_number / 100.0,
        std_atomic_number / 50.0,
        mean_electronegativity / 4.0,
        std_electronegativity / 2.0,
        mean_atomic_radius / 3.0,
        std_atomic_radius / 1.5,
        mean_atomic_mass / 250.0,
        std_atomic_mass / 100.0,
        min(max(metal_fraction, 0.0), 1.0),
        mean_group / 18.0,
        std_group / 9.0,
        mean_row / 9.0,
        std_row / 4.5,
    ]
    return [float(feature) for feature in features]


def load_teacher_signal_map(path: str | Path | None) -> dict[str, float]:
    if not path:
        return {}
    signal_path = Path(path)
    if not signal_path.exists():
        raise FileNotFoundError(f"teacher signal file not found: {signal_path}")
    with signal_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        signal_map: dict[str, float] = {}
        for row in reader:
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                continue
            signal_map[sample_id] = _safe_float(row.get("gain_target"), default=0.5)
    return signal_map
