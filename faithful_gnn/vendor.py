from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def configure_vendor_paths(root: str | Path | None = None) -> list[str]:
    base = Path(root) if root is not None else project_root()
    candidates = [
        base / ".vendor" / "cgcnn",
        base / ".vendor" / "alignn",
        base / ".vendor" / "matgl" / "src",
    ]
    inserted: list[str] = []
    for candidate in candidates:
        candidate_str = str(candidate.resolve())
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
            inserted.append(candidate_str)
    return inserted

