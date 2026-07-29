from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Despite the ".gz.tmp" suffix, the raw HZW dump is plain (uncompressed) JSONL:
# its leading bytes are '{"', not the gzip magic 0x1f8b. We sniff the magic and
# pick the right opener so the script works regardless of how it was staged.
SRC = (
    ROOT
    / "api_downloads"
    / "api_downloads"
    / "crystal_band_100000.jsonl"
    / "crystal_band_100000.jsonl.gz.tmp"
)
OUT_CSV = OUT_DIR / "hzw_structmeta_lookup.csv"


def open_stream(path: Path):
    with path.open("rb") as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def main() -> None:
    seen: set[str] = set()
    n_lines = 0
    n_written = 0
    n_bad = 0

    with open_stream(SRC) as src, OUT_CSV.open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["hc_id", "space_group_number", "crystal_system"])
        for line in src:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            hc_id = rec.get("hc_id")
            if not hc_id or hc_id in seen:
                continue
            seen.add(hc_id)
            writer.writerow(
                [
                    hc_id,
                    rec.get("space_group_number", ""),
                    rec.get("crystal_system", ""),
                ]
            )
            n_written += 1

    print(
        json.dumps(
            {
                "source": str(SRC),
                "lines_read": n_lines,
                "unique_hc_ids_written": n_written,
                "json_decode_errors": n_bad,
                "output": str(OUT_CSV),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
