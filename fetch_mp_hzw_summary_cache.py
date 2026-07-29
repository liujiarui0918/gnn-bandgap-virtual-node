from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterator

import requests


DEFAULT_ENDPOINT = "https://api.materialsproject.org/materials/summary/"
DEFAULT_FIELDS = (
    "material_id,band_gap,cbm,vbm,is_metal,formula_pretty,nsites,density,structure"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch MP summary cache for HZW mp-* source ids.")
    parser.add_argument("--api_key", default=os.environ.get("MP_API_KEY"), required=False)
    parser.add_argument(
        "--hzw_raw_path",
        default="api_downloads/api_downloads/crystal_band_100000.jsonl/crystal_band_100000.jsonl.gz.tmp",
    )
    parser.add_argument("--cache_path", default="results/data_quality/mp_hzw_summary_fetch_cache.jsonl")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--chunk_size", type=int, default=150)
    parser.add_argument("--request_timeout", type=float, default=120.0)
    parser.add_argument("--request_pause", type=float, default=0.35)
    parser.add_argument("--max_retries", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def is_gzip(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(2) == b"\x1f\x8b"


def iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if is_gzip(path) else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def valid_hzw_gap(record: dict[str, Any]) -> bool:
    crystal_band = record.get("crystal_band") or []
    if not crystal_band:
        return False
    try:
        gap = float(crystal_band[0].get("band_gap"))
    except (TypeError, ValueError):
        return False
    return math.isfinite(gap) and 0.0 <= gap <= 15.0


def collect_mp_ids(hzw_raw_path: Path, limit: int | None) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for record in iter_json_records(hzw_raw_path):
        if not valid_hzw_gap(record):
            continue
        source_id = str(record.get("source_id") or "").strip()
        if not source_id.startswith("mp-") or source_id in seen:
            continue
        seen.add(source_id)
        ids.append(source_id)
        if limit is not None and len(ids) >= limit:
            break
    return sorted(ids)


def load_cache_ids(cache_path: Path) -> set[str]:
    if not cache_path.exists():
        return set()
    ids: set[str] = set()
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            material_id = str(record.get("material_id") or "").strip()
            if material_id:
                ids.add(material_id)
    return ids


def append_cache(cache_path: Path, records: list[dict[str, Any]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def fetch_chunk(
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
                raise ValueError(f"unexpected MP payload: {str(payload)[:200]}")
            return data
        except Exception as exc:
            last_error = exc
            sleep_seconds = min(60.0, 2**attempt)
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
    raise RuntimeError(f"failed to fetch chunk after {max_retries} retries") from last_error


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise ValueError("MP API key required via --api_key or MP_API_KEY.")
    if args.chunk_size <= 0 or args.chunk_size > 300:
        raise ValueError("chunk_size must be in [1, 300].")

    cache_path = Path(args.cache_path)
    all_ids = collect_mp_ids(Path(args.hzw_raw_path), args.limit)
    cached_ids = load_cache_ids(cache_path)
    missing = [material_id for material_id in all_ids if material_id not in cached_ids]
    print(
        json.dumps(
            {
                "event": "mp_cache_start",
                "unique_hzw_mp_ids": len(all_ids),
                "cached_ids": len(cached_ids),
                "missing_ids": len(missing),
                "cache_path": str(cache_path.resolve()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if not missing:
        return

    session = requests.Session()
    started_at = time.time()
    for start in range(0, len(missing), args.chunk_size):
        chunk = missing[start : start + args.chunk_size]
        records = fetch_chunk(
            session=session,
            endpoint=args.endpoint,
            api_key=args.api_key,
            material_ids=chunk,
            timeout=args.request_timeout,
            max_retries=args.max_retries,
        )
        append_cache(cache_path, records)
        print(
            json.dumps(
                {
                    "event": "mp_cache_progress",
                    "chunks_done": (start // args.chunk_size) + 1,
                    "chunks_total": math.ceil(len(missing) / args.chunk_size),
                    "chunk_size": len(chunk),
                    "returned": len(records),
                    "elapsed_seconds": round(time.time() - started_at, 2),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        time.sleep(max(0.0, args.request_pause))


if __name__ == "__main__":
    main()
