from __future__ import annotations

import argparse
from pathlib import Path

from crystal_gnn.data import preprocess_records, save_processed_splits
from crystal_gnn.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess crystal band gap data into graph shards.")
    parser.add_argument("--data_path", required=True, help="Path to the raw jsonl/jsonl.gz file.")
    parser.add_argument("--output_dir", required=True, help="Directory to write processed shards.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cutoff", type=float, default=8.0)
    parser.add_argument("--max_neighbors", type=int, default=12)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--shard_size", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits, metadata = preprocess_records(
        raw_path=args.data_path,
        cutoff=args.cutoff,
        max_neighbors=args.max_neighbors,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        limit=args.limit,
    )
    save_processed_splits(args.output_dir, splits, shard_size=args.shard_size)
    save_json(Path(args.output_dir) / "metadata.json", metadata)
    print(f"processed splits saved under {args.output_dir}")
    print(metadata)


if __name__ == "__main__":
    main()

