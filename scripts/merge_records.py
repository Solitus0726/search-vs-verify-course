# merge_records.py - merge shard records
# Merge data/results/part-*.jsonl -> records.jsonl (dedup by idem_key, keep write order)
# Also used as the merge core of run_experiment.py --finalize (merge_part_files)
# Run: python scripts/merge_records.py [--out-dir data/results] [--output records.jsonl]

import argparse
import glob
import json
import os
import sys
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def list_part_files(out_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(out_dir, "part-*.jsonl")))


def merge_part_files(out_dir: str, output: str = "records.jsonl") -> Tuple[str, int, int, int]:
    # Merge shards -> one file; returns (output path, records, dropped dupes, bad lines)
    files = list_part_files(out_dir)
    if not files:
        raise SystemExit(f"no shard files found: {out_dir}/part-*.jsonl")
    seen = set()
    total = 0
    dropped = 0
    bad = 0
    out_path = os.path.join(out_dir, output)
    with open(out_path, "w", encoding="utf-8") as out:
        for path in files:
            # errors="replace": disk-full / interrupted writes may leave half a UTF-8 char
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        bad += 1  # half-written trailing line: skip (idempotent rerun after restart)
                        continue
                    key = d.get("idem_key")
                    if key in seen:
                        dropped += 1  # idempotent dedup (same key across shards: first writer wins)
                        continue
                    seen.add(key)
                    out.write(json.dumps(d, ensure_ascii=False) + "\n")
                    total += 1
    return out_path, total, dropped, bad


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge incremental record shards part-N.jsonl -> records.jsonl")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "results"))
    ap.add_argument("--output", default="records.jsonl")
    args = ap.parse_args()

    out_path, total, dropped, bad = merge_part_files(args.out_dir, args.output)
    print(f"Merged -> {out_path}: {total} records (dropped {dropped} dupes, {bad} bad lines)")


if __name__ == "__main__":
    main()
