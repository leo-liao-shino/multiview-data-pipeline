"""
Sample a balanced subset from prompts.jsonl such that each (category, operation)
pair has exactly N records (default 20). If a pair has fewer than N records,
all of them are kept and a warning is printed.

Usage:
    python scripts/subsample.py [--input FILE] [--output FILE] [--n N] [--seed N]
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

DEFAULT_INPUT  = "/workspace/multiview-data-pipeline/resume/prompts.jsonl"
DEFAULT_OUTPUT = "/workspace/multiview-data-pipeline/resume/prompts_subset.jsonl"
DEFAULT_N      = 20


def main():
    parser = argparse.ArgumentParser(description="Sample N records per (category, operation) pair")
    parser.add_argument("--input",  default=DEFAULT_INPUT,  help="Input JSONL file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSONL file")
    parser.add_argument("--n",      type=int, default=DEFAULT_N, help="Records per (category, operation) pair")
    parser.add_argument("--seed",   type=int, default=42,        help="Random seed")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # ------------------------------------------------------------------
    # Load all records and bucket by (category, operation)
    # ------------------------------------------------------------------
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    total = 0
    skipped = 0

    with open(input_path, "r") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = (rec["category"], rec["operation"])
                buckets[key].append(rec)
                total += 1
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  [WARN] Skipping line {lineno}: {e}")
                skipped += 1

    print(f"Loaded {total} records ({skipped} skipped) across {len(buckets)} (category, operation) pairs.")

    # ------------------------------------------------------------------
    # Sample N per bucket
    # ------------------------------------------------------------------
    random.seed(args.seed)
    subset: list[dict] = []
    short_pairs: list[tuple] = []

    for key in sorted(buckets.keys()):
        records = buckets[key]
        category, operation = key
        if len(records) < args.n:
            short_pairs.append((category, operation, len(records)))
            subset.extend(records)
        else:
            subset.extend(random.sample(records, args.n))

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    if short_pairs:
        print(f"\n[WARN] {len(short_pairs)} pair(s) had fewer than {args.n} records (kept all):")
        for cat, op, count in short_pairs:
            print(f"  {cat:30s} / {op:15s} — {count} records")

    # Summary table
    print(f"\n{'Category':<30} {'Operation':<15} {'Available':>9} {'Sampled':>7}")
    print("-" * 65)
    for key in sorted(buckets.keys()):
        cat, op = key
        available = len(buckets[key])
        sampled   = min(available, args.n)
        print(f"  {cat:<28} {op:<15} {available:>9} {sampled:>7}")

    print("-" * 65)
    print(f"  {'TOTAL':<43} {total:>9} {len(subset):>7}")

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in subset:
            f.write(json.dumps(rec) + "\n")

    print(f"\nSubset written to: {output_path}  ({len(subset)} records)")


if __name__ == "__main__":
    main()