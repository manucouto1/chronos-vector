#!/usr/bin/env python3
"""Preprocess RSDD dataset into unified JSONL format.

RSDD (Reddit Self-reported Depression Diagnosis) format:
- Each line is a JSON array with one user object: [{"id": int, "label": str, "posts": [[ts, text], ...]}]
- Files are large (~6 GB each), must stream line-by-line
- Labels: "depression", "control", null (filtered)
- Timestamps: Unix epoch seconds

Usage:
    python scripts/preprocess_rsdd.py \
        --input data/RSDD/ \
        --output data/RSDD/unified.jsonl \
        --max-users 500
"""

import argparse
import json
from datetime import datetime
from pathlib import Path


def process_rsdd_file(file_path: Path, max_users: int | None, min_posts: int) -> list[dict]:
    """Stream-process an RSDD file, yielding user records."""
    records = []
    n_users = 0
    n_skipped_label = 0
    n_skipped_posts = 0

    with open(file_path) as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                if isinstance(data, list):
                    data = data[0]
            except json.JSONDecodeError:
                continue

            label = data.get("label")
            if label not in ("depression", "control"):
                n_skipped_label += 1
                continue

            posts = data.get("posts", [])
            if len(posts) < min_posts:
                n_skipped_posts += 1
                continue

            user_id = str(data.get("id", line_num))

            for ts_epoch, text in posts:
                if not text or not isinstance(text, str):
                    continue
                text = text.strip()
                if len(text) < 10:  # skip very short posts
                    continue

                try:
                    ts = datetime.fromtimestamp(int(ts_epoch)).isoformat()
                except (ValueError, OSError, OverflowError):
                    continue

                records.append({
                    "user_id": user_id,
                    "timestamp": ts,
                    "text": text[:2000],  # cap text length
                    "label": label,
                })

            n_users += 1
            if max_users and n_users >= max_users:
                break

            if n_users % 1000 == 0:
                print(f"  Processed {n_users} users, {len(records)} records...")

    print(f"  File: {file_path.name} — {n_users} users, {len(records)} records "
          f"(skipped: {n_skipped_label} no-label, {n_skipped_posts} too-few-posts)")
    return records


def main():
    parser = argparse.ArgumentParser(description="Preprocess RSDD data")
    parser.add_argument("--input", required=True, help="RSDD directory")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--max-users", type=int, default=None, help="Max users per file (for testing)")
    parser.add_argument("--min-posts", type=int, default=10, help="Min posts per user")
    args = parser.parse_args()

    input_dir = Path(args.input)
    all_records = []

    for fname in ["training-002", "validation-001", "testing-003"]:
        fpath = input_dir / fname
        if fpath.exists():
            print(f"Processing {fname}...")
            records = process_rsdd_file(fpath, args.max_users, args.min_posts)
            all_records.extend(records)

    # Sort by user then timestamp
    all_records.sort(key=lambda r: (r["user_id"], r["timestamp"]))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    n_users = len(set(r["user_id"] for r in all_records))
    n_dep = len(set(r["user_id"] for r in all_records if r["label"] == "depression"))
    print(f"\nDone: {len(all_records)} records, {n_users} users "
          f"({n_dep} depression, {n_users - n_dep} control)")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
