#!/usr/bin/env python3
"""Preprocess CLPsych 2015 dataset into unified JSONL format.

CLPsych15 format:
- CSV metadata: anonymized_user_info_by_chunk.csv (user labels + demographics)
- Tweet data: .tgz archives per chunk, each containing .tweets files (one per user)
- Each .tweets file has one Twitter JSON object per line
- Three conditions: depression, ptsd, control

Usage:
    python scripts/preprocess_clpsych.py \
        --input data/CLPsych15/ \
        --output data/CLPsych15/unified.jsonl
"""

import argparse
import csv
import json
import os
import tarfile
from datetime import datetime
from pathlib import Path


TWITTER_DATE_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def load_user_labels(csv_path: Path) -> dict[str, str]:
    """Load user labels from the metadata CSV."""
    labels = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            screen_name = row.get("anonymized_screen_name", "").strip()
            condition = row.get("condition", "").strip().lower()
            if screen_name and condition in ("depression", "ptsd", "control"):
                labels[screen_name] = condition
    return labels


def parse_tweet_json(line: str) -> dict | None:
    """Parse a single tweet JSON line."""
    try:
        tweet = json.loads(line)
    except json.JSONDecodeError:
        return None

    text = tweet.get("text", "")
    if not text or len(text.strip()) < 10:
        return None

    created_at = tweet.get("created_at", "")
    try:
        ts = datetime.strptime(created_at, TWITTER_DATE_FORMAT)
        timestamp = ts.isoformat()
    except ValueError:
        return None

    return {
        "text": text.strip()[:2000],
        "timestamp": timestamp,
    }


def process_tgz(tgz_path: Path, labels: dict) -> list[dict]:
    """Extract and process a .tgz chunk archive."""
    records = []

    try:
        with tarfile.open(tgz_path, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.name.endswith(".tweets"):
                    continue

                # Extract screen name from filename
                screen_name = Path(member.name).stem

                label = labels.get(screen_name)
                if label is None:
                    continue

                f = tar.extractfile(member)
                if f is None:
                    continue

                for line in f:
                    line = line.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue

                    tweet = parse_tweet_json(line)
                    if tweet is None:
                        continue

                    records.append({
                        "user_id": screen_name,
                        "timestamp": tweet["timestamp"],
                        "text": tweet["text"],
                        "label": label,
                    })
    except (tarfile.TarError, OSError) as e:
        print(f"  Warning: Failed to process {tgz_path}: {e}")

    return records


def main():
    parser = argparse.ArgumentParser(description="Preprocess CLPsych 2015 data")
    parser.add_argument("--input", required=True, help="CLPsych15 root directory")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--conditions", nargs="+", default=["depression", "control"],
                        help="Conditions to include (default: depression control)")
    args = parser.parse_args()

    base_dir = Path(args.input)

    # Find and load labels CSV
    csv_files = list(base_dir.rglob("*user_info*.csv"))
    if not csv_files:
        print("ERROR: Cannot find user info CSV. Expected: anonymized_user_info_by_chunk.csv")
        return

    labels = load_user_labels(csv_files[0])
    print(f"Loaded labels for {len(labels)} users from {csv_files[0].name}")
    for cond in set(labels.values()):
        n = sum(1 for v in labels.values() if v == cond)
        print(f"  {cond}: {n}")

    # Filter to requested conditions
    labels = {k: v for k, v in labels.items() if v in args.conditions}
    print(f"Filtered to conditions {args.conditions}: {len(labels)} users")

    # Process all .tgz archives
    all_records = []
    seen_keys = set()  # deduplicate across chunks

    tgz_files = sorted(base_dir.rglob("*.tgz"))
    print(f"Found {len(tgz_files)} .tgz archives")

    for tgz_path in tgz_files:
        records = process_tgz(tgz_path, labels)
        # Deduplicate (same user + timestamp + text can appear across chunks)
        for r in records:
            key = (r["user_id"], r["timestamp"], hash(r["text"][:50]))
            if key not in seen_keys:
                seen_keys.add(key)
                all_records.append(r)
        print(f"  {tgz_path.name}: +{len(records)} records (total: {len(all_records)})")

    # Sort by user then timestamp
    all_records.sort(key=lambda r: (r["user_id"], r["timestamp"]))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    n_users = len(set(r["user_id"] for r in all_records))
    by_label = {}
    for r in all_records:
        uid = r["user_id"]
        by_label.setdefault(r["label"], set()).add(uid)

    print(f"\nDone: {len(all_records)} records, {n_users} users")
    for label, users in sorted(by_label.items()):
        print(f"  {label}: {len(users)} users")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
