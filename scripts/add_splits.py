#!/usr/bin/env python3
"""Add train/val/test split labels to embedding Parquet files.

Determines split membership from dataset structure:
- eRisk: user_id prefix (train_*, test_*) + golden truth year
- CLPsych: chunk index (0-50 = train, 60-89 = test)
- RSDD: source file (training-002, validation-001, testing-003)

Usage:
    python scripts/add_splits.py
"""

import json
from pathlib import Path
from collections import defaultdict

import pandas as pd


def build_erisk_split_map(erisk_dir: Path) -> dict[str, str]:
    """Map eRisk user_ids to splits based on golden truth files."""
    split_map = {}

    # 2017 train
    gt = erisk_dir / "2017 (train and test)/2017/depression_train_2017/risk_golden_truth.txt"
    if gt.exists():
        for line in open(gt):
            parts = line.strip().split()
            if len(parts) >= 2:
                split_map[parts[0]] = "train"

    # 2017 test → validation (since 2022 is the real test)
    gt = erisk_dir / "2017 (train and test)/2017/depression_test_2017/test_golden_truth.txt"
    if gt.exists():
        for line in open(gt):
            parts = line.strip().split()
            if len(parts) >= 2:
                split_map[parts[0]] = "val"

    # 2018 → validation
    gt = erisk_dir / "2018 (test cases)/task 1 - depression (test split, train split is 2017 data)/risk-golden-truth-test.txt"
    if gt.exists():
        for line in open(gt):
            parts = line.strip().split()
            if len(parts) >= 2:
                split_map[parts[0]] = "val"

    # 2022 → test
    gt = erisk_dir / "2022 (test cases)/test data/risk_golden_truth.txt"
    if gt.exists():
        for line in open(gt):
            parts = line.strip().split()
            if len(parts) >= 2:
                split_map[parts[0]] = "test"

    return split_map


def add_erisk_splits():
    """Add split column to eRisk embeddings."""
    parquet = Path("data/embeddings/erisk_embeddings.parquet")
    if not parquet.exists():
        print("eRisk parquet not found, skipping")
        return

    df = pd.read_parquet(parquet)
    split_map = build_erisk_split_map(Path("data/eRisk"))

    df["split"] = df["user_id"].map(split_map).fillna("unknown")
    counts = df.groupby("split")["user_id"].nunique()
    print(f"eRisk splits: {dict(counts)}")
    df.to_parquet(parquet, index=False)
    print(f"  Updated {parquet}")


def add_rsdd_splits():
    """Add split column to RSDD embeddings.

    RSDD unified.jsonl was created from 3 files in order:
    training-002 (first 500 users), validation-001 (next 500), testing-003 (next 500).
    We re-read the source files to get the mapping.
    """
    parquet = Path("data/embeddings/rsdd_embeddings.parquet")
    if not parquet.exists():
        print("RSDD parquet not found, skipping")
        return

    # Build user→split map from source files
    rsdd_dir = Path("data/RSDD")
    split_map = {}
    for fname, split in [("training-002", "train"), ("validation-001", "val"), ("testing-003", "test")]:
        fpath = rsdd_dir / fname
        if not fpath.exists():
            continue
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if isinstance(data, list):
                        data = data[0]
                    uid = str(data.get("id", ""))
                    if uid:
                        split_map[uid] = split
                except json.JSONDecodeError:
                    continue

    df = pd.read_parquet(parquet)
    df["split"] = df["user_id"].map(split_map).fillna("unknown")
    counts = df.groupby("split")["user_id"].nunique()
    print(f"RSDD splits: {dict(counts)}")
    df.to_parquet(parquet, index=False)
    print(f"  Updated {parquet}")


def add_clpsych_splits():
    """Add split column to CLPsych embeddings.

    The JSONL was built from tgz archives: chunks 0-50 = train, 60-89 = test.
    Since all users appear in all chunks, we use the chunk index metadata.
    For CLPsych we split by time: first 70% of each user's posts = train, last 30% = test.
    """
    parquet = Path("data/embeddings/clpsych_embeddings.parquet")
    if not parquet.exists():
        print("CLPsych parquet not found, skipping")
        return

    df = pd.read_parquet(parquet)

    # Temporal split per user: 70% train, 15% val, 15% test
    splits = []
    for uid, group in df.groupby("user_id"):
        n = len(group)
        group_sorted = group.sort_values("timestamp")
        s = ["train"] * int(n * 0.7) + ["val"] * int(n * 0.15) + ["test"] * (n - int(n * 0.7) - int(n * 0.15))
        splits.extend(s)

    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    df["split"] = splits
    counts = df.groupby("split")["user_id"].nunique()
    print(f"CLPsych splits: {dict(counts)}")
    df.to_parquet(parquet, index=False)
    print(f"  Updated {parquet}")


if __name__ == "__main__":
    add_erisk_splits()
    add_rsdd_splits()
    add_clpsych_splits()
    print("\nDone.")
