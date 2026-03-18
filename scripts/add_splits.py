#!/usr/bin/env python3
"""Add train/val/test split labels to embedding Parquet files.

Strategy (v2):
- eRisk: pool 2017+2018 → 80/20 stratified train/val, 2022 → test
- RSDD: pool training+validation → 80/20 stratified train/val, testing → test
- CLPsych: user-level stratified 70/15/15 split (not temporal)

Usage:
    python scripts/add_splits.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42


def build_erisk_split_map(erisk_dir: Path) -> dict[str, str]:
    """Map eRisk user_ids: 2017+2018 → pool (train/val), 2022 → test."""
    pool_users = {}   # user_id → label (for stratification)
    test_users = set()

    # 2017 train
    gt = erisk_dir / "2017 (train and test)/2017/depression_train_2017/risk_golden_truth.txt"
    if gt.exists():
        for line in open(gt):
            parts = line.strip().split()
            if len(parts) >= 2:
                pool_users[parts[0]] = int(parts[1])

    # 2017 test → pool
    gt = erisk_dir / "2017 (train and test)/2017/depression_test_2017/test_golden_truth.txt"
    if gt.exists():
        for line in open(gt):
            parts = line.strip().split()
            if len(parts) >= 2:
                pool_users[parts[0]] = int(parts[1])

    # 2018 → pool
    gt = erisk_dir / "2018 (test cases)/task 1 - depression (test split, train split is 2017 data)/risk-golden-truth-test.txt"
    if gt.exists():
        for line in open(gt):
            parts = line.strip().split()
            if len(parts) >= 2:
                pool_users[parts[0]] = int(parts[1])

    # 2022 → test
    gt = erisk_dir / "2022 (test cases)/test data/risk_golden_truth.txt"
    if gt.exists():
        for line in open(gt):
            parts = line.strip().split()
            if len(parts) >= 2:
                test_users.add(parts[0])

    # Stratified 80/20 split of pool
    uids = list(pool_users.keys())
    labels = [pool_users[u] for u in uids]
    train_uids, val_uids = train_test_split(
        uids, test_size=0.2, stratify=labels, random_state=SEED)

    split_map = {}
    for u in train_uids:
        split_map[u] = "train"
    for u in val_uids:
        split_map[u] = "val"
    for u in test_users:
        split_map[u] = "test"

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
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        n_dep = sub[sub["label"] == "depression"]["user_id"].nunique()
        n_ctrl = sub[sub["label"] == "control"]["user_id"].nunique()
        print(f"  {split:>5}: {n_dep} dep + {n_ctrl} ctrl = {n_dep + n_ctrl} users")
    df.to_parquet(parquet, index=False)
    print(f"  Updated {parquet}")


def add_rsdd_splits():
    """Add split column to RSDD embeddings.

    Pool training-002 + validation-001 → 80/20 stratified train/val.
    testing-003 → test.
    """
    parquet = Path("data/embeddings/rsdd_embeddings.parquet")
    if not parquet.exists():
        print("RSDD parquet not found, skipping")
        return

    rsdd_dir = Path("data/RSDD")

    # Build user → (original_split, label) map
    file_map = {}
    for fname, orig in [("training-002", "pool"), ("validation-001", "pool"), ("testing-003", "test")]:
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
                        file_map[uid] = orig
                except json.JSONDecodeError:
                    continue

    df = pd.read_parquet(parquet)

    # Get labels per user
    user_labels = df.groupby("user_id")["label"].first()

    # Pool users → stratified 80/20
    pool_uids = [u for u, s in file_map.items() if s == "pool" and u in user_labels.index]
    pool_labels = [user_labels[u] for u in pool_uids]

    train_uids, val_uids = train_test_split(
        pool_uids, test_size=0.2, stratify=pool_labels, random_state=SEED)

    split_map = {}
    for u in train_uids:
        split_map[u] = "train"
    for u in val_uids:
        split_map[u] = "val"
    for u, s in file_map.items():
        if s == "test":
            split_map[u] = "test"

    df["split"] = df["user_id"].map(split_map).fillna("unknown")
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        n_dep = sub[sub["label"] == "depression"]["user_id"].nunique()
        n_ctrl = sub[sub["label"] == "control"]["user_id"].nunique()
        print(f"  {split:>5}: {n_dep} dep + {n_ctrl} ctrl = {n_dep + n_ctrl} users")
    df.to_parquet(parquet, index=False)
    print(f"  Updated {parquet}")


def add_clpsych_splits():
    """Add split column to CLPsych embeddings.

    User-level stratified 70/15/15 split (NOT temporal).
    All posts from a user go into the same split.
    """
    parquet = Path("data/embeddings/clpsych_embeddings.parquet")
    if not parquet.exists():
        print("CLPsych parquet not found, skipping")
        return

    df = pd.read_parquet(parquet)

    # Get unique users and their labels
    user_labels = df.groupby("user_id")["label"].first()
    uids = user_labels.index.tolist()
    labels = user_labels.values.tolist()

    # Stratified split: 70% train, 15% val, 15% test
    train_uids, temp_uids, _, temp_labels = train_test_split(
        uids, labels, test_size=0.30, stratify=labels, random_state=SEED)
    val_uids, test_uids = train_test_split(
        temp_uids, test_size=0.50, stratify=temp_labels, random_state=SEED)

    split_map = {}
    for u in train_uids:
        split_map[u] = "train"
    for u in val_uids:
        split_map[u] = "val"
    for u in test_uids:
        split_map[u] = "test"

    df["split"] = df["user_id"].map(split_map).fillna("unknown")
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        n_dep = sub[sub["label"] == "depression"]["user_id"].nunique()
        n_ctrl = sub[sub["label"] == "control"]["user_id"].nunique()
        print(f"  {split:>5}: {n_dep} dep + {n_ctrl} ctrl = {n_dep + n_ctrl} users")
    df.to_parquet(parquet, index=False)
    print(f"  Updated {parquet}")


def add_splits_to_parquet(parquet_path: Path, split_map: dict[str, str]):
    """Apply a split map to any parquet file."""
    if not parquet_path.exists():
        return
    df = pd.read_parquet(parquet_path)
    df["split"] = df["user_id"].map(split_map).fillna("unknown")
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        n_dep = sub[sub["label"] == "depression"]["user_id"].nunique()
        n_ctrl = sub[sub["label"] == "control"]["user_id"].nunique()
        print(f"  {split:>5}: {n_dep} dep + {n_ctrl} ctrl = {n_dep + n_ctrl} users")
    df.to_parquet(parquet_path, index=False)
    print(f"  Updated {parquet_path}")


if __name__ == "__main__":
    # Process both original and mental_ parquets
    print("eRisk (pool 2017+2018, 80/20 train/val, 2022 test):")
    add_erisk_splits()
    # Also apply to mental_ version if it exists
    mental = Path("data/embeddings/erisk_mental_embeddings.parquet")
    if mental.exists():
        print(f"\n  Also applying to {mental}...")
        split_map = build_erisk_split_map(Path("data/eRisk"))
        add_splits_to_parquet(mental, split_map)

    print("\nRSDD (pool train+val files, 80/20, testing file = test):")
    add_rsdd_splits()

    print("\nCLPsych (user-level stratified 70/15/15):")
    add_clpsych_splits()

    # Apply to mental_ versions
    for ds in ["clpsych", "rsdd"]:
        mental = Path(f"data/embeddings/{ds}_mental_embeddings.parquet")
        if mental.exists():
            print(f"\n  Also applying splits to {mental}...")
            df_orig = pd.read_parquet(f"data/embeddings/{ds}_embeddings.parquet")
            split_map_ds = dict(zip(df_orig["user_id"],df_orig["split"]))
            add_splits_to_parquet(mental, split_map_ds)

    print("\nDone.")
