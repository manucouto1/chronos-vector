#!/usr/bin/env python3
"""Generate MentalRoBERTa embeddings + relative temporal features.

Changes from v1:
- Uses mental/mental-roberta-base (D=768) instead of MiniLM (D=384)
- Computes per-post temporal features: t_rel, gap_seconds, gap_log, hour_of_day
- Computes per-user behavioral features and stores as separate Parquet

Usage (on HPC with GPU):
    python scripts/generate_embeddings_v2.py --dataset erisk
    python scripts/generate_embeddings_v2.py --dataset clpsych
    python scripts/generate_embeddings_v2.py --dataset rsdd
    python scripts/generate_embeddings_v2.py --dataset all

Requirements:
    pip install sentence-transformers pandas pyarrow transformers torch
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


# ── Dataset config ─────────────────────────────────────────────────
DATASETS = {
    "erisk": {
        "input": "data/erisk_unified.jsonl",
        "output": "data/embeddings/erisk_mental_embeddings.parquet",
    },
    "clpsych": {
        "input": "data/clpsych_unified.jsonl",
        "output": "data/embeddings/clpsych_mental_embeddings.parquet",
    },
    "rsdd": {
        "input": "data/rsdd_unified.jsonl",
        "output": "data/embeddings/rsdd_mental_embeddings.parquet",
    },
}

MODEL_NAME = "mental/mental-roberta-base"
BATCH_SIZE = 128  # Smaller than v1: RoBERTa is larger than MiniLM


def load_jsonl(path: Path) -> list[dict]:
    """Load and sanitize JSONL records."""
    records = []
    n_filtered = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            text = r.get("text")
            if text is None:
                n_filtered += 1
                continue
            if not isinstance(text, str):
                text = str(text)
            text = text.encode("utf-8", errors="replace").decode("utf-8")
            text = text.strip().replace("\x00", "")
            if len(text) == 0:
                n_filtered += 1
                continue
            r["text"] = text[:512]  # RoBERTa max is 512 tokens; cap chars
            records.append(r)
    if n_filtered > 0:
        print(f"  Filtered {n_filtered} invalid/empty texts")
    return records


def generate_embeddings(texts: list[str], model_name: str,
                        batch_size: int) -> np.ndarray:
    """Generate embeddings with mean pooling + L2 normalization.

    Works with any HuggingFace model (including MentalRoBERTa which is
    NOT a sentence-transformer but a masked LM).
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading model: {model_name} (device={device})")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    print(f"  Encoding {len(texts):,} texts (batch_size={batch_size})...")
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=512,
            return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**encoded)

        # Mean pooling over non-padding tokens
        attention_mask = encoded["attention_mask"].unsqueeze(-1)
        token_embeddings = outputs.last_hidden_state
        summed = (token_embeddings * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1)
        mean_pooled = summed / counts

        # L2 normalize
        mean_pooled = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
        all_embeddings.append(mean_pooled.cpu().numpy())

        if (i // batch_size) % 50 == 0:
            print(f"    {i + len(batch):,}/{len(texts):,} "
                  f"({100*(i+len(batch))/len(texts):.0f}%)")

    embeddings = np.vstack(all_embeddings)
    print(f"  Shape: {embeddings.shape}")
    return embeddings


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-post relative temporal features.

    Features:
    - t_rel: normalized time since first post [0, 1]
    - gap_seconds: seconds since previous post (0 for first post)
    - gap_log: log(1 + gap_seconds)
    - post_index: sequential position within user
    - hour_of_day: hour extracted from timestamp (circadian)
    """
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # Per-user relative time
    first_ts = df.groupby("user_id")["timestamp"].transform("min")
    last_ts = df.groupby("user_id")["timestamp"].transform("max")
    span = (last_ts - first_ts).dt.total_seconds().replace(0, 1)  # avoid /0
    df["t_rel"] = ((df["timestamp"] - first_ts).dt.total_seconds() / span
                   ).astype(np.float32)

    # Inter-post gap
    df["gap_seconds"] = (df.groupby("user_id")["timestamp"]
                         .diff()
                         .dt.total_seconds()
                         .fillna(0)
                         .astype(np.float32))
    df["gap_log"] = np.log1p(df["gap_seconds"]).astype(np.float32)

    # Sequential index within user
    df["post_index"] = df.groupby("user_id").cumcount().astype(np.int32)

    # Circadian (fillna for missing timestamps)
    df["hour_of_day"] = df["timestamp"].dt.hour.fillna(0).astype(np.int8)

    return df


def compute_user_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-user behavioral features from posting patterns.

    Returns a DataFrame with one row per user.
    """
    user_feats = []
    for uid, group in df.groupby("user_id"):
        g = group.sort_values("timestamp")
        gaps = g["gap_seconds"].values[1:]  # skip first (=0)
        n_posts = len(g)

        feat = {
            "user_id": uid,
            "label": g["label"].iloc[0],
            "n_posts": n_posts,
            "span_days": (g["timestamp"].max() - g["timestamp"].min()
                          ).total_seconds() / 86400,
        }

        if len(gaps) > 0:
            feat["mean_gap"] = float(np.mean(gaps))
            feat["std_gap"] = float(np.std(gaps))
            feat["max_gap"] = float(np.max(gaps))
            feat["median_gap"] = float(np.median(gaps))
            feat["gap_cv"] = (float(np.std(gaps) / np.mean(gaps))
                              if np.mean(gaps) > 0 else 0.0)
            # Gap trend: slope of gap over post index
            if len(gaps) > 2:
                x = np.arange(len(gaps), dtype=np.float64)
                feat["gap_trend"] = float(np.polyfit(x, gaps, 1)[0])
            else:
                feat["gap_trend"] = 0.0
        else:
            for k in ["mean_gap", "std_gap", "max_gap", "median_gap",
                       "gap_cv", "gap_trend"]:
                feat[k] = 0.0

        # Circadian
        hours = g["hour_of_day"].values
        feat["night_ratio"] = float(np.mean((hours >= 0) & (hours < 6)))
        feat["evening_ratio"] = float(np.mean((hours >= 18) & (hours < 24)))

        # Posting bursts: >5 posts within 1 hour
        ts_arr = g["timestamp"].values.astype("int64") // 10**9  # unix seconds
        burst_count = 0
        for i in range(len(ts_arr)):
            window_end = ts_arr[i] + 3600
            n_in_window = np.searchsorted(ts_arr, window_end, side="right") - i
            if n_in_window >= 5:
                burst_count += 1
        feat["burst_count"] = burst_count

        user_feats.append(feat)

    return pd.DataFrame(user_feats)


def process_dataset(name: str, cfg: dict, model_name: str, batch_size: int):
    """Full pipeline for one dataset."""
    input_path = Path(cfg["input"])
    output_path = Path(cfg["output"])
    behavioral_path = output_path.with_name(
        output_path.stem.replace("_mental_embeddings", "_behavioral") + ".parquet")

    if not input_path.exists():
        print(f"  Input not found: {input_path}, skipping")
        return

    # Load
    print(f"  Loading {input_path}...")
    records = load_jsonl(input_path)
    print(f"  {len(records):,} records")

    # Embed
    texts = [r["text"] for r in records]
    embeddings = generate_embeddings(texts, model_name, batch_size)

    # Build DataFrame
    df = pd.DataFrame({
        "user_id": [r["user_id"] for r in records],
        "timestamp": pd.to_datetime([r["timestamp"] for r in records]),
        "label": [r["label"] for r in records],
        "text_len": [len(r["text"]) for r in records],
    })

    # Add embedding columns
    for dim_idx in range(embeddings.shape[1]):
        df[f"emb_{dim_idx}"] = embeddings[:, dim_idx].astype(np.float32)
    del embeddings  # free memory

    # Add temporal features
    print("  Computing temporal features...")
    df = compute_temporal_features(df)

    # Sort and save
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    n_users = df["user_id"].nunique()
    n_dep = df[df["label"] == "depression"]["user_id"].nunique()
    dim = len([c for c in df.columns if c.startswith("emb_")])

    print(f"\n  Saved: {output_path}")
    print(f"  Records: {len(df):,}")
    print(f"  Users: {n_users} ({n_dep} depression, {n_users - n_dep} control)")
    print(f"  Embedding dim: {dim}")
    print(f"  Temporal cols: t_rel, gap_seconds, gap_log, post_index, hour_of_day")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Behavioral features (separate file, one row per user)
    print("  Computing behavioral features...")
    behav_df = compute_user_behavioral_features(df)
    behav_path = output_path.with_name(
        output_path.stem.replace("_embeddings", "_behavioral") + ".parquet")
    behav_df.to_parquet(behav_path, index=False)
    print(f"  Saved: {behav_path} ({len(behav_df)} users)")

    # Quick discriminative check on behavioral features
    dep = behav_df[behav_df["label"] == "depression"]
    ctrl = behav_df[behav_df["label"] == "control"]
    print("\n  Behavioral feature check (dep vs ctrl means):")
    for col in ["mean_gap", "std_gap", "max_gap", "gap_cv", "gap_trend",
                "night_ratio", "burst_count", "n_posts"]:
        if col in dep.columns:
            d_mean = dep[col].mean()
            c_mean = ctrl[col].mean()
            print(f"    {col:>15}: dep={d_mean:.2f}, ctrl={c_mean:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate MentalRoBERTa embeddings + temporal features")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Dataset to process (default: all)",
    )
    parser.add_argument(
        "--model", default=MODEL_NAME,
        help=f"Model name (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    args = parser.parse_args()

    datasets = (DATASETS.keys() if args.dataset == "all"
                else [args.dataset])

    for name in datasets:
        print(f"\n{'='*60}")
        print(f"Processing: {name}")
        print(f"{'='*60}")
        process_dataset(name, DATASETS[name], args.model, args.batch_size)

    print(f"\n{'='*60}")
    print("Done. Next: run scripts/add_splits.py on the new parquets.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
