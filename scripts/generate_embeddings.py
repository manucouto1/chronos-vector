#!/usr/bin/env python3
"""Generate sentence embeddings from preprocessed JSONL data.

Reads unified JSONL format (user_id, timestamp, text, label) and produces
a Parquet file with embeddings ready for ChronosVector ingestion.

Usage:
    python scripts/generate_embeddings.py \
        --input data/erisk/unified.jsonl \
        --output data/embeddings/erisk_embeddings.parquet \
        --model sentence-transformers/all-MiniLM-L6-v2 \
        --batch-size 256

Requirements:
    pip install sentence-transformers pandas pyarrow
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL records."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def generate_embeddings(records: list[dict], model_name: str, batch_size: int) -> np.ndarray:
    """Generate embeddings using sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    # Sanitize all texts: force string, strip, filter empty/None
    valid_records = []
    n_filtered = 0
    for r in records:
        text = r.get("text")
        if text is None:
            n_filtered += 1
            continue
        # Force to string (handles ints, floats, lists, etc.)
        if not isinstance(text, str):
            text = str(text)
        # Remove surrogate characters (broken emoji from Twitter API) and null bytes
        text = text.encode("utf-8", errors="replace").decode("utf-8")
        text = text.strip().replace("\x00", "")
        if len(text) == 0:
            n_filtered += 1
            continue
        r["text"] = text[:10000]  # cap length to avoid tokenizer OOM
        valid_records.append(r)

    if n_filtered > 0:
        print(f"Filtered {n_filtered} invalid/empty texts")
    records.clear()
    records.extend(valid_records)

    texts = [r["text"] for r in records]
    print(f"Encoding {len(texts)} texts (batch_size={batch_size})...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2 normalize for cosine similarity
    )

    print(f"Embedding shape: {embeddings.shape}")
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Generate sentence embeddings")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output Parquet file")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformer model (default: all-MiniLM-L6-v2, dim=384)",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Encoding batch size")
    args = parser.parse_args()

    records = load_jsonl(Path(args.input))
    print(f"Loaded {len(records)} records")

    embeddings = generate_embeddings(records, args.model, args.batch_size)

    # Build DataFrame
    df = pd.DataFrame({
        "user_id": [r["user_id"] for r in records],
        "timestamp": pd.to_datetime([r["timestamp"] for r in records]),
        "label": [r["label"] for r in records],
        "text_len": [len(r["text"]) for r in records],
    })

    # Store embeddings as list columns
    for dim_idx in range(embeddings.shape[1]):
        df[f"emb_{dim_idx}"] = embeddings[:, dim_idx].astype(np.float32)

    # Sort by user and timestamp
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    n_users = df["user_id"].nunique()
    n_dep = df[df["label"] == "depression"]["user_id"].nunique()
    dim = embeddings.shape[1]

    print(f"\nSaved: {output_path}")
    print(f"  Records: {len(df)}")
    print(f"  Users: {n_users} ({n_dep} depression, {n_users - n_dep} control)")
    print(f"  Embedding dim: {dim}")
    print(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
