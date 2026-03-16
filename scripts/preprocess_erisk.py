#!/usr/bin/env python3
"""Preprocess eRisk dataset into unified JSONL format.

eRisk data typically comes as XML files with chronologically ordered
writings per subject. This script converts them to our unified format.

Usage:
    python scripts/preprocess_erisk.py \
        --input data/erisk/raw/ \
        --output data/erisk/unified.jsonl \
        --labels data/erisk/golden_truth.txt
"""

import argparse
import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path


def parse_erisk_subject(xml_path: Path, label: str) -> list[dict]:
    """Parse a single eRisk subject XML file."""
    records = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        subject_id = xml_path.stem

        for writing in root.findall(".//WRITING"):
            title_el = writing.find("TITLE")
            text_el = writing.find("TEXT")
            date_el = writing.find("DATE")

            title = title_el.text.strip() if title_el is not None and title_el.text else ""
            text = text_el.text.strip() if text_el is not None and text_el.text else ""
            date_str = date_el.text.strip() if date_el is not None and date_el.text else ""

            content = f"{title} {text}".strip()
            if not content:
                continue

            # Parse date (eRisk format varies: YYYY-MM-DD or epoch)
            try:
                if date_str.isdigit():
                    ts = datetime.fromtimestamp(int(date_str))
                else:
                    ts = datetime.fromisoformat(date_str.replace("/", "-"))
                timestamp = ts.isoformat()
            except (ValueError, OSError):
                timestamp = date_str

            records.append({
                "user_id": subject_id,
                "timestamp": timestamp,
                "text": content,
                "label": label,
            })
    except ET.ParseError as e:
        print(f"  Warning: Failed to parse {xml_path}: {e}")

    return records


def load_labels(labels_path: Path) -> dict[str, str]:
    """Load golden truth labels (subject_id -> label)."""
    labels = {}
    with open(labels_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                subject_id = parts[0]
                label_val = int(parts[1])
                labels[subject_id] = "depression" if label_val == 1 else "control"
    return labels


def main():
    parser = argparse.ArgumentParser(description="Preprocess eRisk data")
    parser.add_argument("--input", required=True, help="Directory with subject XML files")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--labels", required=True, help="Golden truth labels file")
    args = parser.parse_args()

    input_dir = Path(args.input)
    labels = load_labels(Path(args.labels))

    all_records = []
    for xml_file in sorted(input_dir.glob("*.xml")):
        subject_id = xml_file.stem
        label = labels.get(subject_id, "unknown")
        records = parse_erisk_subject(xml_file, label)
        all_records.extend(records)
        if records:
            print(f"  {subject_id}: {len(records)} writings ({label})")

    # Sort by user then timestamp
    all_records.sort(key=lambda r: (r["user_id"], r["timestamp"]))

    with open(args.output, "w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    n_users = len(set(r["user_id"] for r in all_records))
    n_dep = len(set(r["user_id"] for r in all_records if r["label"] == "depression"))
    print(f"\nDone: {len(all_records)} records, {n_users} users ({n_dep} depression, {n_users - n_dep} control)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
