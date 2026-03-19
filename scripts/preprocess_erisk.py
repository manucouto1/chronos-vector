#!/usr/bin/env python3
"""Preprocess eRisk dataset into unified JSONL format.

eRisk format varies by year:
- 2017: XML files in chunk subdirectories, one per user per chunk
- 2018: ZIP archives per chunk (needs extraction)
- 2022: One XML per user, all in datos/ directory

Usage:
    python scripts/preprocess_erisk.py \
        --input data/eRisk/ \
        --output data/eRisk/unified.jsonl
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def detect_edition(xml_path: Path) -> str:
    """Detect eRisk edition year from directory structure."""
    path_str = str(xml_path)
    if "2017" in path_str:
        if "train" in path_str.lower():
            return "e2017train"
        return "e2017test"
    elif "2018" in path_str:
        return "e2018"
    elif "2022" in path_str:
        return "e2022"
    return "eunknown"


def parse_erisk_xml(xml_path: Path, edition_prefix: str = "") -> list[dict]:
    """Parse a single eRisk XML file, returning (user_id, writings).

    Args:
        edition_prefix: If non-empty, prepend to user_id to avoid collisions
            between editions (e.g., 'e2018_' + 'subject1093' → 'e2018_subject1093').
            This is critical because user_ids are NOT persistent across editions —
            the same ID in different years refers to different individuals.
    """
    records = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Extract subject ID from <ID> tag or filename
        id_el = root.find(".//ID")
        raw_id = id_el.text.strip() if id_el is not None and id_el.text else xml_path.stem.split("_chunk")[0]
        subject_id = f"{edition_prefix}_{raw_id}" if edition_prefix else raw_id

        for writing in root.findall(".//WRITING"):
            title_el = writing.find("TITLE")
            text_el = writing.find("TEXT")
            date_el = writing.find("DATE")

            title = title_el.text.strip() if title_el is not None and title_el.text else ""
            text = text_el.text.strip() if text_el is not None and text_el.text else ""
            date_str = date_el.text.strip() if date_el is not None and date_el.text else ""

            content = f"{title} {text}".strip()
            if not content or len(content) < 10:
                continue

            records.append({
                "user_id": subject_id,
                "timestamp": date_str,
                "text": content[:2000],
            })
    except ET.ParseError:
        pass
    return records


def load_golden_truth(dir_path: Path) -> dict[str, str]:
    """Search for golden truth files and build label map."""
    labels = {}
    for gt_file in dir_path.rglob("*golden*truth*"):
        if gt_file.is_file():
            with open(gt_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        subject_id = parts[0]
                        try:
                            label_val = int(parts[1])
                            labels[subject_id] = "depression" if label_val == 1 else "control"
                        except ValueError:
                            pass
    return labels


def process_erisk_dir(base_dir: Path) -> list[dict]:
    """Process all eRisk XML files recursively."""
    labels = load_golden_truth(base_dir)
    print(f"  Loaded {len(labels)} labels from golden truth files")

    all_records = []
    seen_user_records = {}  # user_id -> set of (timestamp, text_hash) to deduplicate

    for xml_path in sorted(base_dir.rglob("*.xml")):
        edition = detect_edition(xml_path)
        records = parse_erisk_xml(xml_path, edition_prefix=edition)
        for r in records:
            uid = r["user_id"]
            dedup_key = (r["timestamp"], hash(r["text"][:100]))

            if uid not in seen_user_records:
                seen_user_records[uid] = set()
            if dedup_key in seen_user_records[uid]:
                continue
            seen_user_records[uid].add(dedup_key)

            # Assign label if known
            label = labels.get(uid, "unknown")
            # Try partial matches (some IDs have prefixes)
            if label == "unknown":
                for key, val in labels.items():
                    if key in uid or uid in key:
                        label = val
                        break

            r["label"] = label
            all_records.append(r)

    return all_records


def main():
    parser = argparse.ArgumentParser(description="Preprocess eRisk data")
    parser.add_argument("--input", required=True, help="eRisk root directory")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args()

    base_dir = Path(args.input)
    print(f"Scanning {base_dir}...")
    all_records = process_erisk_dir(base_dir)

    # Filter out unknown labels
    known = [r for r in all_records if r["label"] != "unknown"]
    unknown_count = len(all_records) - len(known)
    if unknown_count:
        print(f"  Filtered {unknown_count} records with unknown labels")
    all_records = known

    # Sort by user then timestamp
    all_records.sort(key=lambda r: (r["user_id"], r["timestamp"]))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    n_users = len(set(r["user_id"] for r in all_records))
    n_dep = len(set(r["user_id"] for r in all_records if r["label"] == "depression"))
    n_ctrl = len(set(r["user_id"] for r in all_records if r["label"] == "control"))
    print(f"\nDone: {len(all_records)} records, {n_users} users "
          f"({n_dep} depression, {n_ctrl} control)")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
