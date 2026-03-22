#!/usr/bin/env python3
"""
Migrate old .cvx index files to the new directory format.

Old format: single postcard binary file (TemporalHnsw snapshot)
New format: directory with index.bin + temporal_edges.bin

Usage:
    python scripts/migrate_index.py data/cache/erisk_index.cvx data/cache/erisk_index/

If the old file can't be loaded (incompatible struct), you'll need to
rebuild from source data. This script handles format migration, not
schema migration.
"""
import sys
import os


def main():
    if len(sys.argv) < 3:
        print("Usage: python migrate_index.py <old_file.cvx> <new_directory>")
        print("\nMigrates a legacy .cvx file to the new directory format.")
        print("If the file can't be loaded due to struct changes, rebuild")
        print("from source data instead.")
        sys.exit(1)

    old_path = sys.argv[1]
    new_path = sys.argv[2]

    if not os.path.isfile(old_path):
        print(f"Error: {old_path} is not a file")
        sys.exit(1)

    if os.path.exists(new_path):
        print(f"Error: {new_path} already exists")
        sys.exit(1)

    try:
        import chronos_vector as cvx
    except ImportError:
        print("Error: chronos_vector not installed.")
        print("Run: maturin develop --release --manifest-path crates/cvx-python/Cargo.toml")
        sys.exit(1)

    print(f"Loading {old_path}...")
    try:
        index = cvx.TemporalIndex.load(old_path)
        print(f"  Loaded: {len(index)} points")
    except Exception as e:
        print(f"  Failed to load: {e}")
        print("\nThe file format is incompatible with the current version.")
        print("You need to rebuild the index from source data.")
        print("\nCommon causes:")
        print("  - TemporalSnapshot struct changed (fields added/removed)")
        print("  - Postcard doesn't support schema evolution")
        print("\nTo rebuild, use the original data (parquet/json) and re-run")
        print("bulk_insert + save.")
        sys.exit(1)

    print(f"Saving to {new_path}/...")
    os.makedirs(new_path, exist_ok=True)
    index.save(new_path)
    print(f"  Saved: {new_path}/index.bin + temporal_edges.bin")

    # Verify
    loaded = cvx.TemporalIndex.load(new_path)
    print(f"  Verified: {len(loaded)} points")
    print("\nMigration complete. You can delete the old .cvx file.")


if __name__ == "__main__":
    main()
