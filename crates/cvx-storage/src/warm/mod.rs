//! Warm storage tier: file-based partitioned storage.
//!
//! Stores temporal points in postcard-serialized files, partitioned by entity_id.
//! Designed for data that's accessed less frequently than hot tier but still
//! needs reasonable read performance.
//!
//! ## Directory Layout
//!
//! ```text
//! warm/
//! ├── entity_0000000042/
//! │   ├── space_0000_chunk_000000.warm
//! │   └── space_0000_chunk_000001.warm
//! └── manifest.json
//! ```
//!
//! Each chunk file contains a sorted sequence of serialized `TemporalPoint`s.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use cvx_core::StorageBackend;
use cvx_core::error::StorageError;
use cvx_core::types::TemporalPoint;
use parking_lot::RwLock;

/// Maximum number of points per chunk file.
const DEFAULT_CHUNK_SIZE: usize = 10_000;

/// Per-chunk metadata for zone map pruning (RFC-002-08).
///
/// Stores min/max timestamps so range queries can skip non-overlapping chunks
/// without deserializing their contents. See Lamb et al. (VLDB 2012).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ChunkMeta {
    filename: String,
    min_timestamp: i64,
    max_timestamp: i64,
    point_count: u32,
}

/// Zone map manifest: per-entity-space chunk metadata.
/// Keys are serialized as `"entity_id:space_id"` strings for JSON compatibility.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
struct ZoneManifest {
    chunks: BTreeMap<String, Vec<ChunkMeta>>,
}

impl ZoneManifest {
    fn key(entity_id: u64, space_id: u32) -> String {
        format!("{entity_id}:{space_id}")
    }
}

/// Warm storage backend using partitioned files.
pub struct WarmStore {
    dir: PathBuf,
    /// In-memory index: (entity_id, space_id) → sorted timestamps in this tier.
    index: RwLock<BTreeMap<(u64, u32), Vec<i64>>>,
    /// Zone map manifest for chunk-level temporal pruning.
    zone_map: RwLock<ZoneManifest>,
    chunk_size: usize,
}

impl WarmStore {
    /// Open or create a warm store at the given directory.
    pub fn open(dir: &Path) -> Result<Self, StorageError> {
        fs::create_dir_all(dir)?;
        let zone_map = Self::load_zone_manifest(dir)?;
        let store = Self {
            dir: dir.to_path_buf(),
            index: RwLock::new(BTreeMap::new()),
            zone_map: RwLock::new(zone_map),
            chunk_size: DEFAULT_CHUNK_SIZE,
        };
        store.load_index()?;
        Ok(store)
    }

    fn load_zone_manifest(dir: &Path) -> Result<ZoneManifest, StorageError> {
        let manifest_path = dir.join("manifest.json");
        if manifest_path.exists() {
            let content = fs::read_to_string(&manifest_path)?;
            serde_json::from_str(&content).map_err(|_| StorageError::WalCorrupted { offset: 0 })
        } else {
            Ok(ZoneManifest::default())
        }
    }

    fn persist_zone_manifest(&self) -> Result<(), StorageError> {
        let manifest = self.zone_map.read();
        let content = serde_json::to_string_pretty(&*manifest)
            .map_err(|_| StorageError::WalCorrupted { offset: 0 })?;
        let tmp = self.dir.join("manifest.json.tmp");
        fs::write(&tmp, &content)?;
        fs::rename(&tmp, self.dir.join("manifest.json"))?;
        Ok(())
    }

    /// Write a batch of points to warm storage.
    ///
    /// Points are grouped by (entity_id, space_id) and written to chunk files.
    pub fn write_batch(&self, space_id: u32, points: &[TemporalPoint]) -> Result<(), StorageError> {
        // Group by entity_id
        let mut by_entity: BTreeMap<u64, Vec<&TemporalPoint>> = BTreeMap::new();
        for p in points {
            by_entity.entry(p.entity_id()).or_default().push(p);
        }

        let mut index = self.index.write();

        for (entity_id, entity_points) in &by_entity {
            let entity_dir = self.entity_dir(*entity_id);
            fs::create_dir_all(&entity_dir)?;

            // Determine chunk number
            let existing_count = index
                .get(&(*entity_id, space_id))
                .map(|v| v.len())
                .unwrap_or(0);
            let chunk_num = existing_count / self.chunk_size;

            let chunk_path =
                entity_dir.join(format!("space_{space_id:04}_chunk_{chunk_num:06}.warm"));

            // Serialize and append
            let mut data = if chunk_path.exists() {
                fs::read(&chunk_path)?
            } else {
                Vec::new()
            };

            for p in entity_points {
                let encoded = postcard::to_allocvec(p)
                    .map_err(|_| StorageError::WalCorrupted { offset: 0 })?;
                let len = encoded.len() as u32;
                data.extend_from_slice(&len.to_le_bytes());
                data.extend_from_slice(&encoded);

                // Update index
                index
                    .entry((*entity_id, space_id))
                    .or_default()
                    .push(p.timestamp());
            }

            fs::write(&chunk_path, &data)?;

            // Update zone map metadata
            let chunk_filename = format!("space_{space_id:04}_chunk_{chunk_num:06}.warm");
            let timestamps: Vec<i64> = entity_points.iter().map(|p| p.timestamp()).collect();
            let min_ts = timestamps.iter().copied().min().unwrap_or(0);
            let max_ts = timestamps.iter().copied().max().unwrap_or(0);

            let mut zone = self.zone_map.write();
            let zone_key = ZoneManifest::key(*entity_id, space_id);
            let chunk_list = zone.chunks.entry(zone_key).or_default();
            // Update existing chunk meta or add new one
            if let Some(meta) = chunk_list.iter_mut().find(|m| m.filename == chunk_filename) {
                meta.min_timestamp = meta.min_timestamp.min(min_ts);
                meta.max_timestamp = meta.max_timestamp.max(max_ts);
                meta.point_count += entity_points.len() as u32;
            } else {
                chunk_list.push(ChunkMeta {
                    filename: chunk_filename,
                    min_timestamp: min_ts,
                    max_timestamp: max_ts,
                    point_count: entity_points.len() as u32,
                });
            }
        }

        // Sort timestamps in index
        for ts_list in index.values_mut() {
            ts_list.sort_unstable();
            ts_list.dedup();
        }

        // Persist zone manifest
        self.persist_zone_manifest()?;

        Ok(())
    }

    /// Read all points for an entity+space from chunk files.
    fn read_entity_chunks(
        &self,
        entity_id: u64,
        space_id: u32,
    ) -> Result<Vec<TemporalPoint>, StorageError> {
        let entity_dir = self.entity_dir(entity_id);
        if !entity_dir.exists() {
            return Ok(Vec::new());
        }

        let prefix = format!("space_{space_id:04}_chunk_");
        let mut points = Vec::new();

        for entry in fs::read_dir(&entity_dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if !name.starts_with(&prefix) || !name.ends_with(".warm") {
                continue;
            }

            let data = fs::read(entry.path())?;
            let mut offset = 0;
            while offset + 4 <= data.len() {
                let len = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
                offset += 4;
                if offset + len > data.len() {
                    break;
                }
                let point: TemporalPoint = postcard::from_bytes(&data[offset..offset + len])
                    .map_err(|_| StorageError::WalCorrupted { offset: 0 })?;
                points.push(point);
                offset += len;
            }
        }

        points.sort_by_key(|p| p.timestamp());
        Ok(points)
    }

    /// Read entity chunks filtered by zone map — only opens chunks whose
    /// [min_ts, max_ts] overlaps with [start, end]. Falls back to full
    /// read if no zone map data exists.
    fn read_entity_chunks_filtered(
        &self,
        entity_id: u64,
        space_id: u32,
        start: i64,
        end: i64,
    ) -> Result<Vec<TemporalPoint>, StorageError> {
        let entity_dir = self.entity_dir(entity_id);
        if !entity_dir.exists() {
            return Ok(Vec::new());
        }

        let zone = self.zone_map.read();
        let zone_key = ZoneManifest::key(entity_id, space_id);
        let chunk_metas = zone.chunks.get(&zone_key);

        // If we have zone map data, only read overlapping chunks
        if let Some(metas) = chunk_metas {
            let mut points = Vec::new();
            for meta in metas {
                // Zone map pruning: skip if chunk's range doesn't overlap query range
                if meta.max_timestamp < start || meta.min_timestamp > end {
                    continue;
                }

                let chunk_path = entity_dir.join(&meta.filename);
                if !chunk_path.exists() {
                    continue;
                }

                let data = fs::read(&chunk_path)?;
                let mut offset = 0;
                while offset + 4 <= data.len() {
                    let len =
                        u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
                    offset += 4;
                    if offset + len > data.len() {
                        break;
                    }
                    let point: TemporalPoint = postcard::from_bytes(&data[offset..offset + len])
                        .map_err(|_| StorageError::WalCorrupted { offset: 0 })?;
                    points.push(point);
                    offset += len;
                }
            }
            points.sort_by_key(|p| p.timestamp());
            Ok(points)
        } else {
            // No zone map — fall back to full read
            self.read_entity_chunks(entity_id, space_id)
        }
    }

    fn entity_dir(&self, entity_id: u64) -> PathBuf {
        self.dir.join(format!("entity_{entity_id:016}"))
    }

    fn load_index(&self) -> Result<(), StorageError> {
        // Scan existing files to rebuild index
        if !self.dir.exists() {
            return Ok(());
        }

        let mut index = self.index.write();
        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let name = entry.file_name();
            let name = name.to_string_lossy();
            let Some(eid_str) = name.strip_prefix("entity_") else {
                continue;
            };
            let Ok(entity_id) = eid_str.parse::<u64>() else {
                continue;
            };

            for file in fs::read_dir(entry.path())? {
                let file = file?;
                let fname = file.file_name();
                let fname = fname.to_string_lossy();
                if !fname.ends_with(".warm") {
                    continue;
                }
                // Extract space_id from filename: space_XXXX_chunk_YYYYYY.warm
                if let Some(rest) = fname.strip_prefix("space_") {
                    if let Some(space_str) = rest.get(..4) {
                        if let Ok(space_id) = space_str.parse::<u32>() {
                            // Read timestamps from file
                            let data = fs::read(file.path())?;
                            let mut offset = 0;
                            while offset + 4 <= data.len() {
                                let len = u32::from_le_bytes(
                                    data[offset..offset + 4].try_into().unwrap(),
                                ) as usize;
                                offset += 4;
                                if offset + len > data.len() {
                                    break;
                                }
                                if let Ok(point) = postcard::from_bytes::<TemporalPoint>(
                                    &data[offset..offset + len],
                                ) {
                                    index
                                        .entry((entity_id, space_id))
                                        .or_default()
                                        .push(point.timestamp());
                                }
                                offset += len;
                            }
                        }
                    }
                }
            }
        }

        for ts_list in index.values_mut() {
            ts_list.sort_unstable();
            ts_list.dedup();
        }

        Ok(())
    }

    /// Check if a point exists in warm storage.
    pub fn contains(&self, entity_id: u64, space_id: u32, timestamp: i64) -> bool {
        let index = self.index.read();
        index
            .get(&(entity_id, space_id))
            .is_some_and(|ts| ts.binary_search(&timestamp).is_ok())
    }

    /// Number of points tracked in the index.
    pub fn len(&self) -> usize {
        self.index.read().values().map(|v| v.len()).sum()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.index.read().is_empty()
    }
}

impl StorageBackend for WarmStore {
    fn get(
        &self,
        entity_id: u64,
        space_id: u32,
        timestamp: i64,
    ) -> Result<Option<TemporalPoint>, StorageError> {
        if !self.contains(entity_id, space_id, timestamp) {
            return Ok(None);
        }
        let points = self.read_entity_chunks(entity_id, space_id)?;
        Ok(points.into_iter().find(|p| p.timestamp() == timestamp))
    }

    fn put(&self, space_id: u32, point: &TemporalPoint) -> Result<(), StorageError> {
        self.write_batch(space_id, &[point.clone()])
    }

    fn range(
        &self,
        entity_id: u64,
        space_id: u32,
        start: i64,
        end: i64,
    ) -> Result<Vec<TemporalPoint>, StorageError> {
        // Use zone map to skip non-overlapping chunks (RFC-002-08)
        let points = self.read_entity_chunks_filtered(entity_id, space_id, start, end)?;
        Ok(points
            .into_iter()
            .filter(|p| p.timestamp() >= start && p.timestamp() <= end)
            .collect())
    }

    fn delete(&self, entity_id: u64, space_id: u32, timestamp: i64) -> Result<(), StorageError> {
        // Remove from index only (lazy deletion)
        let mut index = self.index.write();
        if let Some(ts_list) = index.get_mut(&(entity_id, space_id)) {
            ts_list.retain(|&t| t != timestamp);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_point(entity_id: u64, timestamp: i64) -> TemporalPoint {
        TemporalPoint::new(entity_id, timestamp, vec![0.1, 0.2, 0.3])
    }

    #[test]
    fn write_and_read_batch() {
        let dir = tempfile::tempdir().unwrap();
        let store = WarmStore::open(dir.path()).unwrap();

        let points: Vec<TemporalPoint> = (0..10).map(|i| sample_point(1, i * 1000)).collect();
        store.write_batch(0, &points).unwrap();

        assert_eq!(store.len(), 10);
        assert!(store.contains(1, 0, 5000));
        assert!(!store.contains(1, 0, 99999));
    }

    #[test]
    fn get_specific_point() {
        let dir = tempfile::tempdir().unwrap();
        let store = WarmStore::open(dir.path()).unwrap();

        let p = sample_point(42, 1000);
        store.put(0, &p).unwrap();

        let result = store.get(42, 0, 1000).unwrap();
        assert_eq!(result, Some(p));
    }

    #[test]
    fn get_nonexistent_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let store = WarmStore::open(dir.path()).unwrap();
        assert_eq!(store.get(1, 0, 1000).unwrap(), None);
    }

    #[test]
    fn range_query() {
        let dir = tempfile::tempdir().unwrap();
        let store = WarmStore::open(dir.path()).unwrap();

        let points: Vec<TemporalPoint> = (0..20).map(|i| sample_point(1, i * 100)).collect();
        store.write_batch(0, &points).unwrap();

        let results = store.range(1, 0, 500, 1500).unwrap();
        assert_eq!(results.len(), 11); // 500, 600, ..., 1500
        assert_eq!(results[0].timestamp(), 500);
        assert_eq!(results.last().unwrap().timestamp(), 1500);
    }

    #[test]
    fn cross_entity_isolation() {
        let dir = tempfile::tempdir().unwrap();
        let store = WarmStore::open(dir.path()).unwrap();

        store.put(0, &sample_point(1, 100)).unwrap();
        store.put(0, &sample_point(2, 100)).unwrap();

        let results = store.range(1, 0, 0, i64::MAX).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entity_id(), 1);
    }

    #[test]
    fn data_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();

        {
            let store = WarmStore::open(dir.path()).unwrap();
            let points: Vec<TemporalPoint> = (0..5).map(|i| sample_point(1, i * 1000)).collect();
            store.write_batch(0, &points).unwrap();
        }

        {
            let store = WarmStore::open(dir.path()).unwrap();
            assert_eq!(store.len(), 5);
            let result = store.get(1, 0, 3000).unwrap();
            assert!(result.is_some());
        }
    }

    #[test]
    fn zone_map_persists_and_prunes() {
        let dir = tempfile::tempdir().unwrap();

        // Write two batches with non-overlapping time ranges
        {
            let store = WarmStore::open(dir.path()).unwrap();
            let early: Vec<TemporalPoint> = (0..10)
                .map(|i| TemporalPoint::new(1, i * 100, vec![0.1; 4]))
                .collect();
            store.write_batch(0, &early).unwrap();

            let late: Vec<TemporalPoint> = (0..10)
                .map(|i| TemporalPoint::new(1, 10_000 + i * 100, vec![0.2; 4]))
                .collect();
            store.write_batch(0, &late).unwrap();
        }

        // Reopen — zone map should survive
        {
            let store = WarmStore::open(dir.path()).unwrap();
            assert_eq!(store.len(), 20);

            // Range query for early timestamps only
            let early_results = store.range(1, 0, 0, 999).unwrap();
            assert_eq!(early_results.len(), 10);

            // Range query for late timestamps only
            let late_results = store.range(1, 0, 10_000, 11_000).unwrap();
            assert_eq!(late_results.len(), 10);

            // Manifest file should exist
            assert!(dir.path().join("manifest.json").exists());
        }
    }

    #[test]
    fn large_batch() {
        let dir = tempfile::tempdir().unwrap();
        let store = WarmStore::open(dir.path()).unwrap();

        let points: Vec<TemporalPoint> = (0..1000)
            .map(|i| TemporalPoint::new(i / 100, (i % 100) as i64 * 1000, vec![i as f32; 8]))
            .collect();
        store.write_batch(0, &points).unwrap();

        assert_eq!(store.len(), 1000);

        // Retrieve specific entity
        let results = store.range(5, 0, 0, 100_000).unwrap();
        assert_eq!(results.len(), 100); // entity 5 has 100 points
    }
}
