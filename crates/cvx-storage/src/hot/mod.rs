//! RocksDB-backed persistent storage (hot tier).
//!
//! Uses separate column families for different data types, each with
//! optimized compression and bloom filter settings.
//!
//! # Column Families
//!
//! | CF | Key | Value | Compression |
//! |---|---|---|---|
//! | `vectors` | entity+space+ts (20B) | raw f32 bytes | None |
//! | `timelines` | entity+space (12B) | EntityTimeline (postcard) | LZ4 |
//! | `system` | string | config/state | None |

use std::path::Path;

use cvx_core::StorageBackend;
use cvx_core::error::StorageError;
use cvx_core::types::{EntityTimeline, TemporalPoint};
use rocksdb::{
    ColumnFamilyDescriptor, DBWithThreadMode, IteratorMode, Options, SingleThreaded, SliceTransform,
};

use crate::keys;

const CF_VECTORS: &str = "vectors";
const CF_TIMELINES: &str = "timelines";
const CF_SYSTEM: &str = "default";

/// RocksDB-backed persistent storage for the hot tier.
///
/// Implements [`StorageBackend`] with data surviving process restarts.
///
/// # Example
///
/// ```no_run
/// use cvx_core::{StorageBackend, TemporalPoint};
/// use cvx_storage::hot::HotStore;
///
/// let store = HotStore::open("/tmp/cvx-test").unwrap();
/// let point = TemporalPoint::new(42, 1000, vec![0.1, 0.2, 0.3]);
/// store.put(0, &point).unwrap();
///
/// let retrieved = store.get(42, 0, 1000).unwrap();
/// assert_eq!(retrieved, Some(point));
/// ```
pub struct HotStore {
    db: DBWithThreadMode<SingleThreaded>,
}

impl HotStore {
    /// Open or create a HotStore at the given path.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, StorageError> {
        let mut db_opts = Options::default();
        db_opts.create_if_missing(true);
        db_opts.create_missing_column_families(true);

        let cf_descriptors = vec![
            ColumnFamilyDescriptor::new(CF_SYSTEM, Options::default()),
            ColumnFamilyDescriptor::new(CF_VECTORS, Self::vectors_cf_options()),
            ColumnFamilyDescriptor::new(CF_TIMELINES, Self::timelines_cf_options()),
        ];

        let db =
            DBWithThreadMode::<SingleThreaded>::open_cf_descriptors(&db_opts, path, cf_descriptors)
                .map_err(|e| StorageError::Io(std::io::Error::other(e.to_string())))?;

        Ok(Self { db })
    }

    /// Configure the vectors column family.
    fn vectors_cf_options() -> Options {
        let mut opts = Options::default();
        // Prefix bloom filter on entity_id + space_id (12 bytes)
        opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(keys::PREFIX_SIZE));
        // No compression: f32 vectors are already dense, don't compress well
        opts.set_compression_type(rocksdb::DBCompressionType::None);
        opts
    }

    /// Configure the timelines column family.
    fn timelines_cf_options() -> Options {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts
    }

    /// Get the CF handle for vectors.
    fn vectors_cf(&self) -> &rocksdb::ColumnFamily {
        self.db.cf_handle(CF_VECTORS).expect("vectors CF missing")
    }

    /// Get the CF handle for timelines.
    fn timelines_cf(&self) -> &rocksdb::ColumnFamily {
        self.db
            .cf_handle(CF_TIMELINES)
            .expect("timelines CF missing")
    }

    /// Serialize a vector to raw bytes (native f32 layout).
    fn serialize_vector(vector: &[f32]) -> Vec<u8> {
        vector.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    /// Deserialize raw bytes back to a vector.
    fn deserialize_vector(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    /// Update the timeline metadata for an entity after a put.
    fn update_timeline(
        &self,
        entity_id: u64,
        space_id: u32,
        timestamp: i64,
    ) -> Result<(), StorageError> {
        let tl_key = keys::encode_prefix(entity_id, space_id);

        let timeline = match self
            .db
            .get_cf(self.timelines_cf(), tl_key)
            .map_err(|e| StorageError::Io(std::io::Error::other(e.to_string())))?
        {
            Some(bytes) => {
                let existing: EntityTimeline = postcard::from_bytes(&bytes)
                    .map_err(|e| StorageError::Io(std::io::Error::other(e.to_string())))?;
                EntityTimeline::new(
                    entity_id,
                    space_id,
                    existing.first_seen().min(timestamp),
                    existing.last_seen().max(timestamp),
                    existing.point_count() + 1,
                    existing.keyframe_interval(),
                )
            }
            None => EntityTimeline::new(entity_id, space_id, timestamp, timestamp, 1, 10),
        };

        let tl_bytes = postcard::to_allocvec(&timeline)
            .map_err(|e| StorageError::Io(std::io::Error::other(e.to_string())))?;
        self.db
            .put_cf(self.timelines_cf(), tl_key, tl_bytes)
            .map_err(|e| StorageError::Io(std::io::Error::other(e.to_string())))?;

        Ok(())
    }

    /// Get the timeline metadata for an entity in a space.
    pub fn get_timeline(
        &self,
        entity_id: u64,
        space_id: u32,
    ) -> Result<Option<EntityTimeline>, StorageError> {
        let tl_key = keys::encode_prefix(entity_id, space_id);
        match self
            .db
            .get_cf(self.timelines_cf(), tl_key)
            .map_err(|e| StorageError::Io(std::io::Error::other(e.to_string())))?
        {
            Some(bytes) => {
                let tl: EntityTimeline = postcard::from_bytes(&bytes)
                    .map_err(|e| StorageError::Io(std::io::Error::other(e.to_string())))?;
                Ok(Some(tl))
            }
            None => Ok(None),
        }
    }
}

impl StorageBackend for HotStore {
    fn get(
        &self,
        entity_id: u64,
        space_id: u32,
        timestamp: i64,
    ) -> Result<Option<TemporalPoint>, StorageError> {
        let key = keys::encode_key(entity_id, space_id, timestamp);
        match self
            .db
            .get_cf(self.vectors_cf(), key)
            .map_err(|e| StorageError::Io(std::io::Error::other(e.to_string())))?
        {
            Some(bytes) => {
                let vector = Self::deserialize_vector(&bytes);
                Ok(Some(TemporalPoint::new(entity_id, timestamp, vector)))
            }
            None => Ok(None),
        }
    }

    fn put(&self, space_id: u32, point: &TemporalPoint) -> Result<(), StorageError> {
        let key = keys::encode_key(point.entity_id(), space_id, point.timestamp());
        let value = Self::serialize_vector(point.vector());

        self.db
            .put_cf(self.vectors_cf(), key, value)
            .map_err(|e| StorageError::Io(std::io::Error::other(e.to_string())))?;

        self.update_timeline(point.entity_id(), space_id, point.timestamp())?;

        Ok(())
    }

    fn range(
        &self,
        entity_id: u64,
        space_id: u32,
        start: i64,
        end: i64,
    ) -> Result<Vec<TemporalPoint>, StorageError> {
        let start_key = keys::encode_key(entity_id, space_id, start);
        let end_key = keys::encode_key(entity_id, space_id, end);
        let prefix = keys::encode_prefix(entity_id, space_id);

        let iter = self.db.iterator_cf(
            self.vectors_cf(),
            IteratorMode::From(&start_key, rocksdb::Direction::Forward),
        );

        let mut results = Vec::new();
        for item in iter {
            let (key, value) =
                item.map_err(|e| StorageError::Io(std::io::Error::other(e.to_string())))?;

            // Stop if we've passed the prefix (different entity/space)
            if key.len() < keys::PREFIX_SIZE || key[..keys::PREFIX_SIZE] != prefix[..] {
                break;
            }
            // Stop if we've passed the end key
            if key[..] > end_key[..] {
                break;
            }

            let (_, _, timestamp) = keys::decode_key(&key);
            let vector = Self::deserialize_vector(&value);
            results.push(TemporalPoint::new(entity_id, timestamp, vector));
        }

        Ok(results)
    }

    fn delete(&self, entity_id: u64, space_id: u32, timestamp: i64) -> Result<(), StorageError> {
        let key = keys::encode_key(entity_id, space_id, timestamp);
        self.db
            .delete_cf(self.vectors_cf(), key)
            .map_err(|e| StorageError::Io(std::io::Error::other(e.to_string())))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store() -> (tempfile::TempDir, HotStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = HotStore::open(dir.path()).unwrap();
        (dir, store)
    }

    fn sample_point(entity_id: u64, timestamp: i64) -> TemporalPoint {
        TemporalPoint::new(entity_id, timestamp, vec![0.1, 0.2, 0.3])
    }

    #[test]
    fn put_and_get() {
        let (_dir, store) = temp_store();
        let p = sample_point(42, 1000);
        store.put(0, &p).unwrap();

        let result = store.get(42, 0, 1000).unwrap();
        assert_eq!(result, Some(p));
    }

    #[test]
    fn get_nonexistent_returns_none() {
        let (_dir, store) = temp_store();
        assert_eq!(store.get(999, 0, 0).unwrap(), None);
    }

    #[test]
    fn range_returns_ordered() {
        let (_dir, store) = temp_store();
        for ts in [100, 200, 300, 400, 500] {
            store.put(0, &sample_point(1, ts)).unwrap();
        }

        let results = store.range(1, 0, 200, 400).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].timestamp(), 200);
        assert_eq!(results[1].timestamp(), 300);
        assert_eq!(results[2].timestamp(), 400);
    }

    #[test]
    fn range_does_not_cross_entities() {
        let (_dir, store) = temp_store();
        store.put(0, &sample_point(1, 100)).unwrap();
        store.put(0, &sample_point(2, 100)).unwrap();

        let results = store.range(1, 0, 0, i64::MAX).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entity_id(), 1);
    }

    #[test]
    fn delete_removes_point() {
        let (_dir, store) = temp_store();
        store.put(0, &sample_point(1, 1000)).unwrap();
        store.delete(1, 0, 1000).unwrap();
        assert_eq!(store.get(1, 0, 1000).unwrap(), None);
    }

    #[test]
    fn timeline_tracks_metadata() {
        let (_dir, store) = temp_store();
        store.put(0, &sample_point(42, 1000)).unwrap();
        store.put(0, &sample_point(42, 2000)).unwrap();
        store.put(0, &sample_point(42, 3000)).unwrap();

        let tl = store.get_timeline(42, 0).unwrap().unwrap();
        assert_eq!(tl.entity_id(), 42);
        assert_eq!(tl.first_seen(), 1000);
        assert_eq!(tl.last_seen(), 3000);
        assert_eq!(tl.point_count(), 3);
    }

    #[test]
    fn negative_timestamps_work() {
        let (_dir, store) = temp_store();
        store.put(0, &sample_point(1, -5000)).unwrap();
        store.put(0, &sample_point(1, -1000)).unwrap();
        store.put(0, &sample_point(1, 0)).unwrap();

        let results = store.range(1, 0, -3000, 0).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].timestamp(), -1000);
        assert_eq!(results[1].timestamp(), 0);
    }

    #[test]
    fn data_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();

        // Write data
        {
            let store = HotStore::open(dir.path()).unwrap();
            store.put(0, &sample_point(42, 1000)).unwrap();
            store.put(0, &sample_point(42, 2000)).unwrap();
        }
        // store dropped → DB closed

        // Reopen and verify
        {
            let store = HotStore::open(dir.path()).unwrap();
            let p1 = store.get(42, 0, 1000).unwrap();
            assert!(p1.is_some());
            assert_eq!(p1.unwrap().timestamp(), 1000);

            let p2 = store.get(42, 0, 2000).unwrap();
            assert!(p2.is_some());

            let tl = store.get_timeline(42, 0).unwrap().unwrap();
            assert_eq!(tl.point_count(), 2);
        }
    }

    #[test]
    fn d768_vectors_roundtrip() {
        let (_dir, store) = temp_store();
        let vector: Vec<f32> = (0..768).map(|i| i as f32 * 0.001).collect();
        let p = TemporalPoint::new(1, 1000, vector.clone());
        store.put(0, &p).unwrap();

        let retrieved = store.get(1, 0, 1000).unwrap().unwrap();
        assert_eq!(retrieved.vector(), vector.as_slice());
    }

    #[test]
    fn insert_100k_and_retrieve() {
        let (_dir, store) = temp_store();
        let dim = 8; // small dim for speed
        for i in 0..100_000u64 {
            let entity = i / 100;
            let ts = (i % 100) as i64 * 1000;
            let vec = vec![i as f32; dim];
            store.put(0, &TemporalPoint::new(entity, ts, vec)).unwrap();
        }

        // Retrieve entity 42's full trajectory
        let results = store.range(42, 0, 0, 100_000).unwrap();
        assert_eq!(results.len(), 100);

        // Verify ordering
        for window in results.windows(2) {
            assert!(window[0].timestamp() < window[1].timestamp());
        }

        // Verify timeline
        let tl = store.get_timeline(42, 0).unwrap().unwrap();
        assert_eq!(tl.point_count(), 100);
        assert_eq!(tl.first_seen(), 0);
        assert_eq!(tl.last_seen(), 99_000);
    }
}
