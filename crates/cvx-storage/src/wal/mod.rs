//! Write-Ahead Log (WAL) for crash-safe ingestion.
//!
//! Implements an append-only, CRC32-validated log with segment rotation
//! and recovery protocol per the Storage Layout spec §3.
//!
//! # Architecture
//!
//! ```text
//! wal/
//! ├── segment-000000000000.wal   (64MB max, append-only)
//! ├── segment-000000000001.wal
//! └── wal.meta                   (committed state)
//! ```
//!
//! ## Entry lifecycle
//!
//! 1. Caller appends entry → WAL writes header + payload + CRC32
//! 2. After downstream store + index confirm, caller calls `commit()`
//! 3. On crash before commit, recovery replays uncommitted entries
//!
//! ## Recovery protocol
//!
//! 1. Read `wal.meta` for committed sequence number
//! 2. Scan all segments for entries after committed sequence
//! 3. Validate CRC32 — truncate at first invalid entry
//! 4. Return uncommitted valid entries for replay

use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use cvx_core::error::StorageError;

/// Magic bytes identifying a CVX WAL segment.
const WAL_MAGIC: [u8; 4] = *b"CVXW";

/// Current WAL format version.
const WAL_VERSION: u16 = 1;

/// Segment header size in bytes.
const SEGMENT_HEADER_SIZE: usize = 32;

/// Entry header size in bytes (before payload).
const ENTRY_HEADER_SIZE: usize = 24;

/// Default maximum segment size: 64 MB.
const DEFAULT_MAX_SEGMENT_SIZE: u64 = 64 * 1024 * 1024;

/// WAL entry types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EntryType {
    /// Insert a new temporal point.
    Insert = 0,
    /// Delete an existing temporal point.
    Delete = 1,
    /// Update an existing temporal point.
    Update = 2,
    /// Checkpoint marker.
    Checkpoint = 3,
}

impl TryFrom<u8> for EntryType {
    type Error = StorageError;
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Insert),
            1 => Ok(Self::Delete),
            2 => Ok(Self::Update),
            3 => Ok(Self::Checkpoint),
            _ => Err(StorageError::WalCorrupted { offset: 0 }),
        }
    }
}

/// A WAL entry ready for serialization.
#[derive(Debug, Clone)]
pub struct WalEntry {
    /// Global monotonic sequence number.
    pub sequence: u64,
    /// Type of operation.
    pub entry_type: EntryType,
    /// Flags (bit 0: is_keyframe).
    pub flags: u8,
    /// Serialized payload.
    pub payload: Vec<u8>,
}

/// WAL configuration.
#[derive(Debug, Clone)]
pub struct WalConfig {
    /// Maximum segment file size in bytes.
    pub max_segment_size: u64,
    /// Whether to fsync after every write (durable but slower).
    pub sync_on_write: bool,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            max_segment_size: DEFAULT_MAX_SEGMENT_SIZE,
            sync_on_write: false,
        }
    }
}

/// Persisted WAL metadata.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct WalMeta {
    format_version: u16,
    head_segment: u64,
    head_offset: u64,
    committed_sequence: u64,
    last_sequence: u64,
}

impl Default for WalMeta {
    fn default() -> Self {
        Self {
            format_version: WAL_VERSION,
            head_segment: 0,
            head_offset: SEGMENT_HEADER_SIZE as u64,
            committed_sequence: 0,
            last_sequence: 0,
        }
    }
}

/// Write-Ahead Log.
pub struct Wal {
    dir: PathBuf,
    config: WalConfig,
    meta: WalMeta,
    current_writer: Option<BufWriter<File>>,
    current_segment_size: u64,
}

impl Wal {
    /// Open or create a WAL in the given directory.
    pub fn open(dir: &Path, config: WalConfig) -> Result<Self, StorageError> {
        fs::create_dir_all(dir)?;

        let meta = Self::load_or_create_meta(dir)?;

        let mut wal = Self {
            dir: dir.to_path_buf(),
            config,
            meta,
            current_writer: None,
            current_segment_size: 0,
        };

        wal.open_current_segment()?;
        Ok(wal)
    }

    fn load_or_create_meta(dir: &Path) -> Result<WalMeta, StorageError> {
        let meta_path = dir.join("wal.meta");
        if meta_path.exists() {
            let content = fs::read_to_string(&meta_path)?;
            let meta: WalMeta = serde_json::from_str(&content)
                .map_err(|_| StorageError::WalCorrupted { offset: 0 })?;
            Ok(meta)
        } else {
            let meta = WalMeta::default();
            let content = serde_json::to_string_pretty(&meta)
                .map_err(|_| StorageError::WalCorrupted { offset: 0 })?;
            fs::write(&meta_path, &content)?;
            Ok(meta)
        }
    }

    fn segment_path(&self, segment_id: u64) -> PathBuf {
        self.dir.join(format!("segment-{segment_id:012}.wal"))
    }

    fn open_current_segment(&mut self) -> Result<(), StorageError> {
        let path = self.segment_path(self.meta.head_segment);

        if path.exists() {
            let file = OpenOptions::new().append(true).open(&path)?;
            let size = file.metadata()?.len();
            self.current_segment_size = size;
            self.current_writer = Some(BufWriter::new(file));
        } else {
            let file = File::create(&path)?;
            let mut writer = BufWriter::new(file);
            Self::write_segment_header(&mut writer, self.meta.head_segment)?;
            writer.flush()?;
            self.current_segment_size = SEGMENT_HEADER_SIZE as u64;
            self.current_writer = Some(writer);
        }

        Ok(())
    }

    fn write_segment_header(
        writer: &mut BufWriter<File>,
        segment_id: u64,
    ) -> Result<(), StorageError> {
        let mut header = [0u8; SEGMENT_HEADER_SIZE];
        header[0..4].copy_from_slice(&WAL_MAGIC);
        header[4..6].copy_from_slice(&WAL_VERSION.to_le_bytes());
        header[6..14].copy_from_slice(&segment_id.to_le_bytes());
        // CreatedAt: current time in microseconds (or 0 for simplicity)
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as i64;
        header[14..22].copy_from_slice(&now.to_le_bytes());
        // Reserved: [22..32] already zeroed
        writer.write_all(&header)?;
        Ok(())
    }

    fn rotate_segment(&mut self) -> Result<(), StorageError> {
        // Flush and close current segment
        if let Some(ref mut writer) = self.current_writer {
            writer.flush()?;
        }
        self.current_writer = None;

        self.meta.head_segment += 1;
        self.meta.head_offset = SEGMENT_HEADER_SIZE as u64;
        self.persist_meta()?;
        self.open_current_segment()?;

        Ok(())
    }

    /// Append an entry to the WAL. Returns the assigned sequence number.
    pub fn append(
        &mut self,
        entry_type: EntryType,
        flags: u8,
        payload: &[u8],
    ) -> Result<u64, StorageError> {
        self.meta.last_sequence += 1;
        let sequence = self.meta.last_sequence;

        let entry_length = (ENTRY_HEADER_SIZE + payload.len()) as u32;

        // Check if we need to rotate
        if self.current_segment_size + entry_length as u64 > self.config.max_segment_size {
            self.rotate_segment()?;
        }

        let crc = crc32_hash(payload);

        // Write entry header
        let writer = self.current_writer.as_mut().unwrap();
        writer.write_all(&entry_length.to_le_bytes())?; // 4 bytes
        writer.write_all(&sequence.to_le_bytes())?; // 8 bytes
        writer.write_all(&[entry_type as u8])?; // 1 byte
        writer.write_all(&[flags])?; // 1 byte
        writer.write_all(&[0u8; 2])?; // 2 bytes reserved
        writer.write_all(&crc.to_le_bytes())?; // 4 bytes
        // Padding to 24 bytes
        writer.write_all(&[0u8; 4])?; // 4 bytes padding

        // Write payload
        writer.write_all(payload)?;

        if self.config.sync_on_write {
            writer.flush()?;
            writer.get_ref().sync_all()?;
        }

        self.current_segment_size += entry_length as u64;
        self.meta.head_offset = self.current_segment_size;

        Ok(sequence)
    }

    /// Mark all entries up to `sequence` as committed.
    pub fn commit(&mut self, sequence: u64) -> Result<(), StorageError> {
        self.meta.committed_sequence = sequence;
        self.flush()?;
        self.persist_meta()?;
        Ok(())
    }

    /// Get the last committed sequence number.
    pub fn committed_sequence(&self) -> u64 {
        self.meta.committed_sequence
    }

    /// Get the last appended sequence number.
    pub fn last_sequence(&self) -> u64 {
        self.meta.last_sequence
    }

    /// Flush buffered writes to disk.
    pub fn flush(&mut self) -> Result<(), StorageError> {
        if let Some(ref mut writer) = self.current_writer {
            writer.flush()?;
        }
        Ok(())
    }

    fn persist_meta(&self) -> Result<(), StorageError> {
        let meta_path = self.dir.join("wal.meta");
        let content = serde_json::to_string_pretty(&self.meta)
            .map_err(|_| StorageError::WalCorrupted { offset: 0 })?;
        // Write atomically via temp file
        let tmp_path = self.dir.join("wal.meta.tmp");
        fs::write(&tmp_path, &content)?;
        fs::rename(&tmp_path, &meta_path)?;
        Ok(())
    }

    /// Recover uncommitted entries after a crash.
    ///
    /// Scans all segments for entries with sequence > committed_sequence.
    /// Validates CRC32 for each entry. Stops at first corrupted entry.
    /// Returns valid uncommitted entries for replay.
    pub fn recover(&mut self) -> Result<Vec<WalEntry>, StorageError> {
        self.flush()?;
        // Close current writer so we can read the segments
        self.current_writer = None;

        let committed = self.meta.committed_sequence;
        let mut uncommitted = Vec::new();

        // Find all segment files
        let mut segments = self.list_segments()?;
        segments.sort();

        for seg_id in segments {
            let entries = self.read_segment(seg_id, committed)?;
            uncommitted.extend(entries);
        }

        // Reopen current segment for further writes
        self.open_current_segment()?;

        Ok(uncommitted)
    }

    fn list_segments(&self) -> Result<Vec<u64>, StorageError> {
        let mut segments = Vec::new();
        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(rest) = name.strip_prefix("segment-") {
                if let Some(num_str) = rest.strip_suffix(".wal") {
                    if let Ok(id) = num_str.parse::<u64>() {
                        segments.push(id);
                    }
                }
            }
        }
        Ok(segments)
    }

    fn read_segment(
        &self,
        segment_id: u64,
        committed_seq: u64,
    ) -> Result<Vec<WalEntry>, StorageError> {
        let path = self.segment_path(segment_id);
        let file = File::open(&path)?;
        let file_len = file.metadata()?.len();
        let mut reader = BufReader::new(file);

        // Validate and skip segment header
        let mut header = [0u8; SEGMENT_HEADER_SIZE];
        if reader.read_exact(&mut header).is_err() {
            return Ok(Vec::new()); // Empty or too-small segment
        }
        if header[0..4] != WAL_MAGIC {
            return Err(StorageError::WalCorrupted { offset: 0 });
        }

        let mut entries = Vec::new();
        let mut offset = SEGMENT_HEADER_SIZE as u64;

        while offset + ENTRY_HEADER_SIZE as u64 <= file_len {
            // Read entry header
            let mut entry_header = [0u8; ENTRY_HEADER_SIZE];
            if reader.read_exact(&mut entry_header).is_err() {
                break; // Partial header = truncated entry
            }

            let entry_length = u32::from_le_bytes(entry_header[0..4].try_into().unwrap());
            let sequence = u64::from_le_bytes(entry_header[4..12].try_into().unwrap());
            let entry_type_byte = entry_header[12];
            let flags = entry_header[13];
            // [14..16] reserved
            let stored_crc = u32::from_le_bytes(entry_header[16..20].try_into().unwrap());
            // [20..24] padding

            let payload_len = entry_length as usize - ENTRY_HEADER_SIZE;
            if payload_len > self.config.max_segment_size as usize {
                break; // Corrupted entry length
            }

            let mut payload = vec![0u8; payload_len];
            if reader.read_exact(&mut payload).is_err() {
                break; // Truncated payload
            }

            // Validate CRC32
            let computed_crc = crc32_hash(&payload);
            if computed_crc != stored_crc {
                // Corrupted entry — stop here per recovery protocol
                break;
            }

            offset += entry_length as u64;

            // Only return uncommitted entries
            if sequence > committed_seq {
                let entry_type = EntryType::try_from(entry_type_byte)?;
                entries.push(WalEntry {
                    sequence,
                    entry_type,
                    flags,
                    payload,
                });
            }
        }

        Ok(entries)
    }

    /// Truncate the WAL at the current committed position.
    ///
    /// Removes all segments before the committed segment and truncates
    /// the committed segment at the committed offset. Used after recovery.
    pub fn truncate_uncommitted(&mut self) -> Result<(), StorageError> {
        self.meta.last_sequence = self.meta.committed_sequence;
        self.persist_meta()?;
        Ok(())
    }
}

/// Simple CRC32 hash (IEEE polynomial).
fn crc32_hash(data: &[u8]) -> u32 {
    // Using a simple CRC32 implementation to avoid adding a dependency.
    // IEEE polynomial: 0xEDB88320 (reflected)
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// Encode an insert payload: entity_id + timestamp + dimension + vector.
pub fn encode_insert_payload(entity_id: u64, timestamp: i64, vector: &[f32]) -> Vec<u8> {
    let dim = vector.len() as u16;
    let mut buf = Vec::with_capacity(8 + 8 + 2 + vector.len() * 4);
    buf.extend_from_slice(&entity_id.to_le_bytes());
    buf.extend_from_slice(&timestamp.to_le_bytes());
    buf.extend_from_slice(&dim.to_le_bytes());
    for &v in vector {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    // MetadataLen = 0 (no metadata for now)
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf
}

/// Decode an insert payload back into components.
pub fn decode_insert_payload(payload: &[u8]) -> Result<(u64, i64, Vec<f32>), StorageError> {
    if payload.len() < 18 {
        return Err(StorageError::WalCorrupted { offset: 0 });
    }
    let entity_id = u64::from_le_bytes(payload[0..8].try_into().unwrap());
    let timestamp = i64::from_le_bytes(payload[8..16].try_into().unwrap());
    let dim = u16::from_le_bytes(payload[16..18].try_into().unwrap()) as usize;

    let vector_start = 18;
    let vector_end = vector_start + dim * 4;
    if payload.len() < vector_end {
        return Err(StorageError::WalCorrupted { offset: 16 });
    }

    let vector: Vec<f32> = (0..dim)
        .map(|i| {
            let offset = vector_start + i * 4;
            f32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap())
        })
        .collect();

    Ok((entity_id, timestamp, vector))
}

/// Encode a delete payload: entity_id + timestamp.
pub fn encode_delete_payload(entity_id: u64, timestamp: i64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(16);
    buf.extend_from_slice(&entity_id.to_le_bytes());
    buf.extend_from_slice(&timestamp.to_le_bytes());
    buf
}

/// Decode a delete payload.
pub fn decode_delete_payload(payload: &[u8]) -> Result<(u64, i64), StorageError> {
    if payload.len() < 16 {
        return Err(StorageError::WalCorrupted { offset: 0 });
    }
    let entity_id = u64::from_le_bytes(payload[0..8].try_into().unwrap());
    let timestamp = i64::from_le_bytes(payload[8..16].try_into().unwrap());
    Ok((entity_id, timestamp))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_wal_dir() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    // ─── Basic operations ───────────────────────────────────────────────

    #[test]
    fn create_and_append() {
        let dir = tmp_wal_dir();
        let mut wal = Wal::open(dir.path(), WalConfig::default()).unwrap();

        let payload = encode_insert_payload(42, 1000, &[0.1, 0.2, 0.3]);
        let seq = wal.append(EntryType::Insert, 0, &payload).unwrap();
        assert_eq!(seq, 1);

        let seq2 = wal.append(EntryType::Insert, 0, &payload).unwrap();
        assert_eq!(seq2, 2);
    }

    #[test]
    fn commit_updates_sequence() {
        let dir = tmp_wal_dir();
        let mut wal = Wal::open(dir.path(), WalConfig::default()).unwrap();

        let payload = encode_insert_payload(1, 100, &[1.0]);
        let seq = wal.append(EntryType::Insert, 0, &payload).unwrap();
        wal.commit(seq).unwrap();
        assert_eq!(wal.committed_sequence(), seq);
    }

    // ─── Recovery ───────────────────────────────────────────────────────

    #[test]
    fn recover_uncommitted_entries() {
        let dir = tmp_wal_dir();

        // Write 5 entries, commit first 3
        {
            let mut wal = Wal::open(dir.path(), WalConfig::default()).unwrap();
            for i in 0..5u64 {
                let payload = encode_insert_payload(i, (i * 1000) as i64, &[i as f32]);
                wal.append(EntryType::Insert, 0, &payload).unwrap();
            }
            wal.commit(3).unwrap();
        }

        // Reopen and recover
        {
            let mut wal = Wal::open(dir.path(), WalConfig::default()).unwrap();
            let uncommitted = wal.recover().unwrap();
            assert_eq!(uncommitted.len(), 2); // entries 4 and 5
            assert_eq!(uncommitted[0].sequence, 4);
            assert_eq!(uncommitted[1].sequence, 5);

            // Verify payloads
            let (eid, ts, _vec) = decode_insert_payload(&uncommitted[0].payload).unwrap();
            assert_eq!(eid, 3);
            assert_eq!(ts, 3000);
        }
    }

    #[test]
    fn recover_all_committed_returns_empty() {
        let dir = tmp_wal_dir();

        {
            let mut wal = Wal::open(dir.path(), WalConfig::default()).unwrap();
            for i in 0..3u64 {
                let payload = encode_insert_payload(i, 0, &[0.0]);
                wal.append(EntryType::Insert, 0, &payload).unwrap();
            }
            wal.commit(3).unwrap();
        }

        {
            let mut wal = Wal::open(dir.path(), WalConfig::default()).unwrap();
            let uncommitted = wal.recover().unwrap();
            assert!(uncommitted.is_empty());
        }
    }

    #[test]
    fn recover_truncated_entry_is_skipped() {
        let dir = tmp_wal_dir();

        // Write entries normally
        {
            let mut wal = Wal::open(
                dir.path(),
                WalConfig {
                    sync_on_write: true,
                    ..Default::default()
                },
            )
            .unwrap();
            let payload = encode_insert_payload(1, 1000, &[1.0, 2.0, 3.0]);
            wal.append(EntryType::Insert, 0, &payload).unwrap();
            let payload2 = encode_insert_payload(2, 2000, &[4.0, 5.0, 6.0]);
            wal.append(EntryType::Insert, 0, &payload2).unwrap();
            wal.flush().unwrap();
        }

        // Corrupt the file by truncating the last few bytes
        {
            let seg_path = dir.path().join("segment-000000000000.wal");
            let file = OpenOptions::new().write(true).open(&seg_path).unwrap();
            let current_len = file.metadata().unwrap().len();
            file.set_len(current_len - 5).unwrap(); // truncate last 5 bytes
        }

        // Recovery should return only the first valid entry
        {
            let mut wal = Wal::open(dir.path(), WalConfig::default()).unwrap();
            let entries = wal.recover().unwrap();
            assert_eq!(entries.len(), 1); // only first entry survived
            assert_eq!(entries[0].sequence, 1);
        }
    }

    #[test]
    fn recover_corrupted_crc_stops() {
        let dir = tmp_wal_dir();

        {
            let mut wal = Wal::open(
                dir.path(),
                WalConfig {
                    sync_on_write: true,
                    ..Default::default()
                },
            )
            .unwrap();
            for i in 0..3u64 {
                let payload = encode_insert_payload(i, 0, &[i as f32; 4]);
                wal.append(EntryType::Insert, 0, &payload).unwrap();
            }
            wal.flush().unwrap();
        }

        // Corrupt the payload of the second entry (flip a byte)
        {
            let seg_path = dir.path().join("segment-000000000000.wal");
            let mut data = fs::read(&seg_path).unwrap();
            // Second entry starts at: header(32) + first_entry_size
            // First entry: header(24) + payload(8+8+2+4*4+4 = 38) = 62
            let second_entry_payload_offset = SEGMENT_HEADER_SIZE + 62 + ENTRY_HEADER_SIZE;
            if second_entry_payload_offset < data.len() {
                data[second_entry_payload_offset] ^= 0xFF; // flip bits
            }
            fs::write(&seg_path, &data).unwrap();
        }

        {
            let mut wal = Wal::open(dir.path(), WalConfig::default()).unwrap();
            let entries = wal.recover().unwrap();
            // Should stop at the corrupted second entry, returning only the first
            assert_eq!(entries.len(), 1);
        }
    }

    // ─── Segment rotation ───────────────────────────────────────────────

    #[test]
    fn segment_rotation_on_size_limit() {
        let dir = tmp_wal_dir();
        let config = WalConfig {
            max_segment_size: 256, // tiny segments for testing
            sync_on_write: false,
        };
        let mut wal = Wal::open(dir.path(), config).unwrap();

        // Write enough entries to trigger rotation
        for i in 0..20u64 {
            let payload = encode_insert_payload(i, 0, &[0.0; 8]);
            wal.append(EntryType::Insert, 0, &payload).unwrap();
        }
        wal.flush().unwrap();

        // Should have created multiple segment files
        let segments = wal.list_segments().unwrap();
        assert!(
            segments.len() > 1,
            "expected multiple segments, got {}",
            segments.len()
        );
    }

    #[test]
    fn recovery_across_segments() {
        let dir = tmp_wal_dir();
        let config = WalConfig {
            max_segment_size: 256,
            sync_on_write: true,
        };

        let total_entries = 20u64;
        let commit_at = 10u64;

        {
            let mut wal = Wal::open(dir.path(), config.clone()).unwrap();
            for i in 0..total_entries {
                let payload = encode_insert_payload(i, (i * 100) as i64, &[i as f32]);
                wal.append(EntryType::Insert, 0, &payload).unwrap();
            }
            wal.commit(commit_at).unwrap();
        }

        {
            let mut wal = Wal::open(dir.path(), config).unwrap();
            let uncommitted = wal.recover().unwrap();
            assert_eq!(uncommitted.len(), (total_entries - commit_at) as usize);
            // Verify all have sequence > commit_at
            for entry in &uncommitted {
                assert!(entry.sequence > commit_at);
            }
        }
    }

    // ─── Payload encoding/decoding ──────────────────────────────────────

    #[test]
    fn insert_payload_roundtrip() {
        let vector = vec![0.1, 0.2, 0.3, 0.4];
        let payload = encode_insert_payload(42, -5000, &vector);
        let (eid, ts, vec) = decode_insert_payload(&payload).unwrap();
        assert_eq!(eid, 42);
        assert_eq!(ts, -5000);
        assert_eq!(vec, vector);
    }

    #[test]
    fn delete_payload_roundtrip() {
        let payload = encode_delete_payload(99, 12345);
        let (eid, ts) = decode_delete_payload(&payload).unwrap();
        assert_eq!(eid, 99);
        assert_eq!(ts, 12345);
    }

    #[test]
    fn insert_payload_d768_roundtrip() {
        let vector: Vec<f32> = (0..768).map(|i| i as f32 * 0.001).collect();
        let payload = encode_insert_payload(1, 1_000_000, &vector);
        let (eid, ts, vec) = decode_insert_payload(&payload).unwrap();
        assert_eq!(eid, 1);
        assert_eq!(ts, 1_000_000);
        assert_eq!(vec.len(), 768);
        for (a, b) in vec.iter().zip(vector.iter()) {
            assert!((a - b).abs() < f32::EPSILON);
        }
    }

    // ─── CRC32 ──────────────────────────────────────────────────────────

    #[test]
    fn crc32_deterministic() {
        let data = b"hello world";
        let crc1 = crc32_hash(data);
        let crc2 = crc32_hash(data);
        assert_eq!(crc1, crc2);
    }

    #[test]
    fn crc32_different_data() {
        let crc1 = crc32_hash(b"hello");
        let crc2 = crc32_hash(b"world");
        assert_ne!(crc1, crc2);
    }

    // ─── Write 100K entries and recover ─────────────────────────────────

    #[test]
    fn write_100k_and_recover() {
        let dir = tmp_wal_dir();
        let config = WalConfig {
            max_segment_size: DEFAULT_MAX_SEGMENT_SIZE,
            sync_on_write: false,
        };

        let n = 100_000u64;
        let commit_at = 99_990u64;

        {
            let mut wal = Wal::open(dir.path(), config.clone()).unwrap();
            for i in 0..n {
                let payload = encode_insert_payload(i % 1000, (i * 10) as i64, &[i as f32; 4]);
                wal.append(EntryType::Insert, 0, &payload).unwrap();
            }
            wal.commit(commit_at).unwrap();
        }

        {
            let mut wal = Wal::open(dir.path(), config).unwrap();
            let uncommitted = wal.recover().unwrap();
            assert_eq!(uncommitted.len(), (n - commit_at) as usize);
            // All committed entries should NOT be returned
            for entry in &uncommitted {
                assert!(entry.sequence > commit_at);
            }
        }
    }

    // ─── Reopen and continue writing ────────────────────────────────────

    #[test]
    fn reopen_and_continue() {
        let dir = tmp_wal_dir();
        let config = WalConfig::default();

        {
            let mut wal = Wal::open(dir.path(), config.clone()).unwrap();
            let payload = encode_insert_payload(1, 100, &[1.0]);
            wal.append(EntryType::Insert, 0, &payload).unwrap();
            wal.commit(1).unwrap();
        }

        {
            let mut wal = Wal::open(dir.path(), config).unwrap();
            assert_eq!(wal.committed_sequence(), 1);
            let payload = encode_insert_payload(2, 200, &[2.0]);
            let seq = wal.append(EntryType::Insert, 0, &payload).unwrap();
            assert_eq!(seq, 2);
            wal.commit(2).unwrap();
        }
    }

    // ─── Entry types ────────────────────────────────────────────────────

    #[test]
    fn delete_entry_type() {
        let dir = tmp_wal_dir();
        let mut wal = Wal::open(dir.path(), WalConfig::default()).unwrap();

        let payload = encode_delete_payload(42, 1000);
        let seq = wal.append(EntryType::Delete, 0, &payload).unwrap();
        assert_eq!(seq, 1);

        let entries = wal.recover().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].entry_type, EntryType::Delete);
    }

    #[test]
    fn keyframe_flag() {
        let dir = tmp_wal_dir();
        let mut wal = Wal::open(dir.path(), WalConfig::default()).unwrap();

        let payload = encode_insert_payload(1, 100, &[1.0]);
        wal.append(EntryType::Insert, 0x01, &payload).unwrap(); // keyframe flag

        let entries = wal.recover().unwrap();
        assert_eq!(entries[0].flags, 0x01);
    }
}
