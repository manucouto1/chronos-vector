//! Big-endian key encoding for RocksDB with correct lexicographic ordering.
//!
//! Keys are encoded as `entity_id (8B) + space_id (4B) + timestamp (8B) = 20 bytes`.
//! All integers use big-endian encoding so that lexicographic byte comparison
//! matches numeric ordering.
//!
//! Timestamps (`i64`) use a **sign-bit flip** (XOR 0x80 on the first byte) so
//! that negative timestamps sort correctly:
//! - Without flip: `-1` = `FF..FF` sorts after `1` = `00..01` (wrong)
//! - With flip: `-1` = `7F..FF` sorts before `1` = `80..01` (correct)
//!
//! # Example
//!
//! ```
//! use cvx_storage::keys;
//!
//! let key = keys::encode_key(42, 0, 1_000_000);
//! let (entity_id, space_id, timestamp) = keys::decode_key(&key);
//! assert_eq!(entity_id, 42);
//! assert_eq!(space_id, 0);
//! assert_eq!(timestamp, 1_000_000);
//! ```

/// Total size of an encoded key in bytes.
pub const KEY_SIZE: usize = 20;

/// Prefix size for entity_id + space_id (used for prefix bloom filters).
pub const PREFIX_SIZE: usize = 12;

/// Encode a key as 20 big-endian bytes with sign-bit flip on timestamp.
pub fn encode_key(entity_id: u64, space_id: u32, timestamp: i64) -> [u8; KEY_SIZE] {
    let mut key = [0u8; KEY_SIZE];
    key[0..8].copy_from_slice(&entity_id.to_be_bytes());
    key[8..12].copy_from_slice(&space_id.to_be_bytes());
    key[12..20].copy_from_slice(&encode_timestamp(timestamp));
    key
}

/// Decode a 20-byte key back into its components.
pub fn decode_key(key: &[u8]) -> (u64, u32, i64) {
    let entity_id = u64::from_be_bytes(key[0..8].try_into().unwrap());
    let space_id = u32::from_be_bytes(key[8..12].try_into().unwrap());
    let timestamp = decode_timestamp(key[12..20].try_into().unwrap());
    (entity_id, space_id, timestamp)
}

/// Encode an entity_id + space_id prefix (12 bytes) for prefix scans.
pub fn encode_prefix(entity_id: u64, space_id: u32) -> [u8; PREFIX_SIZE] {
    let mut prefix = [0u8; PREFIX_SIZE];
    prefix[0..8].copy_from_slice(&entity_id.to_be_bytes());
    prefix[8..12].copy_from_slice(&space_id.to_be_bytes());
    prefix
}

/// Encode a timestamp with sign-bit flip for correct lexicographic ordering.
fn encode_timestamp(ts: i64) -> [u8; 8] {
    let mut bytes = ts.to_be_bytes();
    bytes[0] ^= 0x80;
    bytes
}

/// Decode a sign-bit-flipped timestamp.
fn decode_timestamp(bytes: &[u8; 8]) -> i64 {
    let mut b = *bytes;
    b[0] ^= 0x80;
    i64::from_be_bytes(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_basic() {
        let (e, s, t) = (42u64, 1u32, 1_000_000i64);
        let key = encode_key(e, s, t);
        let (e2, s2, t2) = decode_key(&key);
        assert_eq!((e, s, t), (e2, s2, t2));
    }

    #[test]
    fn roundtrip_negative_timestamp() {
        let key = encode_key(1, 0, -5_000_000);
        let (_, _, ts) = decode_key(&key);
        assert_eq!(ts, -5_000_000);
    }

    #[test]
    fn roundtrip_extremes() {
        for ts in [i64::MIN, i64::MIN + 1, -1, 0, 1, i64::MAX - 1, i64::MAX] {
            let key = encode_key(0, 0, ts);
            let (_, _, decoded) = decode_key(&key);
            assert_eq!(ts, decoded, "failed for ts={ts}");
        }
    }

    #[test]
    fn timestamp_ordering_preserved() {
        let timestamps = [i64::MIN, -1_000_000, -1, 0, 1, 1_000_000, i64::MAX];
        let keys: Vec<[u8; KEY_SIZE]> = timestamps.iter().map(|&ts| encode_key(1, 0, ts)).collect();

        for window in keys.windows(2) {
            assert!(
                window[0] < window[1],
                "ordering broken: {:?} should be < {:?}",
                window[0],
                window[1]
            );
        }
    }

    #[test]
    fn entity_ordering_preserved() {
        let k1 = encode_key(1, 0, 0);
        let k2 = encode_key(2, 0, 0);
        assert!(k1 < k2);
    }

    #[test]
    fn space_ordering_preserved() {
        let k1 = encode_key(1, 0, 0);
        let k2 = encode_key(1, 1, 0);
        assert!(k1 < k2);
    }

    #[test]
    fn prefix_matches_key_start() {
        let key = encode_key(42, 5, 1000);
        let prefix = encode_prefix(42, 5);
        assert_eq!(&key[..PREFIX_SIZE], &prefix);
    }

    #[test]
    fn key_size_is_20() {
        assert_eq!(KEY_SIZE, 20);
        assert_eq!(encode_key(0, 0, 0).len(), 20);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn roundtrip_arbitrary(
            entity_id in any::<u64>(),
            space_id in any::<u32>(),
            timestamp in any::<i64>(),
        ) {
            let key = encode_key(entity_id, space_id, timestamp);
            let (e, s, t) = decode_key(&key);
            prop_assert_eq!(entity_id, e);
            prop_assert_eq!(space_id, s);
            prop_assert_eq!(timestamp, t);
        }

        #[test]
        fn ordering_preserved(
            entity_id in any::<u64>(),
            space_id in any::<u32>(),
            t1 in any::<i64>(),
            t2 in any::<i64>(),
        ) {
            let k1 = encode_key(entity_id, space_id, t1);
            let k2 = encode_key(entity_id, space_id, t2);
            prop_assert_eq!(t1.cmp(&t2), k1.cmp(&k2));
        }
    }
}
