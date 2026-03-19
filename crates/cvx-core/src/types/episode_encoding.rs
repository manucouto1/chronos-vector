//! Episode encoding helpers for episodic trace memory.
//!
//! Encodes `(episode_id, step_index)` into a single `u64` entity_id,
//! and provides episode-level trajectory retrieval utilities.
//!
//! # Encoding scheme
//!
//! ```text
//! entity_id = (episode_id << 16) | step_index
//!
//! Bits: [63..16] = episode_id (48 bits, max 281 trillion episodes)
//!       [15..0]  = step_index (16 bits, max 65535 steps per episode)
//! ```

/// Maximum step index per episode (2^16 - 1 = 65535).
pub const MAX_STEP_INDEX: u32 = 0xFFFF;

/// Maximum episode ID (2^48 - 1).
pub const MAX_EPISODE_ID: u64 = (1u64 << 48) - 1;

/// Encode an episode_id and step_index into a single u64 entity_id.
///
/// # Panics
///
/// Panics if `episode_id > MAX_EPISODE_ID` or `step_index > MAX_STEP_INDEX`.
pub fn encode_entity_id(episode_id: u64, step_index: u32) -> u64 {
    assert!(
        episode_id <= MAX_EPISODE_ID,
        "episode_id {episode_id} exceeds 48-bit limit"
    );
    assert!(
        step_index <= MAX_STEP_INDEX as u32,
        "step_index {step_index} exceeds 16-bit limit"
    );
    (episode_id << 16) | (step_index as u64)
}

/// Decode an entity_id back into (episode_id, step_index).
pub fn decode_entity_id(entity_id: u64) -> (u64, u32) {
    let episode_id = entity_id >> 16;
    let step_index = (entity_id & 0xFFFF) as u32;
    (episode_id, step_index)
}

/// Get the entity_id for step 0 of an episode (used for initial-state queries).
pub fn episode_start(episode_id: u64) -> u64 {
    encode_entity_id(episode_id, 0)
}

/// Get the range of entity_ids for all steps of an episode.
///
/// Returns `(start_entity_id, end_entity_id)` where the range is
/// `[start, end]` inclusive. Use with `TemporalFilter` or entity lookups.
pub fn episode_range(episode_id: u64) -> (u64, u64) {
    let start = encode_entity_id(episode_id, 0);
    let end = encode_entity_id(episode_id, MAX_STEP_INDEX as u32);
    (start, end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_roundtrip() {
        let episode = 42;
        let step = 7;
        let encoded = encode_entity_id(episode, step);
        let (ep, st) = decode_entity_id(encoded);
        assert_eq!(ep, episode);
        assert_eq!(st, step);
    }

    #[test]
    fn encode_step_zero() {
        let id = episode_start(100);
        let (ep, st) = decode_entity_id(id);
        assert_eq!(ep, 100);
        assert_eq!(st, 0);
    }

    #[test]
    fn episode_range_covers_all_steps() {
        let (start, end) = episode_range(42);
        let (ep_s, st_s) = decode_entity_id(start);
        let (ep_e, st_e) = decode_entity_id(end);
        assert_eq!(ep_s, 42);
        assert_eq!(st_s, 0);
        assert_eq!(ep_e, 42);
        assert_eq!(st_e, MAX_STEP_INDEX as u32);
    }

    #[test]
    fn different_episodes_dont_collide() {
        let a = encode_entity_id(1, 0);
        let b = encode_entity_id(2, 0);
        assert_ne!(a, b);
    }

    #[test]
    fn max_values() {
        let id = encode_entity_id(MAX_EPISODE_ID, MAX_STEP_INDEX as u32);
        let (ep, st) = decode_entity_id(id);
        assert_eq!(ep, MAX_EPISODE_ID);
        assert_eq!(st, MAX_STEP_INDEX as u32);
    }

    #[test]
    #[should_panic(expected = "exceeds 48-bit")]
    fn episode_overflow_panics() {
        encode_entity_id(MAX_EPISODE_ID + 1, 0);
    }

    #[test]
    #[should_panic(expected = "exceeds 16-bit")]
    fn step_overflow_panics() {
        encode_entity_id(0, MAX_STEP_INDEX as u32 + 1);
    }
}
