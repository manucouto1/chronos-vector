//! Metadata about an entity's history in a given embedding space.

use serde::{Deserialize, Serialize};

/// Summary metadata for an entity's trajectory within one embedding space.
///
/// Stored in the `timelines` column family, keyed by `(entity_id, space_id)`.
/// Provides quick access to temporal bounds and point count without scanning the full trajectory.
///
/// # Example
///
/// ```
/// use cvx_core::EntityTimeline;
///
/// let tl = EntityTimeline::new(42, 0, 1000, 5000, 100, 10);
/// assert_eq!(tl.duration(), 4000);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntityTimeline {
    entity_id: u64,
    space_id: u32,
    first_seen: i64,
    last_seen: i64,
    point_count: u32,
    keyframe_interval: u32,
}

impl EntityTimeline {
    /// Create a new entity timeline record.
    pub fn new(
        entity_id: u64,
        space_id: u32,
        first_seen: i64,
        last_seen: i64,
        point_count: u32,
        keyframe_interval: u32,
    ) -> Self {
        Self {
            entity_id,
            space_id,
            first_seen,
            last_seen,
            point_count,
            keyframe_interval,
        }
    }

    /// The entity ID.
    pub fn entity_id(&self) -> u64 {
        self.entity_id
    }

    /// The embedding space ID.
    pub fn space_id(&self) -> u32 {
        self.space_id
    }

    /// Timestamp of the first observation.
    pub fn first_seen(&self) -> i64 {
        self.first_seen
    }

    /// Timestamp of the most recent observation.
    pub fn last_seen(&self) -> i64 {
        self.last_seen
    }

    /// Total number of temporal points stored.
    pub fn point_count(&self) -> u32 {
        self.point_count
    }

    /// How often a full keyframe is stored (every N updates).
    pub fn keyframe_interval(&self) -> u32 {
        self.keyframe_interval
    }

    /// Duration of the timeline in timestamp units (microseconds).
    pub fn duration(&self) -> i64 {
        self.last_seen - self.first_seen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_is_correct() {
        let tl = EntityTimeline::new(1, 0, 1000, 5000, 10, 10);
        assert_eq!(tl.duration(), 4000);
    }

    #[test]
    fn negative_timestamps_work() {
        let tl = EntityTimeline::new(1, 0, -5000, -1000, 5, 10);
        assert_eq!(tl.duration(), 4000);
    }

    #[test]
    fn postcard_roundtrip() {
        let tl = EntityTimeline::new(42, 1, -1_000_000, 1_000_000, 500, 10);
        let bytes = postcard::to_allocvec(&tl).unwrap();
        let recovered: EntityTimeline = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(tl, recovered);
    }
}
