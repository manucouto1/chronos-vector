//! A query result scored by combined spatiotemporal distance.

use serde::{Deserialize, Serialize};

use super::TemporalPoint;

/// A search result combining semantic and temporal distance.
///
/// Produced by kNN queries. Contains the matched point and its distance
/// decomposition, enabling clients to understand *why* a result was ranked.
///
/// The combined score is $d_{ST} = \alpha \cdot d_{sem} + (1 - \alpha) \cdot d_{time} \cdot decay$.
///
/// # Example
///
/// ```
/// use cvx_core::{ScoredResult, TemporalPoint};
///
/// let point = TemporalPoint::new(42, 1000, vec![0.1, 0.2]);
/// let result = ScoredResult::new(point, 0.15, 0.05, 0.12);
/// assert!(result.combined_score() < result.semantic_distance());
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoredResult {
    point: TemporalPoint,
    semantic_distance: f32,
    temporal_distance: f32,
    combined_score: f32,
}

impl ScoredResult {
    /// Create a new scored result.
    pub fn new(
        point: TemporalPoint,
        semantic_distance: f32,
        temporal_distance: f32,
        combined_score: f32,
    ) -> Self {
        Self {
            point,
            semantic_distance,
            temporal_distance,
            combined_score,
        }
    }

    /// The matched temporal point.
    pub fn point(&self) -> &TemporalPoint {
        &self.point
    }

    /// Pure semantic distance (cosine, L2, etc.).
    pub fn semantic_distance(&self) -> f32 {
        self.semantic_distance
    }

    /// Temporal distance component.
    pub fn temporal_distance(&self) -> f32 {
        self.temporal_distance
    }

    /// Combined spatiotemporal score used for ranking.
    pub fn combined_score(&self) -> f32 {
        self.combined_score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accessors_work() {
        let p = TemporalPoint::new(1, 100, vec![0.5]);
        let r = ScoredResult::new(p.clone(), 0.2, 0.1, 0.15);
        assert_eq!(r.point(), &p);
        assert!((r.semantic_distance() - 0.2).abs() < f32::EPSILON);
        assert!((r.temporal_distance() - 0.1).abs() < f32::EPSILON);
        assert!((r.combined_score() - 0.15).abs() < f32::EPSILON);
    }

    #[test]
    fn postcard_roundtrip() {
        let p = TemporalPoint::new(42, 1000, vec![0.1, 0.2, 0.3]);
        let r = ScoredResult::new(p, 0.5, 0.3, 0.42);
        let bytes = postcard::to_allocvec(&r).unwrap();
        let recovered: ScoredResult = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(r, recovered);
    }
}
