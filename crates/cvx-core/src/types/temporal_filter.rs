//! Temporal constraints for queries.

use serde::{Deserialize, Serialize};

/// Temporal filter applied to search and retrieval queries.
///
/// Controls which timestamps are considered during index traversal.
///
/// # Example
///
/// ```
/// use cvx_core::TemporalFilter;
///
/// let filter = TemporalFilter::Range(1000, 5000);
/// assert!(filter.matches(3000));
/// assert!(!filter.matches(6000));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalFilter {
    /// Exact timestamp match (snapshot query).
    Snapshot(i64),
    /// Inclusive time range `[start, end]`.
    Range(i64, i64),
    /// All timestamps before (inclusive).
    Before(i64),
    /// All timestamps after (inclusive).
    After(i64),
    /// No temporal constraint.
    All,
}

impl TemporalFilter {
    /// Check whether a timestamp satisfies this filter.
    pub fn matches(&self, timestamp: i64) -> bool {
        match self {
            Self::Snapshot(t) => timestamp == *t,
            Self::Range(start, end) => timestamp >= *start && timestamp <= *end,
            Self::Before(t) => timestamp <= *t,
            Self::After(t) => timestamp >= *t,
            Self::All => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_matches_exact() {
        let f = TemporalFilter::Snapshot(100);
        assert!(f.matches(100));
        assert!(!f.matches(99));
        assert!(!f.matches(101));
    }

    #[test]
    fn range_is_inclusive() {
        let f = TemporalFilter::Range(10, 20);
        assert!(f.matches(10));
        assert!(f.matches(15));
        assert!(f.matches(20));
        assert!(!f.matches(9));
        assert!(!f.matches(21));
    }

    #[test]
    fn before_is_inclusive() {
        let f = TemporalFilter::Before(100);
        assert!(f.matches(100));
        assert!(f.matches(50));
        assert!(!f.matches(101));
    }

    #[test]
    fn after_is_inclusive() {
        let f = TemporalFilter::After(100);
        assert!(f.matches(100));
        assert!(f.matches(200));
        assert!(!f.matches(99));
    }

    #[test]
    fn all_matches_everything() {
        let f = TemporalFilter::All;
        assert!(f.matches(i64::MIN));
        assert!(f.matches(0));
        assert!(f.matches(i64::MAX));
    }

    #[test]
    fn negative_timestamps_work() {
        let f = TemporalFilter::Range(-5000, -1000);
        assert!(f.matches(-3000));
        assert!(!f.matches(0));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn all_always_matches(ts in any::<i64>()) {
            prop_assert!(TemporalFilter::All.matches(ts));
        }

        #[test]
        fn snapshot_only_matches_exact(target in any::<i64>(), ts in any::<i64>()) {
            let f = TemporalFilter::Snapshot(target);
            prop_assert_eq!(f.matches(ts), ts == target);
        }

        #[test]
        fn range_is_consistent(a in any::<i64>(), b in any::<i64>(), ts in any::<i64>()) {
            let (start, end) = if a <= b { (a, b) } else { (b, a) };
            let f = TemporalFilter::Range(start, end);
            prop_assert_eq!(f.matches(ts), ts >= start && ts <= end);
        }
    }
}
