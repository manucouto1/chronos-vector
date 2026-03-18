//! Topological features via persistent homology.
//!
//! Tracks topological changes in point clouds over time using Betti numbers
//! from Vietoris-Rips persistent homology. Detects structural shifts like
//! "the topic space fragmented" or "two clusters merged".
//!
//! # Betti Numbers
//!
//! - β₀ = number of connected components (clusters)
//! - β₁ = number of loops (cyclic patterns)
//!
//! Changes in Betti numbers over time reveal structural evolution:
//! - Increasing β₀ → fragmentation
//! - Decreasing β₀ → convergence
//! - β₁ appearing → cyclic behavior emerges
//!
//! # Implementation
//!
//! Uses Vietoris-Rips filtration on pairwise distances between points.
//! Applied on **region centroids** (K~80 at L3), NOT raw points, for tractability.
//!
//! # References
//!
//! - Edelsbrunner, H. & Harer, J. (2010). *Computational Topology*. AMS.
//! - Zigzag persistence for temporal networks. *EPJ Data Science*, 2023.

/// A persistence interval: a topological feature that is "born" at one
/// filtration radius and "dies" at another.
#[derive(Debug, Clone, PartialEq)]
pub struct PersistenceInterval {
    /// Homology dimension (0 = component, 1 = loop).
    pub dimension: usize,
    /// Birth radius: when this feature appears.
    pub birth: f64,
    /// Death radius: when this feature disappears. f64::INFINITY for essential features.
    pub death: f64,
}

impl PersistenceInterval {
    /// Persistence = death - birth. Longer-lived features are more significant.
    pub fn persistence(&self) -> f64 {
        if self.death.is_infinite() {
            f64::INFINITY
        } else {
            self.death - self.birth
        }
    }
}

/// Persistence diagram: collection of birth-death intervals.
#[derive(Debug, Clone)]
pub struct PersistenceDiagram {
    /// All persistence intervals.
    pub intervals: Vec<PersistenceInterval>,
    /// Number of points in the input.
    pub n_points: usize,
}

impl PersistenceDiagram {
    /// Count features alive at a given radius (Betti number).
    pub fn betti(&self, dimension: usize, radius: f64) -> usize {
        self.intervals.iter()
            .filter(|iv| iv.dimension == dimension && iv.birth <= radius && iv.death > radius)
            .count()
    }

    /// Compute Betti curve: β(r) for a range of radii.
    pub fn betti_curve(&self, dimension: usize, radii: &[f64]) -> Vec<usize> {
        radii.iter().map(|&r| self.betti(dimension, r)).collect()
    }

    /// Total persistence for a given dimension (sum of lifetimes, excluding infinite).
    pub fn total_persistence(&self, dimension: usize) -> f64 {
        self.intervals.iter()
            .filter(|iv| iv.dimension == dimension && iv.death.is_finite())
            .map(|iv| iv.persistence())
            .sum()
    }

    /// Number of features with persistence above a threshold.
    pub fn n_significant(&self, dimension: usize, min_persistence: f64) -> usize {
        self.intervals.iter()
            .filter(|iv| iv.dimension == dimension && iv.persistence() > min_persistence)
            .count()
    }
}

/// Compute Vietoris-Rips persistent homology (dimension 0 only).
///
/// Dimension 0 (connected components) tracks how clusters merge as the
/// filtration radius grows. This is equivalent to single-linkage clustering.
///
/// # Algorithm
///
/// 1. Compute pairwise distance matrix.
/// 2. Sort edges by distance (Kruskal's algorithm).
/// 3. Use Union-Find to track component merges.
/// 4. Each merge creates a death event for the younger component.
///
/// # Complexity
///
/// O(N² log N) for N points (dominated by sorting N² edges).
///
/// For region centroids (N~80), this is ~6,400 edges — instant.
pub fn vietoris_rips_h0(points: &[&[f32]]) -> PersistenceDiagram {
    let n = points.len();
    if n == 0 {
        return PersistenceDiagram { intervals: vec![], n_points: 0 };
    }

    // All points born at radius 0
    let mut birth_times = vec![0.0f64; n];

    // Compute and sort all pairwise edges
    let mut edges: Vec<(f64, usize, usize)> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            edges.push((l2_dist(points[i], points[j]), i, j));
        }
    }
    edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Union-Find
    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank = vec![0usize; n];

    let find = |parent: &mut Vec<usize>, mut x: usize| -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]]; // path compression
            x = parent[x];
        }
        x
    };

    let mut intervals = Vec::new();

    for (dist, i, j) in edges {
        let ri = find(&mut parent, i);
        let rj = find(&mut parent, j);
        if ri == rj {
            continue; // already in same component
        }

        // Merge: younger component dies (the one born later, or arbitrary if same)
        let (survivor, dying) = if birth_times[ri] <= birth_times[rj] {
            (ri, rj)
        } else {
            (rj, ri)
        };

        intervals.push(PersistenceInterval {
            dimension: 0,
            birth: birth_times[dying],
            death: dist,
        });

        // Union by rank
        if rank[survivor] < rank[dying] {
            parent[survivor] = dying;
            birth_times[dying] = birth_times[dying].min(birth_times[survivor]);
        } else {
            parent[dying] = survivor;
            if rank[survivor] == rank[dying] {
                rank[survivor] += 1;
            }
        }
    }

    // The last surviving component has infinite persistence
    let final_root = find(&mut parent, 0);
    intervals.push(PersistenceInterval {
        dimension: 0,
        birth: birth_times[final_root],
        death: f64::INFINITY,
    });

    PersistenceDiagram { intervals, n_points: n }
}

/// Topological summary features extracted from a persistence diagram.
#[derive(Debug, Clone)]
pub struct TopologicalFeatures {
    /// Number of significant components (persistence > threshold).
    pub n_components: usize,
    /// Total persistence of H₀ (sum of lifetimes).
    pub total_persistence_h0: f64,
    /// Max persistence (most prominent feature).
    pub max_persistence: f64,
    /// Mean persistence of finite intervals.
    pub mean_persistence: f64,
    /// Persistence entropy: -Σ (p_i / P) log(p_i / P) where p_i = persistence of interval i.
    pub persistence_entropy: f64,
    /// Betti curve at selected radii.
    pub betti_curve: Vec<usize>,
    /// Radii used for Betti curve.
    pub radii: Vec<f64>,
}

/// Extract topological summary features from a point cloud.
///
/// Computes H₀ persistent homology and summarizes into a fixed-size feature vector.
///
/// # Arguments
///
/// * `points` - Point cloud (e.g., region centroids).
/// * `n_radii` - Number of radii for Betti curve sampling (default 20).
/// * `persistence_threshold` - Minimum persistence to count as significant (default 0.1).
pub fn topological_summary(
    points: &[&[f32]],
    n_radii: usize,
    persistence_threshold: f64,
) -> TopologicalFeatures {
    let diagram = vietoris_rips_h0(points);

    let finite: Vec<f64> = diagram.intervals.iter()
        .filter(|iv| iv.death.is_finite())
        .map(|iv| iv.persistence())
        .collect();

    let max_persistence = finite.iter().cloned().fold(0.0f64, f64::max);
    let total = finite.iter().sum::<f64>();
    let mean_persistence = if finite.is_empty() { 0.0 } else { total / finite.len() as f64 };

    // Persistence entropy
    let persistence_entropy = if total > 0.0 {
        finite.iter()
            .map(|&p| {
                let q = p / total;
                if q > 0.0 { -q * q.ln() } else { 0.0 }
            })
            .sum()
    } else {
        0.0
    };

    // Betti curve: sample at evenly-spaced radii from 0 to max_death
    let max_death = finite.iter().cloned().fold(0.0f64, f64::max);
    let radii: Vec<f64> = (0..n_radii)
        .map(|i| max_death * (i as f64 + 0.5) / n_radii as f64)
        .collect();
    let betti_curve = diagram.betti_curve(0, &radii);

    let n_components = diagram.n_significant(0, persistence_threshold);

    TopologicalFeatures {
        n_components,
        total_persistence_h0: total,
        max_persistence,
        mean_persistence,
        persistence_entropy,
        betti_curve,
        radii,
    }
}

/// L2 distance between two vectors.
#[inline]
fn l2_dist(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b.iter())
        .map(|(x, y)| { let d = *x as f64 - *y as f64; d * d })
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_point_one_component() {
        let points: Vec<&[f32]> = vec![&[0.0f32, 0.0]];
        let d = vietoris_rips_h0(&points);
        assert_eq!(d.intervals.len(), 1);
        assert!(d.intervals[0].death.is_infinite()); // one eternal component
        assert_eq!(d.betti(0, 0.0), 1);
    }

    #[test]
    fn two_distant_clusters() {
        // Two well-separated clusters
        let points: Vec<&[f32]> = vec![
            &[0.0f32, 0.0], &[0.1, 0.0], &[0.0, 0.1],  // cluster A
            &[10.0, 10.0], &[10.1, 10.0], &[10.0, 10.1], // cluster B
        ];
        let d = vietoris_rips_h0(&points);

        // At small radius: 2 components (β₀ = 2 for the two clusters)
        assert_eq!(d.betti(0, 0.5), 2, "should see 2 clusters at r=0.5");

        // At large radius: 1 component (everything connected)
        assert_eq!(d.betti(0, 20.0), 1, "should see 1 component at r=20");
    }

    #[test]
    fn three_equidistant_points() {
        // Equilateral triangle: all pairwise distances = 1.0
        let s = 3.0f32.sqrt() / 2.0;
        let p0 = [0.0f32, 0.0];
        let p1 = [1.0f32, 0.0];
        let p2 = [0.5f32, s];
        let points: Vec<&[f32]> = vec![&p0, &p1, &p2];
        let d = vietoris_rips_h0(&points);

        // 3 points → 2 death events + 1 infinite
        assert_eq!(d.intervals.len(), 3);
        assert_eq!(d.betti(0, 0.5), 3); // all separate
        assert_eq!(d.betti(0, 1.5), 1); // all connected
    }

    #[test]
    fn betti_curve_monotone_decreasing() {
        let points: Vec<&[f32]> = vec![
            &[0.0f32], &[1.0], &[3.0], &[6.0], &[10.0],
        ];
        let d = vietoris_rips_h0(&points);
        let radii: Vec<f64> = (0..20).map(|i| i as f64 * 0.6).collect();
        let curve = d.betti_curve(0, &radii);

        // β₀ should be non-increasing
        for i in 1..curve.len() {
            assert!(
                curve[i] <= curve[i - 1],
                "β₀ should be non-increasing: β₀[{}]={} > β₀[{}]={}",
                i, curve[i], i - 1, curve[i - 1]
            );
        }
    }

    #[test]
    fn topological_summary_two_clusters() {
        let points: Vec<&[f32]> = vec![
            &[0.0f32, 0.0], &[0.1, 0.0],
            &[10.0, 0.0], &[10.1, 0.0],
        ];
        let feat = topological_summary(&points, 20, 1.0);

        // Two clusters with large gap → at least 1 significant component separation
        assert!(feat.n_components >= 1, "should detect cluster structure");
        assert!(feat.max_persistence > 5.0, "max persistence should reflect cluster gap");
    }

    #[test]
    fn persistence_entropy_uniform() {
        // Many points equally spaced → all intervals have similar persistence
        let pts: Vec<[f32; 1]> = (0..10).map(|i| [i as f32]).collect();
        let point_refs: Vec<&[f32]> = pts.iter().map(|p| p.as_slice()).collect();
        let feat = topological_summary(&point_refs, 10, 0.01);

        // Uniform persistence → high entropy
        assert!(feat.persistence_entropy > 1.0, "uniform should have high entropy, got {}", feat.persistence_entropy);
    }

    #[test]
    fn empty_points() {
        let points: Vec<&[f32]> = vec![];
        let d = vietoris_rips_h0(&points);
        assert_eq!(d.intervals.len(), 0);
        assert_eq!(d.n_points, 0);
    }
}
