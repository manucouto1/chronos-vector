//! Performance metrics report for the HNSW index.
//!
//! Run with: `cargo test -p cvx-index --test metrics_report -- --nocapture`

use cvx_core::TemporalFilter;
use cvx_index::hnsw::temporal::TemporalHnsw;
use cvx_index::hnsw::{HnswConfig, HnswGraph, recall_at_k};
use cvx_index::metrics::{CosineDistance, L2Distance};

fn random_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::rng();
    (0..n)
        .map(|_| {
            (0..dim)
                .map(|_| rand::Rng::random::<f32>(&mut rng))
                .collect()
        })
        .collect()
}

fn report_vanilla_recall_with<D: cvx_core::DistanceMetric>(
    metric: D,
    n: u32,
    dim: usize,
    k: usize,
    ef_search: usize,
    label: &str,
) {
    let config = HnswConfig {
        m: 16,
        ef_construction: 200,
        ef_search,
        ..Default::default()
    };
    let mut graph = HnswGraph::new(config, metric);

    let vectors = random_vectors(n as usize, dim);
    for (i, v) in vectors.iter().enumerate() {
        graph.insert(i as u32, v);
    }

    let reachable = graph.count_reachable();
    let reachability = reachable as f64 / n as f64 * 100.0;

    let n_queries = 100;
    let mut total_recall = 0.0;
    let queries = random_vectors(n_queries, dim);

    for q in &queries {
        let approx = graph.search(q, k);
        let truth = graph.brute_force_knn(q, k);
        total_recall += recall_at_k(&approx, &truth);
    }
    let avg_recall = total_recall / n_queries as f64;

    println!(
        "  {label:<45} recall@{k}={avg_recall:.4}  reachability={reachability:.1}%  nodes={n}"
    );
}

fn report_temporal_recall(
    n: u32,
    dim: usize,
    k: usize,
    filter: TemporalFilter,
    alpha: f32,
    label: &str,
) {
    let config = HnswConfig {
        m: 16,
        ef_construction: 200,
        ef_search: 200,
        ..Default::default()
    };
    let mut index = TemporalHnsw::new(config, L2Distance);
    let mut rng = rand::rng();

    for i in 0..n {
        let ts = (i as i64) * 100;
        let v: Vec<f32> = (0..dim)
            .map(|_| rand::Rng::random::<f32>(&mut rng))
            .collect();
        index.insert(i as u64, ts, &v);
    }

    let n_queries = 100;
    let mut total_recall = 0.0;
    let queries = random_vectors(n_queries, dim);

    // Build filtered ground truth
    let bitmap = index.build_filter_bitmap(&filter);

    for q in &queries {
        let results = index.search(q, k, filter, alpha, 50_000);

        // Brute force within filter
        let mut truth: Vec<(u32, f32)> = (0..n)
            .filter(|&i| bitmap.contains(i))
            .map(|i| {
                let d = index
                    .graph()
                    .brute_force_knn(q, n as usize)
                    .iter()
                    .find(|&&(id, _)| id == i)
                    .map(|&(_, d)| d)
                    .unwrap_or(f32::INFINITY);
                (i, d)
            })
            .collect();
        truth.sort_by(|a, b| a.1.total_cmp(&b.1));
        truth.truncate(k);

        total_recall += recall_at_k(&results, &truth);
    }
    let avg_recall = total_recall / n_queries as f64;

    println!("  {label:<45} recall@{k}={avg_recall:.4}  alpha={alpha}  nodes={n}");
}

#[test]
fn metrics_report() {
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║              ChronosVector Performance Metrics                  ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");

    // ─── Vanilla HNSW Recall (L2) ──────────────────────────────────────
    println!("║ Vanilla HNSW — L2Distance (heuristic=true)                     ║");
    println!("╟──────────────────────────────────────────────────────────────────╢");
    report_vanilla_recall_with(L2Distance, 1_000, 32, 10, 50, "L2  1K D=32  ef=50");
    report_vanilla_recall_with(L2Distance, 1_000, 128, 10, 50, "L2  1K D=128 ef=50");
    report_vanilla_recall_with(L2Distance, 10_000, 128, 10, 200, "L2  10K D=128 ef=200");

    // ─── Vanilla HNSW Recall (Cosine) ───────────────────────────────────
    println!("╟──────────────────────────────────────────────────────────────────╢");
    println!("║ Vanilla HNSW — CosineDistance (heuristic=true)                 ║");
    println!("╟──────────────────────────────────────────────────────────────────╢");
    report_vanilla_recall_with(CosineDistance, 1_000, 32, 10, 50, "Cos 1K D=32  ef=50");
    report_vanilla_recall_with(CosineDistance, 1_000, 128, 10, 50, "Cos 1K D=128 ef=50");

    // ─── ST-HNSW Temporal Recall ─────────────────────────────────────
    println!("╟──────────────────────────────────────────────────────────────────╢");
    println!("║ ST-HNSW Temporal (L2Distance)                                  ║");
    println!("╟──────────────────────────────────────────────────────────────────╢");
    report_temporal_recall(
        1_000,
        32,
        10,
        TemporalFilter::All,
        1.0,
        "1K D=32, alpha=1.0 (pure semantic)",
    );
    report_temporal_recall(
        1_000,
        32,
        10,
        TemporalFilter::Range(25_000, 75_000),
        1.0,
        "1K D=32, Range[25K,75K], alpha=1.0",
    );
    report_temporal_recall(
        1_000,
        32,
        10,
        TemporalFilter::All,
        0.5,
        "1K D=32, alpha=0.5 (mixed)",
    );

    // ─── Alpha comparison ────────────────────────────────────────────
    println!("╟──────────────────────────────────────────────────────────────────╢");
    println!("║ Alpha Temporal Proximity Effect                                 ║");
    println!("╟──────────────────────────────────────────────────────────────────╢");
    {
        let config = HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            ..Default::default()
        };
        let mut index = TemporalHnsw::new(config, L2Distance);
        let mut rng = rand::rng();
        let dim = 16;
        let n = 500;

        for i in 0..n {
            let ts = (i as i64) * 1000;
            let v: Vec<f32> = (0..dim)
                .map(|_| rand::Rng::random::<f32>(&mut rng))
                .collect();
            index.insert(i as u64, ts, &v);
        }

        let query: Vec<f32> = (0..dim)
            .map(|_| rand::Rng::random::<f32>(&mut rng))
            .collect();
        let query_ts = 250_000; // middle
        let k = 10;

        for alpha in [1.0, 0.75, 0.5, 0.25, 0.0] {
            let results = index.search(&query, k, TemporalFilter::All, alpha, query_ts);
            let avg_temporal_dist: f64 = results
                .iter()
                .map(|&(id, _)| (index.timestamp(id) - query_ts).unsigned_abs() as f64)
                .sum::<f64>()
                / k as f64;

            println!("  alpha={alpha:.2}  avg_temporal_dist={avg_temporal_dist:>10.0}µs");
        }
    }

    // ─── Bitmap memory ───────────────────────────────────────────────
    println!("╟──────────────────────────────────────────────────────────────────╢");
    println!("║ Roaring Bitmap Memory                                          ║");
    println!("╟──────────────────────────────────────────────────────────────────╢");
    {
        for n in [1_000, 10_000, 100_000] {
            // Rebuild for each size
            let mut idx = TemporalHnsw::new(HnswConfig::default(), L2Distance);
            for i in 0..n as u32 {
                idx.insert(i as u64, i as i64, &[i as f32]);
            }
            let mem = idx.bitmap_memory_bytes();
            let bytes_per_vec = mem as f64 / n as f64;
            println!("  {n:>7} vectors: {mem:>6} bytes total, {bytes_per_vec:.3} bytes/vector");
        }
    }

    println!("╚══════════════════════════════════════════════════════════════════╝\n");
}
