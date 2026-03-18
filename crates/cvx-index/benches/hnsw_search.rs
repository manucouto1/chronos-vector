use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use cvx_core::TemporalFilter;
use cvx_index::hnsw::temporal::TemporalHnsw;
use cvx_index::hnsw::{HnswConfig, HnswGraph};
use cvx_index::metrics::L2Distance;

fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    // Deterministic pseudo-random for reproducible benchmarks
    let mut state = seed;
    (0..n)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((state >> 33) as f32) / (u32::MAX as f32) - 0.5
                })
                .collect()
        })
        .collect()
}

fn build_graph(n: u32, dim: usize) -> HnswGraph<L2Distance> {
    let config = HnswConfig {
        m: 16,
        ef_construction: 200,
        ef_search: 50,
        ..Default::default()
    };
    let mut graph = HnswGraph::new(config, L2Distance);
    let vectors = random_vectors(n as usize, dim, 42);
    for (i, v) in vectors.iter().enumerate() {
        graph.insert(i as u32, v);
    }
    graph
}

fn build_temporal(n: u32, dim: usize) -> TemporalHnsw<L2Distance> {
    let config = HnswConfig {
        m: 16,
        ef_construction: 200,
        ef_search: 50,
        ..Default::default()
    };
    let mut index = TemporalHnsw::new(config, L2Distance);
    let vectors = random_vectors(n as usize, dim, 42);
    for (i, v) in vectors.iter().enumerate() {
        index.insert(i as u64, (i as i64) * 1000, v);
    }
    index
}

fn bench_vanilla_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search");
    group.sample_size(50);

    for &(n, dim) in &[(1_000u32, 32), (1_000, 128), (10_000, 128)] {
        let graph = build_graph(n, dim);
        let queries = random_vectors(100, dim, 123);

        group.bench_with_input(
            BenchmarkId::new(format!("vanilla_k10_D{dim}"), n),
            &n,
            |bencher, _| {
                let mut qi = 0;
                bencher.iter(|| {
                    let q = &queries[qi % queries.len()];
                    qi += 1;
                    black_box(graph.search(q, 10))
                });
            },
        );
    }
    group.finish();
}

fn bench_temporal_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("st_hnsw_search");
    group.sample_size(50);

    let n = 1_000u32;
    let dim = 32;
    let index = build_temporal(n, dim);
    let queries = random_vectors(100, dim, 456);

    // Pure semantic
    group.bench_function("alpha1.0_All_k10", |bencher| {
        let mut qi = 0;
        bencher.iter(|| {
            let q = &queries[qi % queries.len()];
            qi += 1;
            black_box(index.search(q, 10, TemporalFilter::All, 1.0, 500_000))
        });
    });

    // With temporal filter
    group.bench_function("alpha1.0_Range_k10", |bencher| {
        let mut qi = 0;
        bencher.iter(|| {
            let q = &queries[qi % queries.len()];
            qi += 1;
            black_box(index.search(q, 10, TemporalFilter::Range(250_000, 750_000), 1.0, 500_000))
        });
    });

    // Mixed alpha
    group.bench_function("alpha0.5_All_k10", |bencher| {
        let mut qi = 0;
        bencher.iter(|| {
            let q = &queries[qi % queries.len()];
            qi += 1;
            black_box(index.search(q, 10, TemporalFilter::All, 0.5, 500_000))
        });
    });

    group.finish();
}

fn bench_insert_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_insert");
    group.sample_size(10);

    let dim = 128;
    let vectors = random_vectors(5_000, dim, 789);

    group.bench_function("5K_D128", |bencher| {
        bencher.iter(|| {
            let config = HnswConfig {
                m: 16,
                ef_construction: 200,
                ..Default::default()
            };
            let mut graph = HnswGraph::new(config, L2Distance);
            for (i, v) in vectors.iter().enumerate() {
                graph.insert(i as u32, v);
            }
            black_box(graph.len())
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_vanilla_search,
    bench_temporal_search,
    bench_insert_throughput
);
criterion_main!(benches);
