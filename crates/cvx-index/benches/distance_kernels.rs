use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use cvx_core::DistanceMetric;
use cvx_index::metrics::{CosineDistance, DotProductDistance, L2Distance};

fn generate_vectors(dim: usize) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.00123).sin()).collect();
    let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.00456).cos()).collect();
    (a, b)
}

fn bench_cosine(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_distance");
    for dim in [32, 128, 256, 768, 1536] {
        let (a, b) = generate_vectors(dim);
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bencher, _| {
            bencher.iter(|| CosineDistance.distance(black_box(&a), black_box(&b)));
        });
    }
    group.finish();
}

fn bench_l2(c: &mut Criterion) {
    let mut group = c.benchmark_group("l2_distance");
    for dim in [32, 128, 256, 768, 1536] {
        let (a, b) = generate_vectors(dim);
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bencher, _| {
            bencher.iter(|| L2Distance.distance(black_box(&a), black_box(&b)));
        });
    }
    group.finish();
}

fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product_distance");
    for dim in [32, 128, 256, 768, 1536] {
        let (a, b) = generate_vectors(dim);
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bencher, _| {
            bencher.iter(|| DotProductDistance.distance(black_box(&a), black_box(&b)));
        });
    }
    group.finish();
}

fn bench_throughput_d768(c: &mut Criterion) {
    let dim = 768;
    let n_pairs = 10_000;
    let vectors_a: Vec<Vec<f32>> = (0..n_pairs)
        .map(|i| {
            (0..dim)
                .map(|j| ((i * dim + j) as f32 * 0.001).sin())
                .collect()
        })
        .collect();
    let vectors_b: Vec<Vec<f32>> = (0..n_pairs)
        .map(|i| {
            (0..dim)
                .map(|j| ((i * dim + j) as f32 * 0.002).cos())
                .collect()
        })
        .collect();

    c.bench_function("cosine_10k_pairs_d768", |bencher| {
        bencher.iter(|| {
            for (a, b) in vectors_a.iter().zip(vectors_b.iter()) {
                black_box(CosineDistance.distance(a, b));
            }
        });
    });
}

criterion_group!(
    benches,
    bench_cosine,
    bench_l2,
    bench_dot_product,
    bench_throughput_d768
);
criterion_main!(benches);
