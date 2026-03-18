use criterion::{Criterion, black_box, criterion_group, criterion_main};
use cvx_analytics::calculus;
use cvx_analytics::pelt::{self, PeltConfig};
use cvx_analytics::temporal_ml::{AnalyticBackend, TemporalOps};

fn make_trajectory(n: usize, dim: usize) -> Vec<(i64, Vec<f32>)> {
    (0..n)
        .map(|i| {
            let v: Vec<f32> = (0..dim).map(|d| (i * dim + d) as f32 * 0.001).collect();
            (i as i64 * 1_000_000, v)
        })
        .collect()
}

fn as_refs(points: &[(i64, Vec<f32>)]) -> Vec<(i64, &[f32])> {
    points.iter().map(|(t, v)| (*t, v.as_slice())).collect()
}

fn bench_velocity(c: &mut Criterion) {
    let points = make_trajectory(1000, 128);
    let traj = as_refs(&points);

    c.bench_function("velocity_1K_D128", |b| {
        b.iter(|| black_box(calculus::velocity(&traj, 500_000_000)))
    });
}

fn bench_pelt(c: &mut Criterion) {
    let mut group = c.benchmark_group("pelt");
    group.sample_size(20);

    for &n in &[100, 500, 1000] {
        let mut points: Vec<(i64, Vec<f32>)> = Vec::new();
        // 3 segments
        for i in 0..n {
            let seg = i / (n / 3);
            let base = seg as f32 * 5.0;
            points.push((i as i64 * 1000, vec![base, base * 0.5]));
        }
        let traj = as_refs(&points);

        group.bench_function(format!("n{n}_D2"), |b| {
            b.iter(|| black_box(pelt::detect(1, &traj, &PeltConfig::default())))
        });
    }
    group.finish();
}

fn bench_temporal_features(c: &mut Criterion) {
    let backend = AnalyticBackend::new();

    let mut group = c.benchmark_group("temporal_features");
    for &n in &[50, 200, 1000] {
        let points = make_trajectory(n, 32);
        let traj = as_refs(&points);

        group.bench_function(format!("n{n}_D32"), |b| {
            b.iter(|| black_box(backend.extract_features(&traj)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_velocity, bench_pelt, bench_temporal_features);
criterion_main!(benches);
