#![allow(clippy::needless_range_loop)]
//! Performance metrics report for analytics modules.
//!
//! Run with: `cargo test -p cvx-analytics --test metrics_report -- --nocapture`

use cvx_analytics::bocpd::{BocpdConfig, BocpdDetector};
use cvx_analytics::calculus;
use cvx_analytics::pelt::{self, PeltConfig};
use cvx_analytics::temporal_ml::{AnalyticBackend, TemporalOps};

fn make_trajectory(points: &[(i64, Vec<f32>)]) -> Vec<(i64, &[f32])> {
    points.iter().map(|(t, v)| (*t, v.as_slice())).collect()
}

#[test]
fn metrics_report() {
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║              ChronosVector Analytics Metrics                    ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");

    // ─── PELT Change Point Detection ─────────────────────────────────
    println!("║ PELT Change Point Detection                                    ║");
    println!("╟──────────────────────────────────────────────────────────────────╢");

    // Stationary
    {
        let points: Vec<(i64, Vec<f32>)> = (0..200)
            .map(|i| (i as i64 * 1000, vec![1.0, 2.0, 3.0]))
            .collect();
        let traj = make_trajectory(&points);
        let cps = pelt::detect(1, &traj, &PeltConfig::default());
        println!(
            "  Stationary (200 pts, D=3):         detected={:<3} expected=0   FP={}",
            cps.len(),
            cps.len()
        );
    }

    // 1 change point
    {
        let mut points: Vec<(i64, Vec<f32>)> = Vec::new();
        for i in 0..100 {
            points.push((i as i64 * 1000, vec![0.0, 0.0]));
        }
        for i in 100..200 {
            points.push((i as i64 * 1000, vec![10.0, 10.0]));
        }
        let traj = make_trajectory(&points);
        let cps = pelt::detect(1, &traj, &PeltConfig::default());
        let near_100 = cps.iter().any(|cp| (cp.timestamp() - 100_000).abs() < 5000);
        println!(
            "  1 planted CP (200 pts, D=2):       detected={:<3} near_target={} severity={:.3}",
            cps.len(),
            near_100,
            cps.first().map(|c| c.severity()).unwrap_or(0.0)
        );
    }

    // 3 change points
    {
        let means = [
            vec![0.0, 0.0],
            vec![5.0, 5.0],
            vec![0.0, 10.0],
            vec![10.0, 0.0],
        ];
        let mut points: Vec<(i64, Vec<f32>)> = Vec::new();
        for seg in 0..4 {
            for i in 0..50 {
                let idx = seg * 50 + i;
                points.push((idx as i64 * 1000, means[seg].clone()));
            }
        }
        let traj = make_trajectory(&points);
        let cps = pelt::detect(1, &traj, &PeltConfig::default());

        let expected_ts = [50_000, 100_000, 150_000];
        let mut found = 0;
        for &expected in &expected_ts {
            if cps
                .iter()
                .any(|cp| (cp.timestamp() - expected).abs() < 5000)
            {
                found += 1;
            }
        }
        let precision = found as f64 / cps.len().max(1) as f64;
        let recall = found as f64 / 3.0;
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        println!(
            "  3 planted CPs (200 pts, D=2):      detected={:<3} precision={:.3} recall={:.3} F1={:.3}",
            cps.len(),
            precision,
            recall,
            f1
        );
    }

    // ─── Online Detector ─────────────────────────────────────────────
    println!("╟──────────────────────────────────────────────────────────────────╢");
    println!("║ Online Change Point Detector (EWMA)                            ║");
    println!("╟──────────────────────────────────────────────────────────────────╢");

    // FPR on stationary
    {
        let config = BocpdConfig {
            alpha: 0.05,
            threshold_sigmas: 3.0,
            min_observations: 10,
            cooldown: 5,
        };
        let mut det = BocpdDetector::new(1, config);
        let mut fp = 0;
        let n = 500;
        for i in 0..n {
            if det.observe(i as i64 * 1000, &[1.0, 2.0, 3.0]).is_some() {
                fp += 1;
            }
        }
        let fpr = fp as f64 / n as f64;
        println!("  Stationary FPR (500 obs, D=3):     FP={fp:<3} FPR={fpr:.4}");
    }

    // Detection latency
    {
        let config = BocpdConfig {
            alpha: 0.1,
            threshold_sigmas: 3.0,
            min_observations: 5,
            cooldown: 3,
        };
        let mut det = BocpdDetector::new(1, config);
        let change_at = 100;
        let mut latency = None;
        for i in 0..200 {
            let v = if i < change_at {
                vec![0.0, 0.0, 0.0]
            } else {
                vec![10.0, 10.0, 10.0]
            };
            if let Some(_cp) = det.observe(i as i64 * 1000, &v) {
                if latency.is_none() && i >= change_at {
                    latency = Some(i - change_at);
                }
            }
        }
        println!(
            "  Detection latency (change@100):    latency={} observations",
            latency
                .map(|l| l.to_string())
                .unwrap_or("NOT DETECTED".into())
        );
    }

    // ─── Hurst Exponent ──────────────────────────────────────────────
    println!("╟──────────────────────────────────────────────────────────────────╢");
    println!("║ Stochastic Characterization                                    ║");
    println!("╟──────────────────────────────────────────────────────────────────╢");

    // Brownian motion
    {
        let mut rng = rand::rng();
        let n = 2000;
        let mut x = 0.0f32;
        let points: Vec<(i64, Vec<f32>)> = (0..n)
            .map(|i| {
                x += rand::Rng::random::<f32>(&mut rng) - 0.5;
                (i as i64 * 1000, vec![x])
            })
            .collect();
        let traj = make_trajectory(&points);
        let h = calculus::hurst_exponent(&traj).unwrap();
        let adf = calculus::adf_statistic(&traj).unwrap();
        println!("  Brownian motion (n={n}):           Hurst={h:.3} (expect≈0.5)  ADF={adf:.3}");
    }

    // Mean-reverting (OU process)
    {
        let mut rng = rand::rng();
        let n = 2000;
        let mut x = 0.0f32;
        let theta = 0.3;
        let mu = 5.0f32;
        let points: Vec<(i64, Vec<f32>)> = (0..n)
            .map(|i| {
                x += theta * (mu - x) + 0.5 * (rand::Rng::random::<f32>(&mut rng) - 0.5);
                (i as i64 * 1000, vec![x])
            })
            .collect();
        let traj = make_trajectory(&points);
        let h = calculus::hurst_exponent(&traj).unwrap();
        let adf = calculus::adf_statistic(&traj).unwrap();
        println!(
            "  OU process (n={n}, θ=0.3):         Hurst={h:.3} (expect<0.5)  ADF={adf:.3} (expect<-2.86)"
        );
    }

    // ─── Temporal ML Features ────────────────────────────────────────
    println!("╟──────────────────────────────────────────────────────────────────╢");
    println!("║ Temporal ML Feature Extraction                                 ║");
    println!("╟──────────────────────────────────────────────────────────────────╢");

    {
        let backend = AnalyticBackend::new();
        for n in [10, 50, 100, 500] {
            let points: Vec<(i64, Vec<f32>)> = (0..n)
                .map(|i| (i as i64 * 1_000_000, vec![i as f32; 8]))
                .collect();
            let traj = make_trajectory(&points);
            let feats = backend.extract_features(&traj).unwrap();
            println!(
                "  n={n:<4} D=8 → {dim} features  drift={:.4}  vol[0]={:.6}  cp_count={:.1}",
                feats[8],  // drift magnitude
                feats[9],  // vol dim 0
                feats[17], // soft cp count (2*8+1 = 17)
                dim = feats.len()
            );
        }
    }

    println!("╚══════════════════════════════════════════════════════════════════╝\n");
}
