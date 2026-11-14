//! `EMeans::train` end to end at T=5, n=100000, k=256, the two configs of the design's reference timings
//! (64 cores, 2026-11-13): d=32 (throughput law, B=8192, `Kernel::Gemm`), `NumpyEmeans` 0.41 s; d=8
//! (quality law, B=1024, `Kernel::Direct`), `NumpyEmeans` 4.35 s. The Phase 9 bound is a median below each.
//!
//! ```text
//! cargo bench --bench train
//! ```

mod common;

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use e_means::EMeans;

fn train(c: &mut Criterion) {
    let mut group = c.benchmark_group("train");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));
    for d in [32, 8] {
        let x = common::mixture(d, 10);
        let builder = EMeans::new(d, common::K).epochs(5.0);
        group.bench_function(format!("n{}_d{d}_k{}_t5", common::N, common::K), |b| {
            b.iter(|| builder.train(black_box(&x)).expect("train"))
        });
    }
    group.finish();
}

criterion_group!(benches, train);
criterion_main!(benches);
