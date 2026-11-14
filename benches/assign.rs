//! `Model::assign` over n=100000 rows at k=256, both kernels, d=8 and d=32. The kernel is the run's, picked
//! by the recipe's `B * d` against 128000, so each model is trained with a batch size on one side of it.
//!
//! ```text
//! cargo bench --bench assign
//! ```

mod common;

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use e_means::EMeans;

fn assign(c: &mut Criterion) {
    let mut group = c.benchmark_group("assign");
    group.sample_size(20);
    group.throughput(Throughput::Elements(common::N as u64));
    for d in [8, 32] {
        let x = common::mixture(d, 10);
        // Direct below the threshold, Gemm at or above it.
        for (kernel, batch) in [("direct", 1024), ("gemm", 128_000_usize.div_ceil(d))] {
            let model = EMeans::new(d, common::K)
                .epochs(1.0)
                .batch_size(batch)
                .train(&x)
                .expect("train");
            group.bench_function(format!("{kernel}_d{d}_k{}", common::K), |b| {
                b.iter(|| model.assign(black_box(&x)).expect("assign"))
            });
        }
    }
    group.finish();
}

criterion_group!(benches, assign);
criterion_main!(benches);
