//! Train at large k and print the wall time of `train()` alone, plus a digest of the result.
//!
//! n=200000, d=32, k=65536, T=1 on the seeded mixture of `benches/common/mod.rs` (256 components, the
//! generator `otto bench` uses). Data generation is outside the timed region. The digest is FNV-1a over the
//! bits of the final centroids then the captured labels: two runs with the same digest produced the same
//! result, so a speedup that moved a bit shows up here. At 64 threads, read peak memory with
//! `/usr/bin/time -f "%M KB" target/release/examples/large-k`.
//!
//! ```text
//! cargo run --release --example large-k
//! ```

#[allow(dead_code)]
#[path = "../benches/common/mod.rs"]
mod common;

use std::time::Instant;

use e_means::EMeans;

const N: usize = 200_000;
const D: usize = 32;
const K: usize = 65_536;
const DATA_SEED: u64 = 10;

/// FNV-1a 64 over a byte stream.
fn fnv1a(bytes: impl IntoIterator<Item = u8>) -> u64 {
    bytes.into_iter().fold(0xCBF2_9CE4_8422_2325u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01B3)
    })
}

fn main() {
    let x = common::mixture_of(N, common::K, D, DATA_SEED);
    println!(
        "large-k: n={N} d={D} k={K} T=1 threads={}",
        rayon::current_num_threads()
    );
    let started = Instant::now();
    let model = EMeans::new(D, K).epochs(1.0).train(&x).expect("train");
    let train = started.elapsed();
    let labels = model.labels().expect("labels are captured by default");
    let digest = fnv1a(
        model.centroids().iter().flat_map(|c| c.to_bits().to_le_bytes()).chain(
            labels
                .iter()
                .flat_map(|l| l.expect("every row was visited at T=1").to_le_bytes()),
        ),
    );
    println!("train: {:.3} s", train.as_secs_f64());
    println!("digest: {digest:016x}");
}
