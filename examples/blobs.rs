//! Cluster four well-separated 2-d blobs with `EMeans::train`, then label the rows with `Model::assign`.
//!
//! The data comes from a fixed integer recipe, so the output is the same on every run and machine.
//!
//! ```text
//! cargo run --example blobs
//! ```

use e_means::EMeans;

const D: usize = 2;
const K: usize = 4;
const ROWS_PER_BLOB: usize = 100;
const CENTERS: [[f32; D]; K] = [[0.0, 0.0], [20.0, 0.0], [0.0, 20.0], [20.0, 20.0]];

/// A 64-bit LCG mapped to `[-1, 1)`: enough noise for a blob, no RNG dependency.
fn noise(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

fn main() {
    let mut state = 42;
    let mut x = Vec::with_capacity(K * ROWS_PER_BLOB * D);
    for i in 0..K * ROWS_PER_BLOB {
        let center = CENTERS[i % K];
        for c in center {
            x.push(c + noise(&mut state));
        }
    }

    let model = EMeans::new(D, K)
        .epochs(5.0)
        .seed(7)
        .train(&x)
        .expect("train on four blobs");

    for (j, c) in model.centroids().chunks(D).enumerate() {
        println!("centroid {j}: [{:.2}, {:.2}]", c[0], c[1]);
    }

    let assignment = model.assign(&x).expect("assign the training rows");
    for (blob, center) in CENTERS.iter().enumerate() {
        let label = assignment.labels[blob];
        let same_blob = assignment.labels.iter().skip(blob).step_by(K).all(|&l| l == label);
        assert!(same_blob, "blob {blob} was split across clusters");
        let c = &model.centroids()[label as usize * D..(label as usize + 1) * D];
        let off = ((c[0] - center[0]).powi(2) + (c[1] - center[1]).powi(2)).sqrt();
        assert!(off < 1.0, "centroid {label} is {off} from blob {blob}'s center");
    }
    let mut seen = assignment.labels.clone();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), K, "each blob should own one cluster");
    println!("{} rows, {K} clusters, every blob in its own cluster", x.len() / D);
}
