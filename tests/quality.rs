//! The quality claim, measured in-repo: e-means at 5 epochs against Lloyd at 25 iterations.
//!
//! Data: a seeded Gaussian mixture, 256 equal-size components in d = 32 (the shape of the parity fixture
//! `large`), centers ~ N(0, 3^2) per coordinate, unit noise; 100000 training rows and 20000 held-out rows
//! from the same mixture. Lloyd is the baseline the design names: 25 iterations over all rows from a
//! random-row init with the same seed, an empty cluster keeping its position (no split). Held-out MSE is
//! the mean over held-out rows of the squared distance to the nearest centroid, in f64 by direct
//! difference.

use std::thread;

use e_means::EMeans;

const D: usize = 32;
const K: usize = 256;
const TRAIN_ROWS: usize = 100_000;
const HELD_OUT_ROWS: usize = 20_000;
const CENTER_SCALE: f64 = 3.0;
const LLOYD_ITERATIONS: usize = 25;
const EMEANS_EPOCHS: f64 = 5.0;

/// SplitMix64: the test's own seeded stream, independent of the crate's RNG.
struct SplitMix(u64);

impl SplitMix {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `(0, 1]`, 53 bits.
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) + 1) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal by Box-Muller (one of the pair).
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

/// `rows` rows of the mixture, row i from component `i % K` (equal-size components, as in the parity
/// fixtures' blobs).
fn mixture_rows(rng: &mut SplitMix, centers: &[f64], rows: usize) -> Vec<f32> {
    let mut x = Vec::with_capacity(rows * D);
    for i in 0..rows {
        let c = i % K;
        for j in 0..D {
            x.push((centers[c * D + j] + rng.normal()) as f32);
        }
    }
    x
}

/// Training rows and held-out rows of the mixture for `seed`.
fn mixture(seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut rng = SplitMix(seed);
    let centers: Vec<f64> = (0..K * D).map(|_| CENTER_SCALE * rng.normal()).collect();
    let train = mixture_rows(&mut rng, &centers, TRAIN_ROWS);
    let held_out = mixture_rows(&mut rng, &centers, HELD_OUT_ROWS);
    (train, held_out)
}

fn squared_distance(row: &[f32], centroid: &[f32]) -> f64 {
    row.iter()
        .zip(centroid)
        .map(|(&x, &c)| (f64::from(x) - f64::from(c)).powi(2))
        .sum()
}

fn nearest(row: &[f32], centroids: &[f32]) -> (usize, f64) {
    centroids
        .as_chunks::<D>()
        .0
        .iter()
        .enumerate()
        .map(|(j, c)| (j, squared_distance(row, c)))
        .fold((0, f64::INFINITY), |best, cur| if cur.1 < best.1 { cur } else { best })
}

/// Mean over `rows` of the squared distance to the nearest centroid.
fn held_out_mse(rows: &[f32], centroids: &[f32]) -> f64 {
    let total: f64 = rows
        .as_chunks::<D>()
        .0
        .iter()
        .map(|row| nearest(row, centroids).1)
        .sum();
    total / (rows.len() / D) as f64
}

/// Lloyd's algorithm from k distinct random rows, `iterations` full assign-then-mean rounds.
fn lloyd(x: &[f32], seed: u64, iterations: usize) -> Vec<f32> {
    let n = x.len() / D;
    let mut rng = SplitMix(seed ^ 0x4C4C_4F59_4421_4E49);
    let mut ids: Vec<usize> = (0..n).collect();
    for i in 0..K {
        let j = i + rng.below(n - i);
        ids.swap(i, j);
    }
    let mut centroids: Vec<f32> = ids[..K].iter().flat_map(|&i| x[i * D..(i + 1) * D].to_vec()).collect();
    for _ in 0..iterations {
        let mut sums = vec![0f64; K * D];
        let mut counts = vec![0u64; K];
        for row in x.as_chunks::<D>().0 {
            let (j, _) = nearest(row, &centroids);
            counts[j] += 1;
            for (s, &v) in sums[j * D..(j + 1) * D].iter_mut().zip(row) {
                *s += f64::from(v);
            }
        }
        for (j, &count) in counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            for (c, &s) in centroids[j * D..(j + 1) * D].iter_mut().zip(&sums[j * D..(j + 1) * D]) {
                *c = (s / count as f64) as f32;
            }
        }
    }
    centroids
}

/// Train both on the mixture for `seed` and return their held-out MSEs, e-means first.
fn compare(seed: u64) -> (f64, f64) {
    let (train, held_out) = mixture(seed);
    let (emeans, lloyd) = thread::scope(|scope| {
        let lloyd = scope.spawn(|| lloyd(&train, seed, LLOYD_ITERATIONS));
        let model = EMeans::new(D, K)
            .epochs(EMEANS_EPOCHS)
            .seed(seed)
            .train(&train)
            .expect("e-means trains on the mixture");
        (model.centroids().to_vec(), lloyd.join().expect("Lloyd thread"))
    });
    let (e, l) = (held_out_mse(&held_out, &emeans), held_out_mse(&held_out, &lloyd));
    println!("seed {seed}: held-out MSE e-means T=5 {e:.4}, Lloyd-25 {l:.4}");
    (e, l)
}

fn check(seed: u64) {
    let (e, l) = compare(seed);
    assert!(
        e < l,
        "seed {seed}: e-means held-out MSE {e} is not below Lloyd-25's {l}"
    );
}

#[test]
fn emeans_beats_lloyd_seed_10() {
    check(10);
}

#[test]
fn emeans_beats_lloyd_seed_11() {
    check(11);
}

#[test]
fn emeans_beats_lloyd_seed_12() {
    check(12);
}

#[test]
fn mixture_is_seeded() {
    let (a, ha) = mixture(10);
    let (b, hb) = mixture(10);
    let (c, _) = mixture(11);
    assert_eq!(a, b);
    assert_eq!(ha, hb);
    assert_ne!(a, c);
    assert_eq!(a.len(), TRAIN_ROWS * D);
    assert_eq!(ha.len(), HELD_OUT_ROWS * D);
}
