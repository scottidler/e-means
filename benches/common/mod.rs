//! Bench data: the seeded Gaussian mixture of `tests/quality.rs` (256 equal-size components, centers
//! ~ N(0, 3^2) per coordinate, unit noise), in any `d`.

pub const N: usize = 100_000;
pub const K: usize = 256;
pub const CENTER_SCALE: f64 = 3.0;

/// SplitMix64 and Box-Muller: the bench's own seeded stream, independent of the crate's RNG.
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

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// `N` rows of `d` columns, row `i` from component `i % K`.
pub fn mixture(d: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix(seed);
    let centers: Vec<f64> = (0..K * d).map(|_| CENTER_SCALE * rng.normal()).collect();
    let mut x = Vec::with_capacity(N * d);
    for i in 0..N {
        let c = i % K;
        for j in 0..d {
            x.push((centers[c * d + j] + rng.normal()) as f32);
        }
    }
    x
}
