//! The crate's seeded random stream.
//!
//! `ChaCha8Rng` supplies the bit stream and nothing else. This module owns everything that turns a `u64`
//! seed into numbers, so a `rand_*` release cannot change a result: the seed expansion (four SplitMix64
//! outputs, little-endian, as the 32-byte ChaCha key), the stream id per use, bounded integers (Lemire's
//! multiply with rejection), Fisher-Yates, and the 53-bit uniform. The golden vectors in `rng/tests.rs`
//! fail if any of it moves.

use rand_chacha::ChaCha8Rng;
use rand_core::{Rng, SeedableRng};

/// SplitMix64 increment (the golden gamma).
const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
/// 2^-53: scales the top 53 bits of a `u64` into `[0, 1)`.
const UNIFORM_SCALE: f64 = 1.0 / (1u64 << 53) as f64;

/// Which ChaCha stream of a seed a draw comes from; each use gets its own, so none shifts another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StreamId {
    /// Row order of pass `i`: stream id `i`.
    Pass(u64),
    /// Initial centroid rows: reserved id `u64::MAX`.
    Init,
    /// Revival donor choice: reserved id `u64::MAX - 1`.
    Revival,
    /// Rows kept by `max_points_per_centroid`: reserved id `u64::MAX - 2`.
    Subset,
}

impl StreamId {
    fn id(self) -> u64 {
        match self {
            StreamId::Pass(pass) => pass,
            StreamId::Init => u64::MAX,
            StreamId::Revival => u64::MAX - 1,
            StreamId::Subset => u64::MAX - 2,
        }
    }
}

/// One seeded stream of random numbers.
#[derive(Debug, Clone)]
pub(crate) struct Stream {
    rng: ChaCha8Rng,
}

impl Stream {
    /// The stream `id` of `seed`.
    pub(crate) fn new(seed: u64, id: StreamId) -> Self {
        let mut rng = ChaCha8Rng::from_seed(expand_seed(seed));
        rng.set_stream(id.id());
        Stream { rng }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.rng.next_u64()
    }

    /// Uniform in `[0, 1)` from the top 53 bits of one `u64`.
    pub(crate) fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * UNIFORM_SCALE
    }

    /// Uniform in `[0, bound)`, unbiased: Lemire's multiply-shift with rejection.
    ///
    /// Panics on `bound == 0`, which has no value to return.
    pub(crate) fn below(&mut self, bound: u64) -> u64 {
        assert!(bound > 0, "Stream::below needs a positive bound");
        let mut product = u128::from(self.next_u64()) * u128::from(bound);
        if (product as u64) < bound {
            let reject_below = bound.wrapping_neg() % bound;
            while (product as u64) < reject_below {
                product = u128::from(self.next_u64()) * u128::from(bound);
            }
        }
        (product >> 64) as u64
    }

    /// A uniformly random permutation of `0..n` (Durstenfeld's Fisher-Yates, from the top).
    pub(crate) fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = self.below(i as u64 + 1) as usize;
            order.swap(i, j);
        }
        order
    }
}

/// The 32-byte ChaCha key of a seed: four SplitMix64 outputs, each little-endian.
fn expand_seed(seed: u64) -> [u8; 32] {
    let mut state = seed;
    let mut key = [0u8; 32];
    for word in key.as_chunks_mut::<8>().0 {
        *word = splitmix64(&mut state).to_le_bytes();
    }
    key
}

/// One SplitMix64 output (Steele, Lea and Flood 2014).
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(SPLITMIX_GAMMA);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests;
