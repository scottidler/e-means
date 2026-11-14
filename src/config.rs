//! The `EMeans` builder: options, validation, and `begin`.

use log::debug;

use crate::error::{Error, Result};
use crate::recipe::{BatchLaw, Recipe, RecipeSpec};
use crate::trainer::Trainer;

/// The reference constructor's default seed (`REF/e_means.py:169`).
const DEFAULT_SEED: u64 = 1234;

/// Builder for an e-means run. Options follow the reference constructor (`REF/e_means.py:161-177`).
#[derive(Debug, Clone, PartialEq)]
pub struct EMeans {
    d: usize,
    k: usize,
    epochs: f64,
    batch_law: BatchLaw,
    batch_size: Option<usize>,
    alpha0: Option<f64>,
    unused_threshold: Option<f32>,
    seed: u64,
}

impl EMeans {
    /// `k` clusters of `d`-dimensional rows, 5 epochs, `BatchLaw::Auto`.
    pub fn new(d: usize, k: usize) -> Self {
        EMeans {
            d,
            k,
            epochs: 5.0,
            batch_law: BatchLaw::Auto,
            batch_size: None,
            alpha0: None,
            unused_threshold: None,
            seed: DEFAULT_SEED,
        }
    }

    /// Epoch budget T; fractional values add a partial pass before the last full one.
    pub fn epochs(mut self, epochs: f64) -> Self {
        self.epochs = epochs;
        self
    }

    /// Which fitted law picks B when `batch_size` is not given.
    pub fn batch_law(mut self, law: BatchLaw) -> Self {
        self.batch_law = law;
        self
    }

    /// Rows per logical batch; skips the batch law, keeps the step law. Capped at n.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Initial EMA step in `(0, 1]`; skips the step law.
    pub fn alpha0(mut self, alpha0: f64) -> Self {
        self.alpha0 = Some(alpha0);
        self
    }

    /// Revival threshold tau as a fraction of the mean mass; 0 disables revival.
    pub fn unused_threshold(mut self, tau: f32) -> Self {
        self.unused_threshold = Some(tau);
        self
    }

    /// Seed of every random draw in the run; revival's donor choice reads it.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Start a caller-driven run over `n` rows from `init`, `k * d` initial centroids.
    ///
    /// Every error is found before a trainer exists.
    pub fn begin(&self, n: usize, init: &[f32]) -> Result<Trainer> {
        debug!(
            "EMeans::begin: n={n} d={} k={} epochs={} batch_law={:?} batch_size={:?} alpha0={:?} unused_threshold={:?} seed={}",
            self.d, self.k, self.epochs, self.batch_law, self.batch_size, self.alpha0, self.unused_threshold, self.seed
        );
        self.validate_shape()?;
        if n < self.k {
            return Err(Error::TooFewRows { rows: n, k: self.k });
        }
        if init.len() != self.k * self.d {
            return Err(Error::DimensionMismatch(format!(
                "init has {} values, want k*d = {}",
                init.len(),
                self.k * self.d
            )));
        }
        if let Some(at) = init.iter().position(|v| !v.is_finite()) {
            return Err(Error::NonFiniteInput(format!(
                "init centroid {} coordinate {} is {}",
                at / self.d,
                at % self.d,
                init[at]
            )));
        }
        let recipe = Recipe::resolve(&RecipeSpec {
            n,
            k: self.k,
            d: self.d,
            epochs: self.epochs,
            batch_law: self.batch_law,
            batch_size: self.batch_size,
            alpha0: self.alpha0,
            unused_threshold: self.unused_threshold,
        })?;
        Ok(Trainer::new(self.d, self.k, recipe, init, self.seed))
    }

    fn validate_shape(&self) -> Result<()> {
        if self.d == 0 {
            return Err(Error::InvalidParameter("d must be >= 1".to_string()));
        }
        if self.k == 0 {
            return Err(Error::InvalidParameter("k must be >= 1".to_string()));
        }
        if u32::try_from(self.k).is_err() {
            return Err(Error::InvalidParameter(format!(
                "k = {} does not fit a u32 label",
                self.k
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
