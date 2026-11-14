//! The `EMeans` builder: options, validation, `begin`, and `train` / `train_from`.
//!
//! `train` is a loop over the caller-driven `Trainer`: every pass feeds batches in a fresh seeded order
//! (`REF/e_means.py:566-636`).

use log::debug;

use crate::error::{Error, Result};
use crate::model::Model;
use crate::recipe::{BatchLaw, Recipe, RecipeSpec};
use crate::rng::{Stream, StreamId};
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
    max_points_per_centroid: Option<usize>,
    capture_labels: bool,
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
            max_points_per_centroid: None,
            capture_labels: true,
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

    /// `train` / `train_from` only: when n exceeds `k * limit`, train on a seeded subset of `k * limit` rows,
    /// kept in row order; the other rows get `None` labels (`REF/base.py:206-226`). Must be >= 1.
    pub fn max_points_per_centroid(mut self, limit: usize) -> Self {
        self.max_points_per_centroid = Some(limit);
        self
    }

    /// Capture last-pass labels (default true); false skips the label buffer and `Model::labels` is `None`.
    pub fn capture_labels(mut self, capture: bool) -> Self {
        self.capture_labels = capture;
        self
    }

    /// Train on `x` (row-major, `d` columns) from seeded initial centroids.
    ///
    /// Init is the first k rows of a seeded permutation, or the rows as given when n == k
    /// (`REF/e_means.py:593-600`). Labels come back in row order.
    pub fn train(&self, x: &[f32]) -> Result<Model> {
        debug!("EMeans::train: values={} d={} k={}", x.len(), self.d, self.k);
        self.train_subset(x, None)
    }

    /// Train on `x` from caller-supplied initial centroids, `k * d` values.
    pub fn train_from(&self, x: &[f32], init: &[f32]) -> Result<Model> {
        debug!(
            "EMeans::train_from: values={} init={} d={} k={}",
            x.len(),
            init.len(),
            self.d,
            self.k
        );
        self.train_subset(x, Some(init))
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
        Ok(Trainer::new(
            self.d,
            self.k,
            recipe,
            init,
            self.seed,
            self.capture_labels,
        ))
    }

    /// Apply `max_points_per_centroid`, train on the kept rows, and put labels back on the rows of `x`.
    fn train_subset(&self, x: &[f32], init: Option<&[f32]>) -> Result<Model> {
        self.validate_shape()?;
        if !x.len().is_multiple_of(self.d) {
            return Err(Error::DimensionMismatch(format!(
                "x has {} values, not a whole number of rows of d={}",
                x.len(),
                self.d
            )));
        }
        let n = x.len() / self.d;
        let Some(keep) = self.training_subset(n)? else {
            return self.train_rows(x, init);
        };
        let rows = gather(x, &keep, self.d);
        let mut model = self.train_rows(&rows, init)?;
        model.scatter_labels(&keep, n);
        Ok(model)
    }

    /// Sorted row ids kept by `max_points_per_centroid`, or `None` for all rows (`REF/base.py:206-226`).
    fn training_subset(&self, n: usize) -> Result<Option<Vec<usize>>> {
        let Some(limit) = self.max_points_per_centroid else {
            return Ok(None);
        };
        if limit == 0 {
            return Err(Error::InvalidParameter(
                "max_points_per_centroid must be >= 1".to_string(),
            ));
        }
        let kept = self.k.saturating_mul(limit);
        if n <= kept {
            return Ok(None);
        }
        debug!("EMeans::training_subset: sampling {kept} of {n} rows (max_points_per_centroid={limit})");
        let mut keep = Stream::new(self.seed, StreamId::Subset).permutation(n);
        keep.truncate(kept);
        keep.sort_unstable();
        Ok(Some(keep))
    }

    /// The training loop: seeded init, then every pass in a fresh seeded order of its own stream.
    fn train_rows(&self, x: &[f32], init: Option<&[f32]>) -> Result<Model> {
        let d = self.d;
        let n = x.len() / d;
        if n < self.k {
            return Err(Error::TooFewRows { rows: n, k: self.k });
        }
        let drawn;
        let init = match init {
            Some(init) => init,
            None if n == self.k => x,
            None => {
                let order = Stream::new(self.seed, StreamId::Init).permutation(n);
                drawn = gather(x, &order[..self.k], d);
                &drawn
            }
        };
        let mut trainer = self.begin(n, init)?;
        let batch = trainer.batch_rows();
        let passes = trainer.recipe().epoch_rows.clone();
        let mut rows = Vec::with_capacity(batch * d);
        let mut last_order = Vec::new();
        for (pass, &pass_rows) in passes.iter().enumerate() {
            let order = Stream::new(self.seed, StreamId::Pass(pass as u64)).permutation(n);
            for ids in order[..pass_rows].chunks(batch) {
                gather_into(&mut rows, x, ids, d);
                trainer.accumulate(&rows)?;
                trainer.update()?;
            }
            trainer.end_epoch()?;
            last_order = order;
        }
        let mut model = trainer.finalize()?;
        model.scatter_labels(&last_order, n);
        Ok(model)
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

/// The rows `ids` of `x`, in that order.
fn gather(x: &[f32], ids: &[usize], d: usize) -> Vec<f32> {
    let mut rows = Vec::with_capacity(ids.len() * d);
    gather_into(&mut rows, x, ids, d);
    rows
}

/// Replace the contents of `rows` with the rows `ids` of `x`, in that order.
fn gather_into(rows: &mut Vec<f32>, x: &[f32], ids: &[usize], d: usize) {
    rows.clear();
    for &i in ids {
        rows.extend_from_slice(&x[i * d..(i + 1) * d]);
    }
}

#[cfg(test)]
mod tests;
