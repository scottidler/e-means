//! The trained model: final centroids, last-pass labels, per-pass statistics and the recipe that produced
//! them, plus final-nearest assignment.

use log::debug;
use rayon::prelude::*;

use crate::assign::{self, Kernel, Metric};
use crate::error::{Error, Result};
use crate::recipe::Recipe;
use crate::trainer::BLOCK_ROWS;

/// Statistics of one pass, recorded by `Trainer::end_epoch`.
#[derive(Debug, Clone, PartialEq)]
pub struct EpochStats {
    /// Sum over the pass's rows of the squared distance to the centroid each was assigned to, while the
    /// centroids were still moving. In spherical mode the per-row term is `2 - 2 * ip` against the unit-norm
    /// readout.
    pub objective: f64,
    /// `k * sum(count_j^2) / rows^2`; 1 when every cluster received the same number of rows.
    pub imbalance_factor: f64,
    /// Clusters revived by a split during the pass.
    pub splits: usize,
    /// Rows in the pass.
    pub rows: usize,
}

/// Nearest final centroid of each row, from `Model::assign`.
#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    /// Index of the nearest centroid per row; ties go to the lowest index.
    pub labels: Vec<u32>,
    /// Squared L2 distance from each row to its nearest centroid; in spherical mode `2 - 2 * ip`, which is
    /// the squared distance when the row has unit norm (deviation D10: the reference returns `ip`).
    pub distances: Vec<f32>,
}

/// The result of a run.
#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    d: usize,
    kernel: Kernel,
    metric: Metric,
    centroids: Vec<f32>,
    epoch_stats: Vec<EpochStats>,
    recipe: Recipe,
    labels: Option<Vec<Option<u32>>>,
}

impl Model {
    pub(crate) fn new(
        d: usize,
        kernel: Kernel,
        metric: Metric,
        centroids: Vec<f32>,
        epoch_stats: Vec<EpochStats>,
        recipe: Recipe,
        labels: Option<Vec<Option<u32>>>,
    ) -> Self {
        Model {
            d,
            kernel,
            metric,
            centroids,
            epoch_stats,
            recipe,
            labels,
        }
    }

    /// Final centroids, `k * d`, row-major.
    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }

    /// One entry per pass, in order.
    pub fn epoch_stats(&self) -> &[EpochStats] {
        &self.epoch_stats
    }

    /// The resolved recipe of the run.
    pub fn recipe(&self) -> &Recipe {
        &self.recipe
    }

    /// Last-pass labels, captured against centroids that were still moving; not final-nearest.
    ///
    /// From `EMeans::train` they are in row order, one per input row; a row the last pass did not visit
    /// (a fractional pass, or a row left out by `max_points_per_centroid`) is `None`. On the caller-driven
    /// path they are in feed order (position within the last pass), all `Some`. The whole result is `None`
    /// with `capture_labels(false)`, or when a last-pass batch was applied with `Trainer::update_with`,
    /// which carries no labels.
    pub fn labels(&self) -> Option<&[Option<u32>]> {
        self.labels.as_deref()
    }

    /// Nearest final centroid of each row of `rows` (row-major, `d` columns), with the same kernel and metric
    /// as the run. In spherical mode the nearest centroid is the one with the largest inner product; rows are
    /// not normalized here, that is the caller's job.
    ///
    /// `Error::DimensionMismatch` when `rows` is not whole rows; `Error::NonFiniteInput` on a NaN or
    /// infinite value, which would otherwise lose every comparison and land on centroid 0.
    pub fn assign(&self, rows: &[f32]) -> Result<Assignment> {
        debug!("Model::assign: values={} d={}", rows.len(), self.d);
        if !rows.len().is_multiple_of(self.d) {
            return Err(Error::DimensionMismatch(format!(
                "{} values is not a whole number of rows of d={}",
                rows.len(),
                self.d
            )));
        }
        if let Some(at) = rows.iter().position(|v| !v.is_finite()) {
            return Err(Error::NonFiniteInput(format!(
                "row {} column {} is {}",
                at / self.d,
                at % self.d,
                rows[at]
            )));
        }
        let n = rows.len() / self.d;
        let mut labels = vec![0; n];
        let mut distances = vec![0.0; n];
        // One block per task on the current `rayon` pool; each row's result depends only on the row.
        rows.par_chunks(BLOCK_ROWS * self.d)
            .zip(labels.par_chunks_mut(BLOCK_ROWS))
            .zip(distances.par_chunks_mut(BLOCK_ROWS))
            .for_each(|((block, block_labels), block_distances)| {
                assign::nearest(
                    self.kernel,
                    self.metric,
                    block,
                    &self.centroids,
                    self.d,
                    block_labels,
                    block_distances,
                );
            });
        if self.metric == Metric::InnerProduct {
            for distance in &mut distances {
                *distance = 2.0 - 2.0 * *distance;
            }
        }
        debug!("Model::assign: {n} rows assigned");
        Ok(Assignment { labels, distances })
    }

    /// Move the labels from feed positions to rows: the label at position `p` becomes row `rows[p]`'s, of
    /// `n` rows; rows no position names get `None`. No-op when labels were not captured.
    pub(crate) fn scatter_labels(&mut self, rows: &[usize], n: usize) {
        if let Some(fed) = self.labels.take() {
            assert!(fed.len() <= rows.len(), "more labels than row positions");
            let mut placed = vec![None; n];
            for (&row, label) in rows.iter().zip(fed) {
                placed[row] = label;
            }
            self.labels = Some(placed);
        }
    }
}

#[cfg(test)]
mod tests;
