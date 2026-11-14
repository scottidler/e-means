//! The trained model: final centroids, per-pass statistics and the recipe that produced them.

use crate::recipe::Recipe;

/// Statistics of one pass, recorded by `Trainer::end_epoch`.
#[derive(Debug, Clone, PartialEq)]
pub struct EpochStats {
    /// Sum over the pass's rows of the squared distance to the centroid each was assigned to, while the
    /// centroids were still moving.
    pub objective: f64,
    /// `k * sum(count_j^2) / rows^2`; 1 when every cluster received the same number of rows.
    pub imbalance_factor: f64,
    /// Clusters revived by a split during the pass.
    pub splits: usize,
    /// Rows in the pass.
    pub rows: usize,
}

/// The result of a run.
#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    centroids: Vec<f32>,
    epoch_stats: Vec<EpochStats>,
    recipe: Recipe,
    labels: Option<Vec<Option<u32>>>,
}

impl Model {
    pub(crate) fn new(
        centroids: Vec<f32>,
        epoch_stats: Vec<EpochStats>,
        recipe: Recipe,
        labels: Option<Vec<Option<u32>>>,
    ) -> Self {
        Model {
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
    /// On the caller-driven path they are in feed order (position within the last pass), all `Some`.
    /// `None` when a last-pass batch was applied with `Trainer::update_with`, which carries no labels.
    pub fn labels(&self) -> Option<&[Option<u32>]> {
        self.labels.as_deref()
    }
}

#[cfg(test)]
mod tests;
