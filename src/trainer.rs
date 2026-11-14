//! The caller-driven trainer: `accumulate` rows of a logical batch, `update` the EMA, `end_epoch`, `finalize`.
//!
//! `REF` is `contrib/e_means` in Faiss at `d1d9789a`. The EMA follows `REF/e_means.py:456-505`; the
//! batch and pass accounting follows `REF/base.py:243-286` and `:439-457`; closing follows
//! `REF/e_means.py:531-564`. Spherical mode follows `REF/e_means.py:217-241`, `:303-307`, `:432-434` and
//! `:512-529`.

use log::{debug, trace};
use rayon::prelude::*;

use crate::assign::{self, Kernel, Metric};
use crate::error::{Error, Result};
use crate::model::{EpochStats, Model};
use crate::recipe::{CHECK_UNUSED_EVERY, INIT_USAGE, Recipe, scheduled_alpha};
use crate::revival;
use crate::rng::{Stream, StreamId};
use crate::spherical;

/// Rows per assignment block, counted from the start of the logical batch.
///
/// A block's labels depend only on its rows and the frozen snapshot, so how the rows arrive in chunks
/// never changes the result.
pub(crate) const BLOCK_ROWS: usize = 256;

/// Statistics of one logical batch computed outside the trainer (`REF/base.py:249-255`).
#[derive(Debug, Clone, Copy)]
pub struct BatchStats<'a> {
    /// Per-cluster row sums, `k * d`, row-major.
    pub sums: &'a [f64],
    /// Rows per cluster, `k`.
    pub counts: &'a [u64],
    /// Sum of the rows' squared distances to their assigned centroid; in spherical mode, of `2 - 2 * ip`.
    pub objective: f64,
}

/// Counts, sums and objective of the logical batch being fed.
#[derive(Debug)]
struct BatchAccumulator {
    counts: Vec<u64>,
    sums: Vec<f64>,
    objective: f64,
    /// Rows already assigned; rows waiting in `carry` are not counted.
    assigned: usize,
}

/// The e-means state machine of one run.
#[derive(Debug)]
pub struct Trainer {
    d: usize,
    k: usize,
    recipe: Recipe,
    kernel: Kernel,
    /// `Metric::InnerProduct` in spherical mode.
    metric: Metric,
    /// EMA mass N_j per cluster.
    usage: Vec<f32>,
    /// EMA centroid c_j, `k * d`; un-normalized in spherical mode.
    centroids: Vec<f32>,
    /// What every row of the current batch is assigned against. Euclidean: the centroids, frozen at the
    /// first chunk of the batch. Spherical: their unit-norm readout, refreshed after each update.
    snapshot: Vec<f32>,
    batch: BatchAccumulator,
    /// Rows of a partial block, waiting for the next chunk.
    carry: Vec<f32>,
    block_labels: Vec<u32>,
    /// Per-row score of the block's assignment: squared distance, or inner product in spherical mode.
    block_scores: Vec<f32>,
    pass_counts: Vec<u64>,
    /// Closing-pass sums; accumulated only in the last pass.
    pass_sums: Vec<f64>,
    pass_objective: f64,
    /// Rows of the current pass applied by `update`.
    pass_rows: usize,
    pass_splits: usize,
    /// Last-pass labels in feed order; `None` outside the last pass, with capture off, or once `update_with`
    /// hides some rows.
    pass_labels: Option<Vec<u32>>,
    capture_labels: bool,
    epoch_stats: Vec<EpochStats>,
    step: usize,
    epoch: usize,
    poisoned: bool,
    /// Revival's donor draws; consumed only by revival events that split.
    split_rng: Stream,
}

impl BatchAccumulator {
    fn new(k: usize, d: usize) -> Self {
        BatchAccumulator {
            counts: vec![0; k],
            sums: vec![0.0; k * d],
            objective: 0.0,
            assigned: 0,
        }
    }

    fn clear(&mut self) {
        self.counts.fill(0);
        self.sums.fill(0.0);
        self.objective = 0.0;
        self.assigned = 0;
    }
}

impl Trainer {
    /// Start a run from validated parts; `EMeans::begin` owns the validation.
    ///
    /// Spherical mode (`Metric::InnerProduct`) normalizes the initial centroids (`REF/e_means.py:322-328`).
    pub(crate) fn new(
        d: usize,
        k: usize,
        recipe: Recipe,
        init: &[f32],
        seed: u64,
        capture_labels: bool,
        metric: Metric,
    ) -> Self {
        let kernel = Kernel::for_batch(recipe.batch_size, d);
        debug!(
            "Trainer::new: d={d} k={k} batch={} alpha0={} passes={:?} total_steps={} seed={seed} capture_labels={capture_labels} metric={metric:?} kernel={kernel:?}",
            recipe.batch_size, recipe.alpha0, recipe.epoch_rows, recipe.total_steps
        );
        let mut centroids = init.to_vec();
        if metric == Metric::InnerProduct {
            spherical::normalize_rows(&mut centroids, d);
        }
        let mut trainer = Trainer {
            d,
            k,
            kernel,
            metric,
            usage: vec![INIT_USAGE; k],
            snapshot: centroids.clone(),
            centroids,
            batch: BatchAccumulator::new(k, d),
            carry: Vec::with_capacity(BLOCK_ROWS.min(recipe.batch_size) * d),
            block_labels: Vec::new(),
            block_scores: Vec::new(),
            pass_counts: vec![0; k],
            pass_sums: vec![0.0; k * d],
            pass_objective: 0.0,
            pass_rows: 0,
            pass_splits: 0,
            pass_labels: None,
            capture_labels,
            epoch_stats: Vec::with_capacity(recipe.epoch_rows.len()),
            step: 0,
            epoch: 0,
            poisoned: false,
            split_rng: Stream::new(seed, StreamId::Revival),
            recipe,
        };
        trainer.refresh_readout();
        trainer.start_pass();
        trainer
    }

    /// Rows per logical batch, B. The last batch of a pass can be shorter.
    pub fn batch_rows(&self) -> usize {
        self.recipe.batch_size
    }

    /// The resolved recipe of this run.
    pub fn recipe(&self) -> &Recipe {
        &self.recipe
    }

    /// The current centroids, k*d row-major: the EMA state, or its unit-norm readout in spherical mode
    /// (`REF/e_means.py:243-258`). Read them again after each `update`, `update_with` or `end_epoch`.
    /// Use them to compute the `BatchStats` that `update_with` applies.
    pub fn current_centroids(&self) -> Result<&[f32]> {
        if self.poisoned {
            return Err(Error::Poisoned);
        }
        Ok(match self.metric {
            Metric::InnerProduct => &self.snapshot,
            Metric::L2 => &self.centroids,
        })
    }

    /// Assign a chunk of the current logical batch and add it to the batch statistics.
    ///
    /// `rows` is row-major with `d` columns. A batch may arrive in any number of chunks; the result is the
    /// same. Every error leaves the trainer unchanged.
    pub fn accumulate(&mut self, rows: &[f32]) -> Result<()> {
        trace!(
            "Trainer::accumulate: values={} step={} epoch={}",
            rows.len(),
            self.step,
            self.epoch
        );
        self.require_running()?;
        if !rows.len().is_multiple_of(self.d) {
            return Err(Error::DimensionMismatch(format!(
                "chunk of {} values is not a whole number of rows of d={}",
                rows.len(),
                self.d
            )));
        }
        let due = self.require_batch_due()?;
        let chunk = rows.len() / self.d;
        let remaining = due - self.received();
        if chunk > remaining {
            return Err(Error::BatchOverflow { chunk, remaining });
        }
        if let Some(at) = rows.iter().position(|v| !v.is_finite()) {
            return Err(Error::NonFiniteInput(format!(
                "chunk row {} column {} is {}",
                at / self.d,
                at % self.d,
                rows[at]
            )));
        }
        if chunk == 0 {
            return Ok(());
        }
        if self.received() == 0 && self.metric == Metric::L2 {
            self.snapshot.copy_from_slice(&self.centroids);
        }
        let d = self.d;
        let mut carry = std::mem::take(&mut self.carry);
        let mut rest = rows;
        // Batch row where the next block starts; blocks are cut relative to the batch start.
        let mut cursor = self.batch.assigned;
        let mut carry_complete = false;
        if !carry.is_empty() {
            let block_len = BLOCK_ROWS.min(due - cursor);
            let take = (block_len - carry.len() / d).min(rest.len() / d);
            let (head, tail) = rest.split_at(take * d);
            carry.extend_from_slice(head);
            rest = tail;
            if carry.len() / d == block_len {
                carry_complete = true;
                cursor += block_len;
            }
        }
        let mut blocks: Vec<&[f32]> = Vec::new();
        if carry_complete {
            blocks.push(&carry);
        }
        while cursor < due {
            let block_len = BLOCK_ROWS.min(due - cursor);
            if rest.len() / d < block_len {
                break;
            }
            let (head, tail) = rest.split_at(block_len * d);
            blocks.push(head);
            rest = tail;
            cursor += block_len;
        }
        self.assign_blocks(&blocks);
        if carry_complete {
            carry.clear();
        }
        carry.extend_from_slice(rest);
        self.carry = carry;
        Ok(())
    }

    /// Apply the EMA for the complete current batch, then advance the step.
    ///
    /// `Error::BatchIncomplete` before every row of the batch has arrived; the trainer is unchanged.
    pub fn update(&mut self) -> Result<()> {
        trace!("Trainer::update: step={} epoch={}", self.step, self.epoch);
        self.require_running()?;
        let due = self.require_batch_due()?;
        let have = self.received();
        if have < due {
            return Err(Error::BatchIncomplete { have, want: due });
        }
        let counts = std::mem::take(&mut self.batch.counts);
        let sums = std::mem::take(&mut self.batch.sums);
        let applied = self.apply(&counts, &sums, self.batch.objective);
        self.batch.counts = counts;
        self.batch.sums = sums;
        applied?;
        self.finish_step(due)
    }

    /// Apply statistics computed elsewhere for the current batch (`REF/base.py:249-255`).
    ///
    /// Checked first: `sums` is `k * d`, `counts` is `k`, the counts add up to the rows the batch expects,
    /// every value is finite, and no rows were accumulated for this batch. A last-pass batch applied here
    /// has no labels, so the run's labels become unavailable.
    pub fn update_with(&mut self, stats: BatchStats<'_>) -> Result<()> {
        trace!(
            "Trainer::update_with: step={} epoch={} objective={}",
            self.step, self.epoch, stats.objective
        );
        self.require_running()?;
        let due = self.require_batch_due()?;
        self.check_statistics(&stats, due)?;
        self.apply(stats.counts, stats.sums, stats.objective)?;
        if self.pass_labels.take().is_some() {
            debug!("Trainer::update_with: last-pass labels dropped, {due} rows have no label");
        }
        self.finish_step(due)
    }

    /// Close the current pass and record its statistics.
    ///
    /// `Error::EpochIncomplete` until `update` has applied every row of the pass; the trainer is unchanged.
    pub fn end_epoch(&mut self) -> Result<()> {
        debug!("Trainer::end_epoch: epoch={} step={}", self.epoch, self.step);
        self.require_running()?;
        let rows = self.recipe.epoch_rows[self.epoch];
        if self.pass_rows < rows {
            return Err(Error::EpochIncomplete(format!(
                "pass {} of {} has {} of {rows} rows applied",
                self.epoch + 1,
                self.passes(),
                self.pass_rows
            )));
        }
        let count_square: f64 = self.pass_counts.iter().map(|&c| (c as f64) * (c as f64)).sum();
        let stats = EpochStats {
            objective: self.pass_objective,
            imbalance_factor: self.k as f64 * count_square / (rows as f64 * rows as f64),
            splits: self.pass_splits,
            rows,
        };
        debug!(
            "Trainer::end_epoch: pass {} of {}: objective={} imbalance_factor={} splits={} rows={}",
            self.epoch + 1,
            self.passes(),
            stats.objective,
            stats.imbalance_factor,
            stats.splits,
            stats.rows
        );
        self.epoch_stats.push(stats);
        self.epoch += 1;
        if self.epoch < self.passes() {
            self.start_pass();
        }
        Ok(())
    }

    /// Replace each centroid assigned in the last pass by the mean of its rows, and return the model.
    ///
    /// Centroids with no rows in the last pass keep their EMA position (`REF/e_means.py:531-564`). In
    /// spherical mode every centroid is then projected to unit norm in f32, a zero one staying zero
    /// (`REF/e_means.py:512-529`).
    pub fn finalize(self) -> Result<Model> {
        debug!("Trainer::finalize: epoch={} step={}", self.epoch, self.step);
        if self.poisoned {
            return Err(Error::Poisoned);
        }
        if self.epoch < self.passes() {
            return Err(Error::EpochIncomplete(format!(
                "pass {} of {} has not ended",
                self.epoch + 1,
                self.passes()
            )));
        }
        let mut centroids = self.centroids;
        for (j, &count) in self.pass_counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let span = j * self.d..(j + 1) * self.d;
            for (c, &s) in centroids[span.clone()].iter_mut().zip(&self.pass_sums[span]) {
                *c = (s / count as f64) as f32;
            }
        }
        if self.metric == Metric::InnerProduct {
            spherical::normalize_rows(&mut centroids, self.d);
        }
        let labels = self.pass_labels.map(|labels| labels.into_iter().map(Some).collect());
        debug!(
            "Trainer::finalize: {} passes, {} steps",
            self.epoch_stats.len(),
            self.step
        );
        Ok(Model::new(
            self.d,
            self.kernel,
            self.metric,
            centroids,
            self.epoch_stats,
            self.recipe,
            labels,
        ))
    }

    fn passes(&self) -> usize {
        self.recipe.epoch_rows.len()
    }

    fn is_last_pass(&self) -> bool {
        self.epoch + 1 == self.passes()
    }

    /// Rows received for the current batch: assigned plus waiting in `carry`.
    fn received(&self) -> usize {
        self.batch.assigned + self.carry.len() / self.d
    }

    fn require_running(&self) -> Result<()> {
        if self.poisoned {
            return Err(Error::Poisoned);
        }
        if self.epoch >= self.passes() {
            return Err(Error::ScheduleComplete(format!(
                "all {} passes have ended",
                self.passes()
            )));
        }
        Ok(())
    }

    /// Rows of the current batch: `min(B, rows left in the pass)` (`REF/base.py:243-247`).
    fn require_batch_due(&self) -> Result<usize> {
        let rows = self.recipe.epoch_rows[self.epoch];
        let due = self.recipe.batch_size.min(rows - self.pass_rows);
        if due == 0 {
            return Err(Error::ScheduleComplete(format!(
                "pass {} of {} has all {rows} rows applied; call end_epoch",
                self.epoch + 1,
                self.passes()
            )));
        }
        Ok(due)
    }

    fn check_statistics(&self, stats: &BatchStats<'_>, due: usize) -> Result<()> {
        let invalid = |reason: String| Err(Error::InvalidStatistics(reason));
        if stats.sums.len() != self.k * self.d {
            return invalid(format!(
                "sums has {} values, want k*d = {}",
                stats.sums.len(),
                self.k * self.d
            ));
        }
        if stats.counts.len() != self.k {
            return invalid(format!("counts has {} values, want k = {}", stats.counts.len(), self.k));
        }
        let have = self.received();
        if have != 0 {
            return invalid(format!("{have} rows were already accumulated for this batch"));
        }
        let total = stats.counts.iter().try_fold(0u64, |sum, &c| sum.checked_add(c));
        if total != Some(due as u64) {
            return invalid(format!("counts add up to {total:?}, the batch expects {due} rows"));
        }
        if !stats.objective.is_finite() {
            return invalid(format!("objective is {}", stats.objective));
        }
        for (j, &count) in stats.counts.iter().enumerate() {
            let sums = &stats.sums[j * self.d..(j + 1) * self.d];
            if let Some(s) = sums.iter().find(|s| !s.is_finite()) {
                return invalid(format!("sums of cluster {j} hold {s}"));
            }
            if count == 0 {
                if let Some(s) = sums.iter().find(|&&s| s != 0.0) {
                    return invalid(format!("cluster {j} has no rows but a sum of {s}"));
                }
            } else if let Some(s) = sums.iter().find(|&&s| !((s / count as f64) as f32).is_finite()) {
                return invalid(format!("mean of cluster {j} overflows f32: sum {s} over {count} rows"));
            }
        }
        Ok(())
    }

    /// Assign complete blocks against the snapshot, one block per task on the current `rayon` pool, then add
    /// them to the batch in row order on this thread.
    ///
    /// A block's labels and scores depend only on its rows and the snapshot, and the counts, sums and
    /// objective are added sequentially in row order, so neither the thread count nor how the rows arrived
    /// changes a bit of the result.
    fn assign_blocks(&mut self, blocks: &[&[f32]]) {
        if blocks.is_empty() {
            return;
        }
        let d = self.d;
        let total: usize = blocks.iter().map(|block| block.len() / d).sum();
        trace!("Trainer::assign_blocks: blocks={} rows={total}", blocks.len());
        let mut labels = std::mem::take(&mut self.block_labels);
        let mut scores = std::mem::take(&mut self.block_scores);
        labels.resize(total, 0);
        scores.resize(total, 0.0);

        let mut jobs = Vec::with_capacity(blocks.len());
        let (mut free_labels, mut free_scores) = (&mut labels[..], &mut scores[..]);
        for &block in blocks {
            let rows = block.len() / d;
            let (block_labels, rest_labels) = std::mem::take(&mut free_labels).split_at_mut(rows);
            let (block_scores, rest_scores) = std::mem::take(&mut free_scores).split_at_mut(rows);
            jobs.push((block, block_labels, block_scores));
            free_labels = rest_labels;
            free_scores = rest_scores;
        }
        let (kernel, metric, snapshot) = (self.kernel, self.metric, &self.snapshot);
        let norms = assign::centroid_norms(kernel, metric, snapshot, d);
        jobs.into_par_iter().for_each(|(block, block_labels, block_scores)| {
            assign::nearest(kernel, metric, block, snapshot, &norms, d, block_labels, block_scores);
        });

        let mut at = 0;
        for &block in blocks {
            let rows = block.len() / d;
            if self.metric == Metric::InnerProduct {
                // `2 - 2 * ip` per row, as `2 * rows` per block plus the f32 values `-2 * ip`
                // (`REF/e_means.py:432-434`).
                self.batch.objective += 2.0 * rows as f64;
            }
            for ((row, &label), &score) in block
                .chunks_exact(d)
                .zip(&labels[at..at + rows])
                .zip(&scores[at..at + rows])
            {
                let j = label as usize;
                self.batch.counts[j] += 1;
                for (s, &x) in self.batch.sums[j * d..(j + 1) * d].iter_mut().zip(row) {
                    *s += f64::from(x);
                }
                self.batch.objective += match self.metric {
                    Metric::L2 => f64::from(score),
                    Metric::InnerProduct => f64::from(-2.0 * score),
                };
            }
            at += rows;
        }
        if let Some(pass_labels) = self.pass_labels.as_mut() {
            pass_labels.extend_from_slice(&labels);
        }
        self.batch.assigned += total;
        self.block_labels = labels;
        self.block_scores = scores;
    }

    /// Algorithm 1 line 8: the EMA of masses and centroids (`REF/e_means.py:456-505`).
    ///
    /// alpha and gamma are computed in f64 and rounded to f32 once; masses, weights and moves are f32.
    fn apply(&mut self, counts: &[u64], sums: &[f64], objective: f64) -> Result<()> {
        let alpha = scheduled_alpha(self.recipe.alpha0, self.step, self.recipe.total_steps);
        let replace_usage = alpha == 1.0;
        let alpha32 = alpha as f32;
        let gamma32 = (1.0 - alpha) as f32;
        self.pass_objective += objective;
        for (pass, &batch) in self.pass_counts.iter_mut().zip(counts) {
            *pass += batch;
        }
        if self.is_last_pass() {
            for (pass, &batch) in self.pass_sums.iter_mut().zip(sums) {
                *pass += batch;
            }
        }
        if !replace_usage {
            for usage in &mut self.usage {
                *usage *= gamma32;
            }
        }
        for (j, &count) in counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let retained = if replace_usage { 0.0 } else { self.usage[j] };
            let mass = retained + alpha32 * count as f32;
            let weight = alpha32 * count as f32 / mass;
            let span = j * self.d..(j + 1) * self.d;
            for (c, &s) in self.centroids[span.clone()].iter_mut().zip(&sums[span]) {
                let mean = (s / count as f64) as f32;
                *c = if weight == 1.0 { mean } else { *c + weight * (mean - *c) };
            }
            self.usage[j] = mass;
            if let Some(at) = self.centroids[j * self.d..(j + 1) * self.d]
                .iter()
                .position(|c| !c.is_finite())
            {
                self.poisoned = true;
                return Err(Error::NonFiniteState(format!(
                    "centroid {j} coordinate {at} left f32 range at step {}",
                    self.step
                )));
            }
        }
        Ok(())
    }

    /// Revival every `CHECK_UNUSED_EVERY` updates, then advance the step and clear the batch (`REF/base.py:277-286`).
    ///
    /// A split that leaves f32 range poisons the trainer.
    fn finish_step(&mut self, due: usize) -> Result<()> {
        if (self.step + 1).is_multiple_of(CHECK_UNUSED_EVERY) {
            match self.revive() {
                Ok(splits) => self.pass_splits += splits,
                Err(err) => {
                    self.poisoned = true;
                    return Err(err);
                }
            }
        }
        self.refresh_readout();
        self.step += 1;
        self.pass_rows += due;
        self.batch.clear();
        Ok(())
    }

    /// Spherical mode: project the EMA state into the unit-norm readout the next batch is assigned against
    /// (`REF/e_means.py:217-241`). The EMA state itself stays un-normalized. Euclidean mode freezes the
    /// snapshot at the next batch's first chunk instead.
    fn refresh_readout(&mut self) {
        if self.metric == Metric::InnerProduct {
            spherical::normalize_into(&self.centroids, &mut self.snapshot, self.d);
        }
    }

    /// Split a heavy cluster into each dead one (`REF/e_means.py:261-300`).
    fn revive(&mut self) -> Result<usize> {
        trace!(
            "Trainer::revive: step={} threshold={}",
            self.step, self.recipe.unused_threshold
        );
        revival::revive(
            &mut self.usage,
            &mut self.centroids,
            self.d,
            self.recipe.unused_threshold,
            &mut self.split_rng,
        )
    }

    /// Reset the pass statistics; the last pass also captures labels (`REF/e_means.py:347-356`).
    fn start_pass(&mut self) {
        self.pass_counts.fill(0);
        self.pass_objective = 0.0;
        self.pass_rows = 0;
        self.pass_splits = 0;
        if self.is_last_pass() && self.capture_labels {
            self.pass_labels = Some(Vec::with_capacity(self.recipe.epoch_rows[self.epoch]));
        }
    }
}

#[cfg(test)]
mod tests;
