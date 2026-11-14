//! The training recipe: named constants, the batch and step laws, the pass schedule and the cosine step schedule.
//!
//! `REF` is `contrib/e_means` in Faiss at `d1d9789a`; every constant and law cites its line in `REF/base.py`.

use log::debug;

use crate::error::{Error, Result};

/// Initial EMA mass of every centroid (`REF/base.py:25`).
pub const INIT_USAGE: f32 = 1.0 / 65536.0;
/// Cosine schedule floor, as a fraction of the initial step (`REF/base.py:18`).
pub const ALPHA_MIN_FACTOR: f64 = 0.1;
/// Updates between revival checks (`REF/base.py:21`).
pub const CHECK_UNUSED_EVERY: usize = 5;
/// Relative coordinate perturbation for donor splits (`REF/base.py:28`).
pub const SPLIT_EPSILON: f32 = 1e-4;
/// Default revival threshold tau above `NO_REVIVAL_MAX_D` (`REF/base.py:31`).
pub const UNUSED_THRESHOLD: f32 = 0.1;
/// Revival is off by default at or below this dimension (`REF/base.py:38`).
pub const NO_REVIVAL_MAX_D: usize = 2;
/// `BatchLaw::Auto` picks the quality law at or below this dimension (`REF/base.py:46`).
pub const AUTO_QUALITY_MAX_D: usize = 16;

/// Batch law coefficients `(C_B, b_k, b_n, b_T)` for the throughput law (`REF/base.py:54-57`).
const THROUGHPUT_LAW: BatchCoefficients = BatchCoefficients {
    c: 0.035134,
    k: 0.134749,
    n: 0.876473,
    t: 0.813344,
};
/// Quality uses one eighth of the throughput coefficient (`REF/base.py:54-57`).
const QUALITY_LAW: BatchCoefficients = BatchCoefficients {
    c: 0.00439175,
    k: 0.134749,
    n: 0.876473,
    t: 0.813344,
};

/// Step law coefficients `(C_alpha, a_k, a_B, a_n, a_T)` (`REF/base.py:64`).
const ALPHA_COEFFICIENT: f64 = 0.8244;
const ALPHA_K_EXP: f64 = -0.4617;
const ALPHA_BATCH_EXP: f64 = 0.9003;
const ALPHA_N_EXP: f64 = -0.4498;
const ALPHA_T_EXP: f64 = -0.1057;
/// `step_law` clamps its result to this range (`REF/base.py:76-86`).
const ALPHA_LAW_MIN: f64 = 1e-4;
const ALPHA_LAW_MAX: f64 = 1.0;

struct BatchCoefficients {
    c: f64,
    k: f64,
    n: f64,
    t: f64,
}

/// Which fitted batch law picks the batch size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchLaw {
    /// Quality at `d <= AUTO_QUALITY_MAX_D`, throughput above.
    Auto,
    Throughput,
    Quality,
}

/// Everything `Recipe::resolve` needs; `None` fields take the law or the dimension default.
#[derive(Debug, Clone, PartialEq)]
pub struct RecipeSpec {
    pub n: usize,
    pub k: usize,
    pub d: usize,
    pub epochs: f64,
    pub batch_law: BatchLaw,
    pub batch_size: Option<usize>,
    pub alpha0: Option<f64>,
    pub unused_threshold: Option<f32>,
}

/// The resolved recipe of one run.
#[derive(Debug, Clone, PartialEq)]
pub struct Recipe {
    /// Rows per logical batch, B.
    pub batch_size: usize,
    /// Initial EMA step.
    pub alpha0: f64,
    /// The law that chose B; `None` when the caller gave `batch_size`. Never `Auto`.
    pub batch_law: Option<BatchLaw>,
    /// Revival threshold tau; 0 disables revival.
    pub unused_threshold: f32,
    /// Rows of each pass: whole passes, a fractional prefix, then the closing pass.
    pub epoch_rows: Vec<usize>,
    /// Sum over passes of `ceil(rows_i / B)`; tail batches are kept.
    pub total_steps: usize,
}

/// Round a positive value to the nearest power of two; values at or below 1 give 1 (`REF/base.py:67-73`).
pub fn round_pow2(value: f64) -> usize {
    if value <= 1.0 {
        return 1;
    }
    let low = 2f64.powi(value.log2().floor() as i32);
    let high = 2.0 * low;
    if value / low < high / value { low as usize } else { high as usize }
}

/// Initial EMA step coupled to a batch size by the fitted step law (`REF/base.py:76-86`).
pub fn step_law(n: usize, k: usize, epochs: f64, batch_size: usize) -> f64 {
    let alpha0 = ALPHA_COEFFICIENT
        * (k as f64).powf(ALPHA_K_EXP)
        * (batch_size as f64).powf(ALPHA_BATCH_EXP)
        * (n as f64).powf(ALPHA_N_EXP)
        * epochs.powf(ALPHA_T_EXP);
    alpha0.clamp(ALPHA_LAW_MIN, ALPHA_LAW_MAX)
}

/// Resolve `Auto` to the law it stands for at dimension `d` (`REF/base.py:89-99`).
pub fn resolve_batch_law(law: BatchLaw, d: usize) -> BatchLaw {
    match law {
        BatchLaw::Auto if d <= AUTO_QUALITY_MAX_D => BatchLaw::Quality,
        BatchLaw::Auto => BatchLaw::Throughput,
        explicit => explicit,
    }
}

/// Revival threshold: the caller's value, else 0 at `d <= NO_REVIVAL_MAX_D`, else `UNUSED_THRESHOLD` (`REF/base.py:102-109`).
pub fn resolve_unused_threshold(requested: Option<f32>, d: usize) -> Result<f32> {
    match requested {
        None if d <= NO_REVIVAL_MAX_D => Ok(0.0),
        None => Ok(UNUSED_THRESHOLD),
        Some(tau) if tau >= 0.0 => Ok(tau),
        Some(tau) => Err(Error::InvalidParameter(format!(
            "unused_threshold must be >= 0 (0 disables revival), got {tau}"
        ))),
    }
}

/// Batch size from a fitted law with its caps (`REF/base.py:140-152`). `law` must not be `Auto`.
///
/// The caps differ on purpose: throughput caps at `round_pow2(n / 2)`, quality at `n / 2`.
pub fn batch_from_law(law: BatchLaw, n: usize, k: usize, epochs: f64) -> usize {
    let (coefficients, cap) = match law {
        BatchLaw::Quality => (QUALITY_LAW, n / 2),
        BatchLaw::Throughput | BatchLaw::Auto => (THROUGHPUT_LAW, round_pow2(n as f64 / 2.0)),
    };
    let raw = coefficients.c
        * (k as f64).powf(coefficients.k)
        * (n as f64).powf(coefficients.n)
        * epochs.powf(coefficients.t);
    round_pow2(raw).min(cap).min(n).max(1)
}

/// Rows of each pass: whole passes, a rounded fractional prefix, then the closing pass (`REF/base.py:156-166`).
///
/// The fractional pass sits before the last full pass. Below one pass only the prefix remains.
pub fn epoch_rows(n: usize, epochs: f64) -> Vec<usize> {
    let whole = epochs.floor();
    let remainder = ((epochs - whole) * n as f64).round_ties_even() as usize;
    if whole == 0.0 {
        return vec![remainder];
    }
    let mut rows = vec![n; whole as usize - 1];
    if remainder != 0 {
        rows.push(remainder);
    }
    rows.push(n);
    rows
}

/// Cosine-anneal an EMA step towards `ALPHA_MIN_FACTOR` of its initial value (`REF/base.py:174-178`).
///
/// `step` is 0-based, so step 0 gives exactly `alpha0`; the floor is never reached.
pub fn scheduled_alpha(alpha0: f64, step: usize, total_steps: usize) -> f64 {
    let fraction = step.min(total_steps) as f64 / total_steps.max(1) as f64;
    let cosine = 0.5 * (1.0 + (std::f64::consts::PI * fraction).cos());
    alpha0 * (ALPHA_MIN_FACTOR + (1.0 - ALPHA_MIN_FACTOR) * cosine)
}

impl Recipe {
    /// Resolve the recipe for a run (`REF/base.py:129-152`).
    pub fn resolve(spec: &RecipeSpec) -> Result<Recipe> {
        debug!(
            "Recipe::resolve: n={} k={} d={} epochs={}",
            spec.n, spec.k, spec.d, spec.epochs
        );
        if !spec.epochs.is_finite() || spec.epochs <= 0.0 {
            return Err(Error::InvalidParameter(format!(
                "epochs must be finite and > 0, got {}",
                spec.epochs
            )));
        }
        let epoch_rows = epoch_rows(spec.n, spec.epochs);
        if epoch_rows.iter().sum::<usize>() == 0 {
            return Err(Error::InvalidParameter(format!(
                "epochs {} is too small: no rows in any pass of {} rows",
                spec.epochs, spec.n
            )));
        }
        let (batch_size, batch_law) = match spec.batch_size {
            Some(0) => return Err(Error::InvalidParameter("batch_size must be >= 1".to_string())),
            Some(b) => (b.min(spec.n), None),
            None => {
                let law = resolve_batch_law(spec.batch_law, spec.d);
                (batch_from_law(law, spec.n, spec.k, spec.epochs), Some(law))
            }
        };
        let alpha0 = match spec.alpha0 {
            Some(a) if a > 0.0 && a <= 1.0 => a,
            Some(a) => return Err(Error::InvalidParameter(format!("alpha0 must be in (0, 1], got {a}"))),
            None => step_law(spec.n, spec.k, spec.epochs, batch_size),
        };
        let unused_threshold = resolve_unused_threshold(spec.unused_threshold, spec.d)?;
        let total_steps = epoch_rows.iter().map(|rows| rows.div_ceil(batch_size)).sum();
        Ok(Recipe {
            batch_size,
            alpha0,
            batch_law,
            unused_threshold,
            epoch_rows,
            total_steps,
        })
    }
}

#[cfg(test)]
mod tests;
