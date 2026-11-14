//! Revival of unused clusters: each dead cluster is re-seeded as a split of a heavy one.
//!
//! `REF` is `contrib/e_means` in Faiss at `d1d9789a`: threshold `REF/e_means.py:41-45`, donor choice
//! `:261-281`, split `:283-300`, separation `:89-104`.

use log::debug;

use crate::error::{Error, Result};
use crate::recipe::SPLIT_EPSILON;
use crate::rng::Stream;

/// Mass below which a cluster is dead: `f32(sum_f32(usage)) * f32(tau) / f32(k)`, in that order.
///
/// The sum is sequential in f32 (deviation D8); NumPy's pairwise sum can differ in the last bits for k > 8.
pub(crate) fn threshold(usage: &[f32], tau: f32, k: usize) -> f32 {
    let total: f32 = usage.iter().fold(0.0f32, |sum, &u| sum + u);
    total * tau / k as f32
}

/// Split one donor into each dead cluster, up to the number of donors; returns the splits made.
///
/// Dead is `usage < threshold` (strict); a donor is alive with positive mass. With no pair, no random
/// number is drawn. Children are built in scratch first: if any coordinate is not finite nothing is
/// committed and the result is `Error::NonFiniteState`.
pub(crate) fn revive(usage: &mut [f32], centroids: &mut [f32], d: usize, tau: f32, rng: &mut Stream) -> Result<usize> {
    let k = usage.len();
    let threshold = threshold(usage, tau, k);
    let dead: Vec<usize> = (0..k).filter(|&j| usage[j] < threshold).collect();
    let donors = usage.iter().filter(|&&u| is_donor(u, threshold)).count();
    let nsplit = dead.len().min(donors);
    if nsplit == 0 {
        return Ok(0);
    }
    let sources = choose_donors(usage, threshold, nsplit, rng);
    let targets = &dead[..nsplit];
    debug!("revival::revive: threshold={threshold} splits={nsplit} sources={sources:?} targets={targets:?}");

    let mut donor_rows = Vec::with_capacity(nsplit * d);
    let mut child_rows = Vec::with_capacity(nsplit * d);
    for &source in &sources {
        let row = &centroids[source * d..(source + 1) * d];
        let donor_start = donor_rows.len();
        for (i, &c) in row.iter().enumerate() {
            let offset = if i % 2 == 0 { SPLIT_EPSILON } else { -SPLIT_EPSILON };
            donor_rows.push(c * (1.0 - offset));
            child_rows.push(c * (1.0 + offset));
        }
        separate(&donor_rows[donor_start..], &mut child_rows[donor_start..]);
    }
    if let Some(at) = child_rows.iter().chain(&donor_rows).position(|v| !v.is_finite()) {
        let at = at % (nsplit * d);
        let source = sources[at / d];
        return Err(Error::NonFiniteState(format!(
            "split of centroid {source} into {} leaves f32 range at coordinate {}",
            targets[at / d],
            at % d
        )));
    }

    for (i, (&source, &target)) in sources.iter().zip(targets).enumerate() {
        let half = usage[source] / 2.0;
        usage[source] = half;
        usage[target] = half;
        let rows = i * d..(i + 1) * d;
        centroids[source * d..(source + 1) * d].copy_from_slice(&donor_rows[rows.clone()]);
        centroids[target * d..(target + 1) * d].copy_from_slice(&child_rows[rows]);
    }
    Ok(nsplit)
}

/// The `nsplit` donors with the largest Gumbel keys `ln(usage) - ln(-ln(u))`, largest first.
///
/// One uniform is drawn per cluster, donor or not, floored at `f64::MIN_POSITIVE`. The log of the mass is
/// taken in f32 and widened, as the reference does; the Gumbel term is f64. Sampling the top keys draws
/// donors without replacement in proportion to their mass.
fn choose_donors(usage: &[f32], threshold: f32, nsplit: usize, rng: &mut Stream) -> Vec<usize> {
    let mut keyed: Vec<(f64, usize)> = Vec::with_capacity(usage.len());
    for (j, &u) in usage.iter().enumerate() {
        let uniform = rng.uniform().max(f64::MIN_POSITIVE);
        if is_donor(u, threshold) {
            keyed.push((f64::from(u.ln()) - (-uniform.ln()).ln(), j));
        }
    }
    keyed.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1.cmp(&a.1)));
    keyed.into_iter().take(nsplit).map(|(_, j)| j).collect()
}

/// Alive (not under the threshold) with mass to give.
fn is_donor(usage: f32, threshold: f32) -> bool {
    usage >= threshold && usage > 0.0
}

/// When a split leaves donor and child equal on every coordinate, move the child's coordinate 0 one ULP
/// toward +inf, or toward -inf if that overflows (`REF/e_means.py:89-104`).
pub(crate) fn separate(donor: &[f32], child: &mut [f32]) {
    if donor != child {
        return;
    }
    let up = child[0].next_up();
    child[0] = if up.is_finite() { up } else { child[0].next_down() };
}

#[cfg(test)]
mod tests;
