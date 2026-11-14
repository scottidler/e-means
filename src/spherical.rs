//! Unit-norm projection of centroid rows for spherical mode (`REF/e_means_spherical.py:32-42`).

/// Project each `d`-wide row of `source` onto the unit sphere, into `out`.
///
/// Each row is divided by its max-abs first and by its norm second (squares summed in f64), so rows near
/// `f32::MAX` or near the subnormal range do not overflow or underflow the norm. A zero row stays zero.
pub(crate) fn normalize_into(source: &[f32], out: &mut [f32], d: usize) {
    assert_eq!(source.len(), out.len(), "source and out disagree on the length");
    out.copy_from_slice(source);
    normalize_rows(out, d);
}

/// `normalize_into` in place.
pub(crate) fn normalize_rows(rows: &mut [f32], d: usize) {
    assert!(d > 0 && rows.len().is_multiple_of(d), "rows are not whole rows of d");
    for row in rows.chunks_exact_mut(d) {
        let maximum = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let scale = if maximum > 0.0 { maximum } else { 1.0 };
        for v in row.iter_mut() {
            *v /= scale;
        }
        // A sequential f32 sum of squares drifts ~3e-6 from unit norm at d=1536; f64 keeps it under 1e-6.
        let norm = row
            .iter()
            .fold(0.0f64, |sum, &v| sum + f64::from(v) * f64::from(v))
            .sqrt() as f32;
        let norm = if norm > 0.0 { norm } else { 1.0 };
        for v in row.iter_mut() {
            *v /= norm;
        }
    }
}

#[cfg(test)]
mod tests;
