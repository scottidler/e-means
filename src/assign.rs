//! Nearest-centroid search.
//!
//! Only the direct kernel exists so far, in f32, ties to the lowest index: smallest squared L2 by difference,
//! or largest inner product in spherical mode (`REF/e_means.py:303-307`).

/// The assignment kernel of a run, fixed at `begin`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kernel {
    /// `sum((x - c)^2)` per centroid, f32, coordinates summed in order.
    Direct,
}

/// What "nearest" means for a run, fixed at `begin`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Metric {
    /// Smallest squared L2 distance; the score is that distance.
    L2,
    /// Largest inner product (spherical mode); the score is that inner product.
    InnerProduct,
}

/// Label each row of `rows` with its nearest centroid under `metric` and record its score.
///
/// `rows` is `labels.len()` rows of `d` coordinates; `centroids` is row-major with `d` columns. The score
/// is the squared distance under `Metric::L2` and the inner product under `Metric::InnerProduct`.
pub(crate) fn nearest(
    kernel: Kernel,
    metric: Metric,
    rows: &[f32],
    centroids: &[f32],
    d: usize,
    labels: &mut [u32],
    scores: &mut [f32],
) {
    assert_eq!(
        rows.len(),
        labels.len() * d,
        "rows and labels disagree on the row count"
    );
    assert_eq!(
        labels.len(),
        scores.len(),
        "labels and scores disagree on the row count"
    );
    assert!(
        !centroids.is_empty() && centroids.len().is_multiple_of(d),
        "centroids are not k rows of d"
    );
    match (kernel, metric) {
        (Kernel::Direct, Metric::L2) => direct_l2(rows, centroids, d, labels, scores),
        (Kernel::Direct, Metric::InnerProduct) => direct_inner_product(rows, centroids, d, labels, scores),
    }
}

fn direct_l2(rows: &[f32], centroids: &[f32], d: usize, labels: &mut [u32], distances: &mut [f32]) {
    for ((row, label), distance) in rows.chunks_exact(d).zip(labels.iter_mut()).zip(distances.iter_mut()) {
        let mut best = 0usize;
        let mut best_distance = f32::INFINITY;
        for (j, centroid) in centroids.chunks_exact(d).enumerate() {
            let squared = squared_l2(row, centroid);
            if squared < best_distance {
                best = j;
                best_distance = squared;
            }
        }
        *label = best as u32;
        *distance = best_distance;
    }
}

/// Largest inner product; a zero row ties every centroid at 0 and takes index 0.
fn direct_inner_product(rows: &[f32], centroids: &[f32], d: usize, labels: &mut [u32], products: &mut [f32]) {
    for ((row, label), product) in rows.chunks_exact(d).zip(labels.iter_mut()).zip(products.iter_mut()) {
        let mut best = 0usize;
        let mut best_product = f32::NEG_INFINITY;
        for (j, centroid) in centroids.chunks_exact(d).enumerate() {
            let ip = inner_product(row, centroid);
            if ip > best_product {
                best = j;
                best_product = ip;
            }
        }
        *label = best as u32;
        *product = best_product;
    }
}

fn inner_product(row: &[f32], centroid: &[f32]) -> f32 {
    row.iter().zip(centroid).fold(0.0f32, |sum, (x, c)| sum + x * c)
}

fn squared_l2(row: &[f32], centroid: &[f32]) -> f32 {
    row.iter().zip(centroid).fold(0.0f32, |sum, (x, c)| {
        let diff = x - c;
        sum + diff * diff
    })
}

#[cfg(test)]
mod tests;
