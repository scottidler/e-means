//! Nearest-centroid search.
//!
//! Only the direct kernel exists so far: squared L2 by difference, in f32, ties to the lowest index.

/// The assignment kernel of a run, fixed at `begin`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kernel {
    /// `sum((x - c)^2)` per centroid, f32, coordinates summed in order.
    Direct,
}

/// Label each row of `rows` with its nearest centroid and record the squared distance.
///
/// `rows` is `labels.len()` rows of `d` coordinates; `centroids` is row-major with `d` columns.
pub(crate) fn nearest(
    kernel: Kernel,
    rows: &[f32],
    centroids: &[f32],
    d: usize,
    labels: &mut [u32],
    distances: &mut [f32],
) {
    assert_eq!(
        rows.len(),
        labels.len() * d,
        "rows and labels disagree on the row count"
    );
    assert_eq!(
        labels.len(),
        distances.len(),
        "labels and distances disagree on the row count"
    );
    assert!(
        !centroids.is_empty() && centroids.len().is_multiple_of(d),
        "centroids are not k rows of d"
    );
    match kernel {
        Kernel::Direct => direct(rows, centroids, d, labels, distances),
    }
}

fn direct(rows: &[f32], centroids: &[f32], d: usize, labels: &mut [u32], distances: &mut [f32]) {
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

fn squared_l2(row: &[f32], centroid: &[f32]) -> f32 {
    row.iter().zip(centroid).fold(0.0f32, |sum, (x, c)| {
        let diff = x - c;
        sum + diff * diff
    })
}

#[cfg(test)]
mod tests;
