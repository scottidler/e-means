//! Nearest-centroid search: smallest squared L2, or largest inner product in spherical mode
//! (`REF/e_means.py:303-307`), in f32, ties to the lowest index.
//!
//! Two kernels behind one predicate on the recipe's `B * d`, as Faiss picks its own by `nx * d`
//! (`faiss/utils/distances.cpp:642,662,705`). The direct kernel computes by difference; the Gemm kernel takes
//! every inner product of a block from one `matrixmultiply::sgemm`, the crate's only `unsafe` call.

/// `B * d` at or above which a run assigns with `Kernel::Gemm`: `faiss.cvar.distance_compute_blas_threshold`
/// (observed 2026-11-13).
pub(crate) const BLAS_THRESHOLD: usize = 128_000;

/// The assignment kernel of a run, fixed at `begin`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kernel {
    /// `sum((x - c)^2)` per centroid, f32, coordinates summed in order.
    Direct,
    /// `||x||^2 + ||c||^2 - 2 x.c`, clamped at 0, with `x.c` from `sgemm` (Faiss's BLAS kernel). Loses the
    /// nearest centroid to cancellation on data with a large common offset.
    Gemm,
}

impl Kernel {
    /// The kernel for a run with batch size `batch_size` in `d` dimensions; keyed on the recipe's B, never
    /// on chunk size or thread count, so one run uses one kernel.
    pub(crate) fn for_batch(batch_size: usize, d: usize) -> Kernel {
        if batch_size.saturating_mul(d) < BLAS_THRESHOLD {
            Kernel::Direct
        } else {
            Kernel::Gemm
        }
    }
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
        (Kernel::Gemm, metric) => gemm_nearest(metric, rows, centroids, d, labels, scores),
    }
}

/// The Gemm kernel over one block: a `rows x k` tile of inner products (the kernel's only scratch), then
/// a scan per row. Under `Metric::L2` the score is `||x||^2 + ||c||^2 - 2 x.c`, clamped at 0, in Faiss's
/// order (`exhaustive_L2sqr_blas`); under `Metric::InnerProduct` it is the inner product.
fn gemm_nearest(metric: Metric, rows: &[f32], centroids: &[f32], d: usize, labels: &mut [u32], scores: &mut [f32]) {
    if labels.is_empty() {
        return;
    }
    let k = centroids.len() / d;
    let mut tile = vec![0.0f32; labels.len() * k];
    inner_products(rows, centroids, d, &mut tile);
    let per_row = rows
        .chunks_exact(d)
        .zip(tile.chunks_exact(k))
        .zip(labels.iter_mut().zip(scores.iter_mut()));
    match metric {
        Metric::L2 => {
            let centroid_norms: Vec<f32> = centroids.chunks_exact(d).map(|c| inner_product(c, c)).collect();
            for ((row, products), (label, distance)) in per_row {
                let row_norm = inner_product(row, row);
                let mut best = 0usize;
                let mut best_distance = f32::INFINITY;
                for (j, (&ip, &centroid_norm)) in products.iter().zip(&centroid_norms).enumerate() {
                    let mut expanded = row_norm + centroid_norm - 2.0 * ip;
                    if expanded < 0.0 {
                        expanded = 0.0;
                    }
                    if expanded < best_distance {
                        best = j;
                        best_distance = expanded;
                    }
                }
                *label = best as u32;
                *distance = best_distance;
            }
        }
        Metric::InnerProduct => {
            for ((_, products), (label, product)) in per_row {
                let mut best = 0usize;
                let mut best_product = f32::NEG_INFINITY;
                for (j, &ip) in products.iter().enumerate() {
                    if ip > best_product {
                        best = j;
                        best_product = ip;
                    }
                }
                *label = best as u32;
                *product = best_product;
            }
        }
    }
}

/// `tile[i * k + j] = rows[i] . centroids[j]` for the `m` rows and `k` centroids of `d` columns, both
/// row-major, by one `sgemm`. Panics unless every length agrees and the tile overlaps neither input.
fn inner_products(rows: &[f32], centroids: &[f32], d: usize, tile: &mut [f32]) {
    assert!(d > 0, "d is 0");
    assert!(rows.len().is_multiple_of(d), "rows are not whole rows of d");
    assert!(centroids.len().is_multiple_of(d), "centroids are not whole rows of d");
    let m = rows.len() / d;
    let k = centroids.len() / d;
    let cells = m.checked_mul(k).expect("rows x centroids overflows usize");
    assert_eq!(tile.len(), cells, "tile is not rows x centroids");
    assert!(
        disjoint(rows, tile) && disjoint(centroids, tile),
        "tile overlaps an input"
    );
    if cells == 0 {
        return;
    }
    let row_stride = isize::try_from(d).expect("d overflows isize");
    let tile_stride = isize::try_from(k).expect("k overflows isize");
    // SAFETY: A is `rows`, m x d with row stride d and column stride 1, so its last element is
    // `rows[m * d - 1]`. B is the transpose of `centroids`, d x k with row stride 1 and column stride d, so its
    // last element is `centroids[k * d - 1]`. C is `tile`, m x k with row stride k, no two elements aliased,
    // last element `tile[m * k - 1]`. The asserts above pin all three lengths and that C overlaps neither A
    // nor B, and beta = 0 means `sgemm` never reads C. `&mut tile` excludes every other access for the call.
    unsafe {
        matrixmultiply::sgemm(
            m,
            d,
            k,
            1.0,
            rows.as_ptr(),
            row_stride,
            1,
            centroids.as_ptr(),
            1,
            row_stride,
            0.0,
            tile.as_mut_ptr(),
            tile_stride,
            1,
        );
    }
}

/// Whether two slices share no element; an empty slice shares none.
fn disjoint(a: &[f32], b: &[f32]) -> bool {
    let (a, b) = (a.as_ptr_range(), b.as_ptr_range());
    a.is_empty() || b.is_empty() || a.end <= b.start || b.end <= a.start
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
