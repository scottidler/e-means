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

/// Lanes of the argmin scan: a row's scores are scanned as 16 interleaved streams, which the compiler
/// vectorizes on the baseline target.
const LANES: usize = 16;

/// Cap on a task's assignment scratch, in f32: 1 MB whatever k is. The Gemm kernel tiles the centroids so
/// a `rows x tile width` block of inner products fits under it. 2^18 against 2^16 at k=65536, 64 threads:
/// 0.624 vs 0.630 s train (median of 7), 155 vs 101 MB peak; the faster at 64 threads wins.
pub(crate) const TILE_FLOATS: usize = 1 << 18;

/// The centroid squared norms `nearest` takes: `||c||^2` per centroid under `Kernel::Gemm` and `Metric::L2`,
/// empty otherwise. Compute it once per top-level call and share it across that call's tasks.
pub(crate) fn centroid_norms(kernel: Kernel, metric: Metric, centroids: &[f32], d: usize) -> Vec<f32> {
    match (kernel, metric) {
        (Kernel::Gemm, Metric::L2) => centroids.chunks_exact(d).map(|c| inner_product(c, c)).collect(),
        _ => Vec::new(),
    }
}

/// Label each row of `rows` with its nearest centroid under `metric` and record its score.
///
/// `rows` is `labels.len()` rows of `d` coordinates; `centroids` is row-major with `d` columns. The score
/// is the squared distance under `Metric::L2` and the inner product under `Metric::InnerProduct`. `norms`
/// is `centroid_norms(kernel, metric, centroids, d)`, hoisted so a call computes it once, not per block.
/// `scratch` is the calling task's buffer, reused across calls; it holds at most `max(TILE_FLOATS, rows)` f32
/// under `Kernel::Gemm` and `max(TILE_FLOATS, LANES * d)` under `Kernel::Direct`, whatever k is.
#[allow(clippy::too_many_arguments)] // the design's signature: inputs, outputs and the task's scratch
pub(crate) fn nearest(
    kernel: Kernel,
    metric: Metric,
    rows: &[f32],
    centroids: &[f32],
    norms: &[f32],
    d: usize,
    labels: &mut [u32],
    scores: &mut [f32],
    scratch: &mut Vec<f32>,
) {
    nearest_capped(
        kernel,
        metric,
        rows,
        centroids,
        norms,
        d,
        labels,
        scores,
        scratch,
        TILE_FLOATS,
    );
}

/// `nearest` with the scratch cap as a parameter, so tests can force many small tiles.
#[allow(clippy::too_many_arguments)]
fn nearest_capped(
    kernel: Kernel,
    metric: Metric,
    rows: &[f32],
    centroids: &[f32],
    norms: &[f32],
    d: usize,
    labels: &mut [u32],
    scores: &mut [f32],
    scratch: &mut Vec<f32>,
    tile_floats: usize,
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
    if kernel == Kernel::Gemm && metric == Metric::L2 {
        assert_eq!(norms.len(), centroids.len() / d, "norms are not one per centroid");
    }
    match (kernel, metric) {
        (Kernel::Direct, Metric::L2) => direct_lanes(
            rows,
            centroids,
            d,
            labels,
            scores,
            scratch,
            tile_floats,
            f32::INFINITY,
            |sum, x, c| {
                let diff = x - c;
                sum + diff * diff
            },
            squared_l2,
            |v, best| v < best,
        ),
        (Kernel::Direct, Metric::InnerProduct) => direct_lanes(
            rows,
            centroids,
            d,
            labels,
            scores,
            scratch,
            tile_floats,
            f32::NEG_INFINITY,
            |sum, x, c| sum + x * c,
            inner_product,
            |v, best| v > best,
        ),
        (Kernel::Gemm, metric) => gemm_nearest(metric, rows, centroids, norms, d, labels, scores, scratch, tile_floats),
    }
}

/// The Gemm kernel over one block, tiled over the centroids: each tile of `width = (tile_floats / rows)`
/// centroids (at least 1, at most k) gets a `rows x width` block of inner products in `scratch`, then a lane
/// scan per row, merged into the row's running best.
///
/// Under `Metric::L2` the score is `||x||^2 + ||c||^2 - 2 x.c`, clamped at 0, in Faiss's order
/// (`exhaustive_L2sqr_blas`); under `Metric::InnerProduct` it is the inner product. Each sgemm element
/// depends only on its row, its centroid and d (matrixmultiply 0.3.11, pinned), never on the tile, so every
/// score equals the untiled one. Tiles run in ascending centroid order and a later tile wins only on the
/// strict order: the first index at the row's best score, as one scan over all k finds.
#[allow(clippy::too_many_arguments)]
fn gemm_nearest(
    metric: Metric,
    rows: &[f32],
    centroids: &[f32],
    norms: &[f32],
    d: usize,
    labels: &mut [u32],
    scores: &mut [f32],
    scratch: &mut Vec<f32>,
    tile_floats: usize,
) {
    let m = labels.len();
    if m == 0 {
        return;
    }
    let k = centroids.len() / d;
    let width = (tile_floats / m).clamp(1, k);
    let worst = match metric {
        Metric::L2 => f32::INFINITY,
        Metric::InnerProduct => f32::NEG_INFINITY,
    };
    labels.fill(0);
    scores.fill(worst);
    let row_norms: Vec<f32> = match metric {
        Metric::L2 => rows.chunks_exact(d).map(|row| inner_product(row, row)).collect(),
        Metric::InnerProduct => Vec::new(),
    };
    scratch.resize(m * width, 0.0);
    for j0 in (0..k).step_by(width) {
        let j1 = (j0 + width).min(k);
        let tile_width = j1 - j0;
        let tile = &mut scratch[..m * tile_width];
        inner_products(rows, &centroids[j0 * d..j1 * d], d, tile);
        let per_row = tile
            .chunks_exact_mut(tile_width)
            .zip(labels.iter_mut().zip(scores.iter_mut()));
        match metric {
            Metric::L2 => {
                for (i, (products, (label, distance))) in per_row.enumerate() {
                    let (id, best) = lane_scan_l2(products, &norms[j0..j1], row_norms[i]);
                    if best < *distance {
                        *label = j0 as u32 + id;
                        *distance = best;
                    }
                }
            }
            Metric::InnerProduct => {
                for (products, (label, product)) in per_row {
                    let (id, best) = lane_scan(products, worst, |v, best| v > best);
                    if best > *product {
                        *label = j0 as u32 + id;
                        *product = best;
                    }
                }
            }
        }
    }
}

/// `lane_scan` over the L2 scores `row_norm + norms[j] - 2 * products[j]`, clamped at 0, computed in the
/// scan instead of stored first: the same per-element value, the same comparisons.
fn lane_scan_l2(products: &[f32], norms: &[f32], row_norm: f32) -> (u32, f32) {
    #[inline(always)]
    fn expanded(row_norm: f32, centroid_norm: f32, product: f32) -> f32 {
        let mut e = row_norm + centroid_norm - 2.0 * product;
        if e < 0.0 {
            e = 0.0;
        }
        e
    }
    let worst = f32::INFINITY;
    let mut lane_best = [worst; LANES];
    let mut lane_id = [0u32; LANES];
    let (groups, tail) = products.as_chunks::<LANES>();
    let (norm_groups, norm_tail) = norms.as_chunks::<LANES>();
    let mut ids: [u32; LANES] = std::array::from_fn(|l| l as u32);
    for (group, norm_group) in groups.iter().zip(norm_groups) {
        let mut vals = [0.0f32; LANES];
        for l in 0..LANES {
            let e = row_norm + norm_group[l] - 2.0 * group[l];
            vals[l] = if e < 0.0 { 0.0 } else { e };
        }
        for l in 0..LANES {
            let v = vals[l];
            let better = v < lane_best[l];
            lane_best[l] = if better { v } else { lane_best[l] };
            lane_id[l] = if better { ids[l] } else { lane_id[l] };
            ids[l] += LANES as u32;
        }
    }
    let base = (groups.len() * LANES) as u32;
    let (mut best, mut best_id) = (worst, 0u32);
    for l in 0..LANES {
        let v = lane_best[l];
        if v < best || (v == best && v != worst && lane_id[l] < best_id) {
            best = v;
            best_id = lane_id[l];
        }
    }
    for (offset, (&p, &cn)) in tail.iter().zip(norm_tail).enumerate() {
        let v = expanded(row_norm, cn, p);
        if v < best {
            best = v;
            best_id = base + offset as u32;
        }
    }
    (best_id, best)
}

/// The first index of the best score in `scores`, and that score, where `better(v, best)` is the strict
/// order (`<` for the smallest, `>` for the largest) and `worst` is its identity (`+inf`, `-inf`).
///
/// Lane `l` scans ids `l, l + LANES, ..` with the strict order, so it keeps its lowest id at its best.
/// Lanes merge in lane order, an equal score going to the lower id; the tail past the last full group has
/// ids above every lane's, so it takes the strict order alone. That is the first index at the best score,
/// as a sequential strict scan finds. A NaN loses every comparison, and a row of all `worst` takes id 0
/// with score `worst`, on both paths (the `v != worst` guard).
fn lane_scan(scores: &[f32], worst: f32, better: impl Fn(f32, f32) -> bool + Copy) -> (u32, f32) {
    let mut lane_best = [worst; LANES];
    let mut lane_id = [0u32; LANES];
    let (groups, tail) = scores.as_chunks::<LANES>();
    let mut base = 0u32;
    for group in groups.iter() {
        for l in 0..LANES {
            if better(group[l], lane_best[l]) {
                lane_best[l] = group[l];
                lane_id[l] = base + l as u32;
            }
        }
        base += LANES as u32;
    }
    let (mut best, mut best_id) = (worst, 0u32);
    for l in 0..LANES {
        let v = lane_best[l];
        if better(v, best) || (v == best && v != worst && lane_id[l] < best_id) {
            best = v;
            best_id = lane_id[l];
        }
    }
    for (offset, &v) in tail.iter().enumerate() {
        if better(v, best) {
            best = v;
            best_id = base + offset as u32;
        }
    }
    (best_id, best)
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

/// The Direct kernel over one block: each row's score against every centroid by today's scalar fold, 16
/// centroids at a time.
///
/// The centroids in full groups of `LANES` are tiled, `width = LANES * max(1, tile_floats / (LANES * d))`
/// centroids per tile, at most the full groups' count. Each tile is transposed into `scratch` so coordinate
/// `c` of centroid `j0 + g * LANES + l` sits at `g * LANES * d + c * LANES + l`; lane `l` of a group then folds
/// `fold(acc, x[c], centroid[c])` from 0.0, coordinate 0 first: the exact ops of `squared_l2` or
/// `inner_product` against that one centroid. A row's lanes keep their best under the strict order and merge
/// as `lane_scan` merges them, so a tile yields the first index at its best score; tiles run in ascending
/// order and a later tile wins only on the strict order. The `k mod LANES` centroids past the last full
/// group go through `scalar` after every tile, ids above every tile's. Together: the first index at the
/// row's best score, as one sequential strict scan finds.
///
/// `scratch` grows to at most `max(tile_floats, LANES * d)` f32 whatever k is.
#[allow(clippy::too_many_arguments)]
fn direct_lanes(
    rows: &[f32],
    centroids: &[f32],
    d: usize,
    labels: &mut [u32],
    scores: &mut [f32],
    scratch: &mut Vec<f32>,
    tile_floats: usize,
    worst: f32,
    fold: impl Fn(f32, f32, f32) -> f32 + Copy,
    scalar: impl Fn(&[f32], &[f32]) -> f32,
    better: impl Fn(f32, f32) -> bool + Copy,
) {
    if labels.is_empty() {
        return;
    }
    let k = centroids.len() / d;
    let grouped = k - k % LANES;
    labels.fill(0);
    scores.fill(worst);
    if grouped > 0 {
        let width = (LANES * (tile_floats / (LANES * d)).max(1)).min(grouped);
        let layout_floats = width * d;
        if scratch.len() < layout_floats {
            scratch.reserve_exact(layout_floats - scratch.len());
            scratch.resize(layout_floats, 0.0);
        }
        for j0 in (0..grouped).step_by(width) {
            let j1 = (j0 + width).min(grouped);
            let layout = &mut scratch[..(j1 - j0) * d];
            transpose_into_lanes(&centroids[j0 * d..j1 * d], d, layout);
            for ((row, label), score) in rows.chunks_exact(d).zip(labels.iter_mut()).zip(scores.iter_mut()) {
                let (id, best) = lane_groups_best(row, layout, d, worst, fold, better);
                if better(best, *score) {
                    *label = j0 as u32 + id;
                    *score = best;
                }
            }
        }
    }
    for ((row, label), score) in rows.chunks_exact(d).zip(labels.iter_mut()).zip(scores.iter_mut()) {
        for (offset, centroid) in centroids[grouped * d..].chunks_exact(d).enumerate() {
            let v = scalar(row, centroid);
            if better(v, *score) {
                *label = (grouped + offset) as u32;
                *score = v;
            }
        }
    }
}

/// Rewrite `centroids` (whole groups of `LANES`, row-major with `d` columns) into lane layout: coordinate `c`
/// of centroid `g * LANES + l` goes to `layout[g * LANES * d + c * LANES + l]`.
fn transpose_into_lanes(centroids: &[f32], d: usize, layout: &mut [f32]) {
    assert_eq!(centroids.len(), layout.len(), "layout is not the tile's size");
    assert!(
        centroids.len().is_multiple_of(LANES * d),
        "centroids are not whole groups of LANES"
    );
    for (group, lanes) in centroids
        .chunks_exact(LANES * d)
        .zip(layout.chunks_exact_mut(LANES * d))
    {
        for (l, centroid) in group.chunks_exact(d).enumerate() {
            for (c, &value) in centroid.iter().enumerate() {
                lanes[c * LANES + l] = value;
            }
        }
    }
}

/// One row against every group of a lane-layout tile: the first tile-local index at the best score, and
/// that score. Lane `l` keeps its lowest id at its best with the strict order; lanes merge in lane order, an
/// equal score going to the lower id, never at `worst` (so a row of all `worst` takes id 0, as a sequential
/// scan does).
fn lane_groups_best(
    row: &[f32],
    layout: &[f32],
    d: usize,
    worst: f32,
    fold: impl Fn(f32, f32, f32) -> f32,
    better: impl Fn(f32, f32) -> bool,
) -> (u32, f32) {
    let mut lane_best = [worst; LANES];
    let mut lane_id = [0u32; LANES];
    let mut base = 0u32;
    for group in layout.chunks_exact(LANES * d) {
        let (coordinates, _) = group.as_chunks::<LANES>();
        let mut acc = [0.0f32; LANES];
        for (&x, lane) in row.iter().zip(coordinates) {
            for l in 0..LANES {
                acc[l] = fold(acc[l], x, lane[l]);
            }
        }
        for l in 0..LANES {
            if better(acc[l], lane_best[l]) {
                lane_best[l] = acc[l];
                lane_id[l] = base + l as u32;
            }
        }
        base += LANES as u32;
    }
    let (mut best, mut best_id) = (worst, 0u32);
    for l in 0..LANES {
        let v = lane_best[l];
        if better(v, best) || (v == best && v != worst && lane_id[l] < best_id) {
            best = v;
            best_id = lane_id[l];
        }
    }
    (best_id, best)
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
