use super::*;

fn direct_nearest(rows: &[f32], centroids: &[f32], d: usize) -> (Vec<u32>, Vec<f32>) {
    let n = rows.len() / d;
    let mut labels = vec![u32::MAX; n];
    let mut distances = vec![f32::NAN; n];
    nearest(
        Kernel::Direct,
        Metric::L2,
        rows,
        centroids,
        d,
        &mut labels,
        &mut distances,
    );
    (labels, distances)
}

#[test]
fn direct_picks_the_nearest_with_exact_squared_distance() {
    let centroids = [0.0, 0.0, 10.0, 10.0, -5.0, 1.0];
    let (labels, distances) = direct_nearest(&[1.0, 2.0, 9.0, 12.0, -6.0, 1.0], &centroids, 2);
    assert_eq!(labels, [0, 1, 2]);
    assert_eq!(distances, [5.0, 5.0, 1.0]);
}

#[test]
fn ties_go_to_the_lowest_index() {
    let (labels, distances) = direct_nearest(&[5.0, 0.0], &[7.0, 3.0, 3.0, 7.0], 1);
    assert_eq!(labels, [0, 1], "row 5 ties all four, row 0 ties centroids 1 and 2");
    assert_eq!(distances, [4.0, 9.0]);
    let (labels, _) = direct_nearest(&[1.0], &[2.0, 2.0, 0.0], 1);
    assert_eq!(labels, [0]);
}

// The offset fixture of the design's Kernel choice section: the direct kernel finds centroid 1; the Gemm
// kernel's expansion cancels to `[0, 0]` and ties to centroid 0. This pins the documented limit.
#[test]
fn common_offset_direct_is_right_and_gemm_cancels() {
    let (labels, distances) = direct_nearest(&[10000.0], &[10002.0, 10001.0], 1);
    assert_eq!(labels, [1]);
    assert_eq!(distances, [1.0]);
    let (labels, distances) = gemm_nearest_of(Metric::L2, &[10000.0], &[10002.0, 10001.0], 1);
    assert_eq!(
        labels,
        [0],
        "the documented Gemm limit: wrong centroid under a common offset"
    );
    assert_eq!(distances, [0.0]);
}

#[test]
fn empty_rows_assign_nothing() {
    let (labels, distances) = direct_nearest(&[], &[1.0, 2.0], 2);
    assert!(labels.is_empty() && distances.is_empty());
}

#[test]
#[should_panic(expected = "rows and labels disagree")]
fn mismatched_buffers_panic() {
    let mut labels = [0u32; 2];
    let mut distances = [0f32; 2];
    nearest(
        Kernel::Direct,
        Metric::L2,
        &[1.0, 2.0, 3.0],
        &[0.0],
        1,
        &mut labels,
        &mut distances,
    );
}

fn inner_product_nearest(rows: &[f32], centroids: &[f32], d: usize) -> (Vec<u32>, Vec<f32>) {
    let n = rows.len() / d;
    let mut labels = vec![u32::MAX; n];
    let mut products = vec![f32::NAN; n];
    nearest(
        Kernel::Direct,
        Metric::InnerProduct,
        rows,
        centroids,
        d,
        &mut labels,
        &mut products,
    );
    (labels, products)
}

#[test]
fn inner_product_picks_the_largest_product() {
    let centroids = [1.0, 0.0, 0.0, 1.0, -1.0, 0.0];
    let (labels, products) = inner_product_nearest(&[0.6, 0.8, -3.0, 1.0, 0.5, -0.25], &centroids, 2);
    assert_eq!(labels, [1, 2, 0]);
    assert_eq!(products, [0.8, 3.0, 0.5]);
}

// Edge-case table: a zero row has inner product 0 with every centroid; label is the lowest index among the
// maxima. Faiss tests/e_means_test_utils.py:273-290 `test_spherical_zero_beats_negative_similarity`
#[test]
fn inner_product_zero_row_takes_the_lowest_index_and_zero_beats_negative() {
    let (labels, products) = inner_product_nearest(&[0.0, 0.0], &[1.0, 0.0, 0.0, 1.0], 2);
    assert_eq!(labels, [0]);
    assert_eq!(products, [0.0]);
    let (labels, products) = inner_product_nearest(&[-1.0, 0.0, 0.0, 0.0], &[0.0, 0.0, 1.0, 0.0], 2);
    assert_eq!(labels, [0, 0], "tests/e_means_test_utils.py:289");
    assert_eq!(products, [0.0, 0.0]);
}

#[test]
#[should_panic(expected = "labels and scores disagree")]
fn mismatched_score_buffer_panics() {
    let mut labels = [0u32; 1];
    let mut products = [0f32; 2];
    nearest(
        Kernel::Direct,
        Metric::InnerProduct,
        &[1.0],
        &[0.0],
        1,
        &mut labels,
        &mut products,
    );
}

fn gemm_nearest_of(metric: Metric, rows: &[f32], centroids: &[f32], d: usize) -> (Vec<u32>, Vec<f32>) {
    let n = rows.len() / d;
    let mut labels = vec![u32::MAX; n];
    let mut scores = vec![f32::NAN; n];
    nearest(Kernel::Gemm, metric, rows, centroids, d, &mut labels, &mut scores);
    (labels, scores)
}

/// Deterministic values in `[-1, 1)` from a 64-bit LCG; tests need data, not a seeded RNG contract.
fn lcg_values(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        })
        .collect()
}

fn squared_norm(values: &[f32]) -> f32 {
    values.iter().map(|v| v * v).sum()
}

#[test]
fn kernel_switches_at_the_blas_threshold() {
    assert_eq!(BLAS_THRESHOLD, 128_000);
    assert_eq!(Kernel::for_batch(15_999, 8), Kernel::Direct, "127992 < 128000");
    assert_eq!(
        Kernel::for_batch(16_000, 8),
        Kernel::Gemm,
        "128000 is not below the threshold"
    );
    assert_eq!(Kernel::for_batch(256, 32), Kernel::Direct);
    assert_eq!(Kernel::for_batch(8192, 32), Kernel::Gemm, "the large fixtures: 262144");
    assert_eq!(Kernel::for_batch(usize::MAX, 2), Kernel::Gemm, "the product saturates");
}

// Phase 9 criterion: on random data with no common offset, the kernels agree on every row whose margin
// `d2 - d1` exceeds `1e-4 * (||x||^2 + ||c1||^2)`.
#[test]
fn gemm_matches_direct_labels_beyond_the_margin() {
    let (n, k) = (700, 64);
    for (seed, d) in [(1, 1), (2, 3), (3, 8), (4, 32), (5, 100)] {
        let rows = lcg_values(n * d, seed);
        let centroids = lcg_values(k * d, seed + 100);
        let (direct_labels, direct_distances) = direct_nearest(&rows, &centroids, d);
        let (gemm_labels, gemm_distances) = gemm_nearest_of(Metric::L2, &rows, &centroids, d);
        let mut compared = 0;
        for (i, row) in rows.chunks_exact(d).enumerate() {
            let mut all: Vec<f32> = centroids.chunks_exact(d).map(|c| squared_l2(row, c)).collect();
            all.sort_by(f32::total_cmp);
            let (d1, d2) = (all[0], all[1]);
            assert_eq!(d1, direct_distances[i]);
            let c1 = &centroids[direct_labels[i] as usize * d..][..d];
            let scale = squared_norm(row) + squared_norm(c1);
            assert!(
                (gemm_distances[i] - d1).abs() <= 1e-4 * scale,
                "d={d} row {i}: gemm {} direct {d1}",
                gemm_distances[i]
            );
            if d2 - d1 > 1e-4 * scale {
                compared += 1;
                assert_eq!(gemm_labels[i], direct_labels[i], "d={d} row {i}");
            }
        }
        assert!(
            compared * 4 >= n * 3,
            "d={d}: only {compared} of {n} rows clear the margin"
        );
    }
}

#[test]
fn gemm_matches_direct_inner_product_beyond_the_margin() {
    let (n, k, d) = (500, 48, 32);
    let rows = lcg_values(n * d, 7);
    let centroids = lcg_values(k * d, 8);
    let (direct_labels, direct_products) = inner_product_nearest(&rows, &centroids, d);
    let (gemm_labels, gemm_products) = gemm_nearest_of(Metric::InnerProduct, &rows, &centroids, d);
    let mut compared = 0;
    for (i, row) in rows.chunks_exact(d).enumerate() {
        let mut all: Vec<f32> = centroids.chunks_exact(d).map(|c| inner_product(row, c)).collect();
        all.sort_by(|a, b| b.total_cmp(a));
        let c1 = &centroids[direct_labels[i] as usize * d..][..d];
        let scale = squared_norm(row) + squared_norm(c1);
        assert!((gemm_products[i] - direct_products[i]).abs() <= 1e-4 * scale, "row {i}");
        if all[0] - all[1] > 1e-4 * scale {
            compared += 1;
            assert_eq!(gemm_labels[i], direct_labels[i], "row {i}");
        }
    }
    assert!(compared * 10 >= n * 9, "only {compared} of {n} rows clear the margin");
}

#[test]
fn gemm_is_exact_on_small_integers_and_ties_to_the_lowest_index() {
    let centroids = [0.0, 0.0, 10.0, 10.0, -5.0, 1.0];
    let (labels, distances) = gemm_nearest_of(Metric::L2, &[1.0, 2.0, 9.0, 12.0, -6.0, 1.0], &centroids, 2);
    assert_eq!(labels, [0, 1, 2]);
    assert_eq!(distances, [5.0, 5.0, 1.0]);
    let (labels, distances) = gemm_nearest_of(Metric::L2, &[5.0, 0.0], &[7.0, 3.0, 3.0, 7.0], 1);
    assert_eq!(labels, [0, 1]);
    assert_eq!(distances, [4.0, 9.0]);
}

#[test]
fn gemm_distance_is_never_negative() {
    let d = 17;
    let centroids = lcg_values(40 * d, 9);
    let (labels, distances) = gemm_nearest_of(Metric::L2, &centroids, &centroids, d);
    assert!(distances.iter().all(|&v| v >= 0.0), "{distances:?}");
    assert!(distances.iter().all(|&v| v <= 1e-5), "{distances:?}");
    assert!(labels.iter().enumerate().all(|(i, &l)| l as usize == i), "{labels:?}");
}

#[test]
fn gemm_inner_product_zero_row_takes_the_lowest_index() {
    let (labels, products) = gemm_nearest_of(Metric::InnerProduct, &[0.0, 0.0], &[1.0, 0.0, 0.0, 1.0], 2);
    assert_eq!(labels, [0]);
    assert_eq!(products, [0.0]);
    let (labels, products) = gemm_nearest_of(Metric::InnerProduct, &[0.6, 0.8], &[1.0, 0.0, 0.0, 1.0], 2);
    assert_eq!(labels, [1]);
    assert_eq!(products, [0.8]);
}

#[test]
fn gemm_empty_rows_assign_nothing() {
    let (labels, distances) = gemm_nearest_of(Metric::L2, &[], &[1.0, 2.0], 2);
    assert!(labels.is_empty() && distances.is_empty());
}

#[test]
fn inner_products_fill_the_row_by_centroid_tile() {
    let rows = [1.0, 2.0, 3.0, -1.0, 0.0, 2.0];
    let centroids = [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, -1.0, -1.0, 0.0];
    let mut tile = [f32::NAN; 8];
    inner_products(&rows, &centroids, 3, &mut tile);
    assert_eq!(tile, [1.0, 5.0, 12.0, -3.0, -1.0, 2.0, 2.0, 1.0]);
}

#[test]
#[should_panic(expected = "tile is not rows x centroids")]
fn inner_products_reject_a_short_tile() {
    let mut tile = [0f32; 3];
    inner_products(&[1.0, 2.0], &[1.0, 2.0, 3.0, 4.0], 1, &mut tile);
}

#[test]
#[should_panic(expected = "rows are not whole rows")]
fn inner_products_reject_partial_rows() {
    let mut tile = [0f32; 2];
    inner_products(&[1.0, 2.0, 3.0], &[1.0, 2.0], 2, &mut tile);
}

#[test]
fn disjoint_sees_overlap_adjacency_and_empty_slices() {
    let values = [0f32; 8];
    assert!(!disjoint(&values[0..4], &values[3..8]));
    assert!(!disjoint(&values[2..3], &values[0..8]));
    assert!(disjoint(&values[0..4], &values[4..8]));
    assert!(disjoint(&values[2..2], &values[0..8]));
}
