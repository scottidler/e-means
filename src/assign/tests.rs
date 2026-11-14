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
        &[],
        d,
        &mut labels,
        &mut distances,
        &mut Vec::new(),
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
        &[],
        1,
        &mut labels,
        &mut distances,
        &mut Vec::new(),
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
        &[],
        d,
        &mut labels,
        &mut products,
        &mut Vec::new(),
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
        &[],
        1,
        &mut labels,
        &mut products,
        &mut Vec::new(),
    );
}

fn gemm_nearest_of(metric: Metric, rows: &[f32], centroids: &[f32], d: usize) -> (Vec<u32>, Vec<f32>) {
    let n = rows.len() / d;
    let mut labels = vec![u32::MAX; n];
    let mut scores = vec![f32::NAN; n];
    let norms = centroid_norms(Kernel::Gemm, metric, centroids, d);
    nearest(
        Kernel::Gemm,
        metric,
        rows,
        centroids,
        &norms,
        d,
        &mut labels,
        &mut scores,
        &mut Vec::new(),
    );
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

// Oracles: today's scalar kernels, copied verbatim. Later phases replace the production scans with lane
// versions; every one is compared with `to_bits` against these. They are never edited to follow the code.

fn oracle_direct_l2(rows: &[f32], centroids: &[f32], d: usize, labels: &mut [u32], distances: &mut [f32]) {
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

fn oracle_direct_inner_product(rows: &[f32], centroids: &[f32], d: usize, labels: &mut [u32], products: &mut [f32]) {
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

fn oracle_gemm_nearest(
    metric: Metric,
    rows: &[f32],
    centroids: &[f32],
    d: usize,
    labels: &mut [u32],
    scores: &mut [f32],
) {
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

/// The oracle for `(kernel, metric)`, labels and `to_bits` of the scores.
fn oracle_nearest(kernel: Kernel, metric: Metric, rows: &[f32], centroids: &[f32], d: usize) -> (Vec<u32>, Vec<u32>) {
    let n = rows.len() / d;
    let mut labels = vec![u32::MAX; n];
    let mut scores = vec![f32::NAN; n];
    match (kernel, metric) {
        (Kernel::Direct, Metric::L2) => oracle_direct_l2(rows, centroids, d, &mut labels, &mut scores),
        (Kernel::Direct, Metric::InnerProduct) => {
            oracle_direct_inner_product(rows, centroids, d, &mut labels, &mut scores)
        }
        (Kernel::Gemm, metric) => oracle_gemm_nearest(metric, rows, centroids, d, &mut labels, &mut scores),
    }
    (labels, scores.iter().map(|s| s.to_bits()).collect())
}

/// `nearest`, labels and `to_bits` of the scores.
fn production_nearest(
    kernel: Kernel,
    metric: Metric,
    rows: &[f32],
    centroids: &[f32],
    d: usize,
) -> (Vec<u32>, Vec<u32>) {
    let n = rows.len() / d;
    let mut labels = vec![u32::MAX; n];
    let mut scores = vec![f32::NAN; n];
    let norms = centroid_norms(kernel, metric, centroids, d);
    nearest(
        kernel,
        metric,
        rows,
        centroids,
        &norms,
        d,
        &mut labels,
        &mut scores,
        &mut Vec::new(),
    );
    (labels, scores.iter().map(|s| s.to_bits()).collect())
}

const KERNELS: [Kernel; 2] = [Kernel::Direct, Kernel::Gemm];
const METRICS: [Metric; 2] = [Metric::L2, Metric::InnerProduct];

/// Every (kernel, metric): `nearest` equals its oracle bitwise, and the oracle's labels are `want`.
fn assert_matches_oracle(rows: &[f32], centroids: &[f32], d: usize, want: &[u32], what: &str) {
    for kernel in KERNELS {
        for metric in METRICS {
            let oracle = oracle_nearest(kernel, metric, rows, centroids, d);
            assert_eq!(oracle.0, want, "{what}: oracle labels, {kernel:?} {metric:?}");
            assert_eq!(
                production_nearest(kernel, metric, rows, centroids, d),
                oracle,
                "{what}: nearest differs from its oracle, {kernel:?} {metric:?}"
            );
        }
    }
}

#[test]
fn nearest_equals_the_oracles_bitwise_on_random_data() {
    let n = 60;
    for d in [1, 3, 8, 17, 32] {
        for k in [1, 2, 15, 16, 17, 31, 33, 100] {
            let rows = lcg_values(n * d, (d * 1000 + k) as u64);
            let centroids = lcg_values(k * d, (d * 1000 + k + 7) as u64);
            for kernel in KERNELS {
                for metric in METRICS {
                    assert_eq!(
                        production_nearest(kernel, metric, &rows, &centroids, d),
                        oracle_nearest(kernel, metric, &rows, &centroids, d),
                        "d={d} k={k} {kernel:?} {metric:?}"
                    );
                }
            }
        }
    }
}

/// k centroids on a line (d = 2): every centroid is the far default `(-5 - j, 0)` except those in `tied`,
/// which sit at `(3, 0)`. Row `(1, 0)`: under L2 the tied centroids are the unique nearest, all at squared
/// distance 4; under inner product they have the unique largest product, 3. Small integers, so every kernel
/// computes every score exactly.
fn tie_fixture(k: usize, tied: &[usize]) -> (Vec<f32>, Vec<f32>) {
    let mut centroids = Vec::with_capacity(k * 2);
    for j in 0..k {
        let x = if tied.contains(&j) { 3.0 } else { -5.0 - j as f32 };
        centroids.extend([x, 0.0]);
    }
    (vec![1.0, 0.0], centroids)
}

#[test]
fn tie_between_two_centroids_in_one_lane_goes_to_the_lower() {
    // LANES = 16: ids 3 and 19 are both lane 3; ids 20 and 4 likewise (given in descending order).
    let (row, centroids) = tie_fixture(32, &[3, 19]);
    assert_matches_oracle(&row, &centroids, 2, &[3], "ids 3, 19");
    let (row, centroids) = tie_fixture(40, &[20, 4]);
    assert_matches_oracle(&row, &centroids, 2, &[4], "ids 4, 20");
}

#[test]
fn tie_between_two_lanes_goes_to_the_lower() {
    let (row, centroids) = tie_fixture(32, &[5, 9]);
    assert_matches_oracle(&row, &centroids, 2, &[5], "ids 5, 9");
    let (row, centroids) = tie_fixture(32, &[30, 2]);
    assert_matches_oracle(&row, &centroids, 2, &[2], "ids 2, 30");
}

#[test]
fn tie_in_the_tail_goes_to_the_lower() {
    // k = 21: ids 16..=20 are past the last full group of 16.
    let (row, centroids) = tie_fixture(21, &[17, 19]);
    assert_matches_oracle(&row, &centroids, 2, &[17], "tail ids 17, 19");
    let (row, centroids) = tie_fixture(21, &[18, 4]);
    assert_matches_oracle(&row, &centroids, 2, &[4], "lane id 4 beats tail id 18");
    let (row, centroids) = tie_fixture(21, &[20]);
    assert_matches_oracle(&row, &centroids, 2, &[20], "a lone tail minimum wins");
}

#[test]
fn a_row_equal_to_every_centroid_takes_index_zero_with_a_positive_zero_score() {
    // Gemm L2: `||x||^2 + ||c||^2 - 2 x.c` is exactly 0 for every centroid, the all-zero clamp case.
    for k in [1, 16, 21, 33] {
        let rows = [0.0f32; 3];
        let centroids = vec![0.0f32; k * 3];
        for kernel in KERNELS {
            for metric in METRICS {
                let (labels, scores) = production_nearest(kernel, metric, &rows, &centroids, 3);
                assert_eq!(labels, [0], "k={k} {kernel:?} {metric:?}");
                assert_eq!(
                    scores,
                    [0.0f32.to_bits()],
                    "k={k} {kernel:?} {metric:?}: +0.0, never -0.0"
                );
                assert_eq!(
                    (labels, scores),
                    oracle_nearest(kernel, metric, &rows, &centroids, 3),
                    "k={k} {kernel:?} {metric:?}"
                );
            }
        }
    }
}

#[test]
fn a_clamped_negative_expansion_scores_positive_zero_and_ties_to_the_lowest_index() {
    // Rows equal to centroids: the Gemm expansion can land a few ulps below 0 and is clamped to +0.0.
    let d = 17;
    let centroids = lcg_values(40 * d, 9);
    let rows = centroids.clone();
    let (labels, scores) = production_nearest(Kernel::Gemm, Metric::L2, &rows, &centroids, d);
    assert!(
        scores
            .iter()
            .all(|&s| f32::from_bits(s) >= 0.0 && !f32::from_bits(s).is_sign_negative()),
        "{scores:?}"
    );
    assert_eq!(
        (labels, scores),
        oracle_nearest(Kernel::Gemm, Metric::L2, &rows, &centroids, d)
    );
}

#[test]
fn an_all_infinite_score_row_takes_index_zero_and_keeps_the_infinity() {
    // 1e30 squared overflows f32: L2 scores are +inf for both kernels. Inner product: every product is -inf.
    for k in [1, 16, 21] {
        let rows = [1e30f32];
        let near = vec![0.0f32; k];
        for kernel in KERNELS {
            let (labels, scores) = production_nearest(kernel, Metric::L2, &rows, &near, 1);
            assert_eq!(labels, [0], "L2 k={k} {kernel:?}");
            assert_eq!(scores, [f32::INFINITY.to_bits()], "L2 k={k} {kernel:?}");
            assert_eq!(
                (labels, scores),
                oracle_nearest(kernel, Metric::L2, &rows, &near, 1),
                "L2 k={k} {kernel:?}"
            );
            let far = vec![-1e30f32; k];
            let (labels, scores) = production_nearest(kernel, Metric::InnerProduct, &rows, &far, 1);
            assert_eq!(labels, [0], "IP k={k} {kernel:?}");
            assert_eq!(scores, [f32::NEG_INFINITY.to_bits()], "IP k={k} {kernel:?}");
            assert_eq!(
                (labels, scores),
                oracle_nearest(kernel, Metric::InnerProduct, &rows, &far, 1),
                "IP k={k} {kernel:?}"
            );
        }
    }
}

/// The sequential strict scan `lane_scan` replaces: first index at the best score.
fn sequential_scan(scores: &[f32], worst: f32, better: impl Fn(f32, f32) -> bool) -> (u32, f32) {
    let (mut best, mut best_id) = (worst, 0u32);
    for (j, &v) in scores.iter().enumerate() {
        if better(v, best) {
            best = v;
            best_id = j as u32;
        }
    }
    (best_id, best)
}

#[test]
fn lane_scan_equals_a_sequential_scan_bitwise_on_tie_heavy_data() {
    for len in (0..=70).chain([255, 256, 257]) {
        for seed in 0..20u64 {
            // Four distinct values, so equal minima fall in the same lane, across lanes and in the tail.
            let mut scores: Vec<f32> = lcg_values(len, seed * 31 + len as u64)
                .iter()
                .map(|v| (v * 2.0).floor())
                .collect();
            if seed % 5 == 1 && len > 3 {
                scores[len / 2] = f32::NAN;
            }
            if seed % 5 == 2 {
                scores.iter_mut().for_each(|v| *v = f32::INFINITY);
            }
            if seed % 5 == 3 {
                scores.iter_mut().for_each(|v| *v = f32::NEG_INFINITY);
            }
            let min = lane_scan(&scores, f32::INFINITY, |v, b| v < b);
            let want = sequential_scan(&scores, f32::INFINITY, |v, b| v < b);
            assert_eq!(
                (min.0, min.1.to_bits()),
                (want.0, want.1.to_bits()),
                "min len={len} seed={seed}"
            );
            let max = lane_scan(&scores, f32::NEG_INFINITY, |v, b| v > b);
            let want = sequential_scan(&scores, f32::NEG_INFINITY, |v, b| v > b);
            assert_eq!(
                (max.0, max.1.to_bits()),
                (want.0, want.1.to_bits()),
                "max len={len} seed={seed}"
            );
        }
    }
}

#[test]
fn lane_scan_of_nothing_is_id_zero_at_the_identity() {
    assert_eq!(lane_scan(&[], f32::INFINITY, |v, b| v < b), (0, f32::INFINITY));
    assert_eq!(lane_scan(&[f32::NAN; 20], f32::INFINITY, |v, b| v < b).0, 0);
}

#[test]
fn centroid_norms_exist_only_for_gemm_l2() {
    let centroids = [1.0f32, 2.0, 3.0, 4.0];
    assert_eq!(centroid_norms(Kernel::Gemm, Metric::L2, &centroids, 2), [5.0, 25.0]);
    assert!(centroid_norms(Kernel::Gemm, Metric::InnerProduct, &centroids, 2).is_empty());
    assert!(centroid_norms(Kernel::Direct, Metric::L2, &centroids, 2).is_empty());
    assert!(centroid_norms(Kernel::Direct, Metric::InnerProduct, &centroids, 2).is_empty());
}

#[test]
#[should_panic(expected = "norms are not one per centroid")]
fn gemm_l2_rejects_missing_norms() {
    let (mut labels, mut scores) = ([0u32], [0.0f32]);
    nearest(
        Kernel::Gemm,
        Metric::L2,
        &[1.0, 0.0],
        &[1.0, 0.0, 0.0, 1.0],
        &[],
        2,
        &mut labels,
        &mut scores,
        &mut Vec::new(),
    );
}

/// `nearest_capped` with a forced tile cap and a caller's (possibly dirty) scratch, labels and `to_bits`.
fn tiled_nearest(
    kernel: Kernel,
    metric: Metric,
    rows: &[f32],
    centroids: &[f32],
    d: usize,
    tile_floats: usize,
    scratch: &mut Vec<f32>,
) -> (Vec<u32>, Vec<u32>) {
    let n = rows.len() / d;
    let mut labels = vec![u32::MAX; n];
    let mut scores = vec![f32::NAN; n];
    let norms = centroid_norms(kernel, metric, centroids, d);
    nearest_capped(
        kernel,
        metric,
        rows,
        centroids,
        &norms,
        d,
        &mut labels,
        &mut scores,
        scratch,
        tile_floats,
    );
    (labels, scores.iter().map(|s| s.to_bits()).collect())
}

#[test]
fn tiled_gemm_equals_the_oracle_bitwise_with_five_or_more_tiles() {
    // One scratch for every call: a tile must not depend on what an earlier, larger call left in it.
    let mut scratch = vec![f32::NAN; 3];
    for d in [1, 3, 8, 17, 32] {
        for k in [17, 33, 100, 256] {
            for n in [1, 7, 60] {
                let rows = lcg_values(n * d, (d * 7000 + k * 10 + n) as u64);
                let centroids = lcg_values(k * d, (d * 7000 + k * 10 + n + 3) as u64);
                // Widths that leave 5 or more tiles; a cap below n rows (0) still makes progress at width 1.
                for width in [1, 2, 3, 15, 16, 17, k / 5].into_iter().filter(|&w| k.div_ceil(w) >= 5) {
                    for tile_floats in [n * width, n * width + n - 1] {
                        for metric in METRICS {
                            assert_eq!(
                                tiled_nearest(Kernel::Gemm, metric, &rows, &centroids, d, tile_floats, &mut scratch),
                                oracle_nearest(Kernel::Gemm, metric, &rows, &centroids, d),
                                "d={d} k={k} n={n} width={width} cap={tile_floats} {metric:?}"
                            );
                        }
                    }
                }
                for metric in METRICS {
                    assert_eq!(
                        tiled_nearest(Kernel::Gemm, metric, &rows, &centroids, d, 0, &mut scratch),
                        oracle_nearest(Kernel::Gemm, metric, &rows, &centroids, d),
                        "d={d} k={k} n={n} cap=0 {metric:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn tiled_tie_straddling_a_tile_boundary_goes_to_the_lower_as_in_the_oracle() {
    // k = 40 at width 8 (one row, cap 8): tiles [0, 8), [8, 16), .., [32, 40), five of them.
    for (tied, want) in [
        (&[7, 8][..], 7),
        (&[8, 7][..], 7),
        (&[15, 16, 39][..], 15),
        (&[31, 32][..], 31),
        (&[23, 24, 25][..], 23),
    ] {
        let (row, centroids) = tie_fixture(40, tied);
        for metric in METRICS {
            let oracle = oracle_nearest(Kernel::Gemm, metric, &row, &centroids, 2);
            assert_eq!(oracle.0, [want], "{tied:?} {metric:?}: oracle");
            for tile_floats in [1, 8, 16] {
                assert_eq!(
                    tiled_nearest(Kernel::Gemm, metric, &row, &centroids, 2, tile_floats, &mut Vec::new()),
                    oracle,
                    "{tied:?} {metric:?} cap={tile_floats}"
                );
            }
        }
    }
}

#[test]
fn tiled_all_infinite_and_nan_rows_match_the_oracle() {
    // Every tile's best is the identity (or a NaN that loses every comparison): label 0 keeps the identity.
    let k = 21;
    let mut scratch = Vec::new();
    for tile_floats in [1, 4, 5] {
        let near = vec![0.0f32; k];
        let far = vec![-1e30f32; k];
        for (metric, centroids) in [(Metric::L2, &near), (Metric::InnerProduct, &far)] {
            for rows in [[1e30f32], [f32::NAN]] {
                assert_eq!(
                    tiled_nearest(Kernel::Gemm, metric, &rows, centroids, 1, tile_floats, &mut scratch),
                    oracle_nearest(Kernel::Gemm, metric, &rows, centroids, 1),
                    "{metric:?} rows={rows:?} cap={tile_floats}"
                );
            }
        }
    }
}

#[test]
fn gemm_scratch_stays_under_the_tile_cap_whatever_k_is() {
    let d = 4;
    let rows = lcg_values(256 * d, 1);
    let centroids = lcg_values(4096 * d, 2);
    let norms = centroid_norms(Kernel::Gemm, Metric::L2, &centroids, d);
    let (mut labels, mut scores) = (vec![0u32; 256], vec![0.0f32; 256]);
    let mut scratch = Vec::new();
    nearest(
        Kernel::Gemm,
        Metric::L2,
        &rows,
        &centroids,
        &norms,
        d,
        &mut labels,
        &mut scores,
        &mut scratch,
    );
    assert!(
        scratch.capacity() <= TILE_FLOATS,
        "scratch holds {} f32, cap {TILE_FLOATS}",
        scratch.capacity()
    );
    // More rows than the cap: one centroid per tile, `rows` f32.
    let mut big = Vec::new();
    let many = TILE_FLOATS + 3;
    let (mut labels, mut scores) = (vec![0u32; many], vec![0.0f32; many]);
    nearest(
        Kernel::Gemm,
        Metric::InnerProduct,
        &vec![0.5; many],
        &[1.0, -1.0],
        &[],
        1,
        &mut labels,
        &mut scores,
        &mut big,
    );
    assert_eq!(big.capacity(), many);
    assert!(labels.iter().all(|&l| l == 0));
}

#[test]
fn gemm_element_is_independent_of_tile_and_split() {
    // matrixmultiply 0.3.11 accumulates each C element in one chain over the depth: element (i, j) depends on
    // row i, centroid j, d and the CPU, never on the block's shape or position. Tiling rests on it.
    let n = 64 + 7;
    for d in [1, 8, 32, 300] {
        for k in [17, 256] {
            let rows = lcg_values(n * d, (d * 31 + k) as u64);
            let centroids = lcg_values(k * d, (d * 31 + k + 1) as u64);
            let mut whole = vec![0.0f32; n * k];
            inner_products(&rows, &centroids, d, &mut whole);
            for width in [1, 15, 16, 17] {
                for split in [1, 7, 64] {
                    for (block, row_chunk) in rows.chunks(split * d).enumerate() {
                        let m = row_chunk.len() / d;
                        for j0 in (0..k).step_by(width) {
                            let j1 = (j0 + width).min(k);
                            let mut tile = vec![f32::NAN; m * (j1 - j0)];
                            inner_products(row_chunk, &centroids[j0 * d..j1 * d], d, &mut tile);
                            for (r, products) in tile.chunks_exact(j1 - j0).enumerate() {
                                let i = block * split + r;
                                for (c, product) in products.iter().enumerate() {
                                    assert_eq!(
                                        product.to_bits(),
                                        whole[i * k + j0 + c].to_bits(),
                                        "d={d} k={k} width={width} split={split} element ({i}, {})",
                                        j0 + c
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Values from `lcg_values` rounded to multiples of 0.5 in `[-1, 1)`: four distinct coordinates, so many
/// rows sit at exactly equal distance from several centroids.
fn quantized_values(len: usize, seed: u64) -> Vec<f32> {
    lcg_values(len, seed).iter().map(|v| (v * 2.0).floor() / 2.0).collect()
}

#[test]
fn lane_direct_equals_the_oracle_bitwise_on_random_and_quantized_data() {
    let n = 40;
    let mut scratch = vec![f32::NAN; 5];
    for d in [1, 2, 3, 8, 16, 33] {
        for k in [1, 15, 16, 17, 33, 256] {
            let seed = (d * 1000 + k) as u64;
            let data = [
                ("random", lcg_values(n * d, seed), lcg_values(k * d, seed + 7)),
                (
                    "quantized",
                    quantized_values(n * d, seed),
                    quantized_values(k * d, seed + 7),
                ),
            ];
            for (what, rows, centroids) in &data {
                for metric in METRICS {
                    let oracle = oracle_nearest(Kernel::Direct, metric, rows, centroids, d);
                    assert_eq!(
                        production_nearest(Kernel::Direct, metric, rows, centroids, d),
                        oracle,
                        "{what} d={d} k={k} {metric:?}"
                    );
                    assert_eq!(
                        tiled_nearest(Kernel::Direct, metric, rows, centroids, d, TILE_FLOATS, &mut scratch),
                        oracle,
                        "{what} d={d} k={k} {metric:?}, reused scratch"
                    );
                }
            }
        }
    }
}

#[test]
fn tiled_lane_direct_equals_the_oracle_bitwise_with_five_or_more_tiles() {
    // One dirty scratch for every call: a tile must not depend on what an earlier call left in it.
    let mut scratch = vec![f32::NAN; 3];
    for d in [1, 2, 3, 8, 16, 33] {
        for k in [80, 100, 256] {
            let seed = (d * 9000 + k) as u64;
            let data = [
                ("random", lcg_values(37 * d, seed), lcg_values(k * d, seed + 1)),
                (
                    "quantized",
                    quantized_values(37 * d, seed),
                    quantized_values(k * d, seed + 1),
                ),
            ];
            let groups = k / LANES;
            // Tile widths of 1, 2 and 3 groups that leave 5 or more tiles; a cap of 0 still takes one group.
            for (what, rows, centroids) in &data {
                for tile_groups in [1, 2, 3].into_iter().filter(|&g| groups.div_ceil(g) >= 5) {
                    for tile_floats in [tile_groups * LANES * d, (tile_groups + 1) * LANES * d - 1, 0] {
                        for metric in METRICS {
                            assert_eq!(
                                tiled_nearest(Kernel::Direct, metric, rows, centroids, d, tile_floats, &mut scratch),
                                oracle_nearest(Kernel::Direct, metric, rows, centroids, d),
                                "{what} d={d} k={k} cap={tile_floats} {metric:?}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn tiled_lane_direct_tie_straddling_a_tile_boundary_goes_to_the_lower_as_in_the_oracle() {
    // k = 100, d = 2, cap 2 * LANES * 2 at most: tiles of 16 or 32 centroids over [0, 96), tail 96..100.
    for (tied, want) in [
        (&[15, 16][..], 15),
        (&[16, 15][..], 15),
        (&[31, 32, 99][..], 31),
        (&[63, 64, 65][..], 63),
        (&[80, 96][..], 80),
        (&[97, 95][..], 95),
        (&[98, 97][..], 97),
    ] {
        let (row, centroids) = tie_fixture(100, tied);
        for metric in METRICS {
            let oracle = oracle_nearest(Kernel::Direct, metric, &row, &centroids, 2);
            assert_eq!(oracle.0, [want], "{tied:?} {metric:?}: oracle");
            for tile_floats in [0, LANES * 2, 2 * LANES * 2, TILE_FLOATS] {
                assert_eq!(
                    tiled_nearest(
                        Kernel::Direct,
                        metric,
                        &row,
                        &centroids,
                        2,
                        tile_floats,
                        &mut Vec::new()
                    ),
                    oracle,
                    "{tied:?} {metric:?} cap={tile_floats}"
                );
            }
        }
    }
}

#[test]
fn tiled_lane_direct_all_infinite_and_nan_rows_match_the_oracle() {
    let k = 53;
    let mut scratch = Vec::new();
    for tile_floats in [0, LANES, TILE_FLOATS] {
        let near = vec![0.0f32; k];
        let far = vec![-1e30f32; k];
        for (metric, centroids) in [(Metric::L2, &near), (Metric::InnerProduct, &far)] {
            for rows in [[1e30f32], [f32::NAN]] {
                assert_eq!(
                    tiled_nearest(Kernel::Direct, metric, &rows, centroids, 1, tile_floats, &mut scratch),
                    oracle_nearest(Kernel::Direct, metric, &rows, centroids, 1),
                    "{metric:?} rows={rows:?} cap={tile_floats}"
                );
            }
        }
    }
}

#[test]
fn direct_scratch_stays_under_the_tile_cap_at_d64_k65536_and_matches_the_oracle() {
    let (d, k, n) = (64, 65536, 2);
    let rows = lcg_values(n * d, 11);
    let centroids = lcg_values(k * d, 12);
    let bound = TILE_FLOATS.max(LANES * d);
    for metric in METRICS {
        let mut scratch = Vec::new();
        let (mut labels, mut scores) = (vec![u32::MAX; n], vec![f32::NAN; n]);
        nearest(
            Kernel::Direct,
            metric,
            &rows,
            &centroids,
            &[],
            d,
            &mut labels,
            &mut scores,
            &mut scratch,
        );
        assert!(
            scratch.capacity() <= bound,
            "{metric:?}: scratch holds {} f32, bound {bound}",
            scratch.capacity()
        );
        let (want_labels, want_scores) = oracle_nearest(Kernel::Direct, metric, &rows, &centroids, d);
        assert_eq!(labels, want_labels, "{metric:?}");
        assert_eq!(
            scores.iter().map(|s| s.to_bits()).collect::<Vec<_>>(),
            want_scores,
            "{metric:?}"
        );
    }
    // d past TILE_FLOATS / LANES: one group per tile, `LANES * d` f32.
    let d = TILE_FLOATS / LANES + 1;
    let centroids = lcg_values(2 * LANES * d, 13);
    let rows = lcg_values(d, 14);
    let mut scratch = Vec::new();
    let (mut labels, mut scores) = ([0u32], [0.0f32]);
    nearest(
        Kernel::Direct,
        Metric::L2,
        &rows,
        &centroids,
        &[],
        d,
        &mut labels,
        &mut scores,
        &mut scratch,
    );
    assert_eq!(scratch.capacity(), LANES * d);
    assert_eq!(
        (labels.to_vec(), vec![scores[0].to_bits()]),
        oracle_nearest(Kernel::Direct, Metric::L2, &rows, &centroids, d)
    );
}

#[test]
fn transpose_into_lanes_puts_coordinate_c_of_lane_l_at_c_times_lanes_plus_l() {
    let d = 3;
    let centroids: Vec<f32> = (0..2 * LANES * d).map(|v| v as f32).collect();
    let mut layout = vec![f32::NAN; centroids.len()];
    transpose_into_lanes(&centroids, d, &mut layout);
    for g in 0..2 {
        for l in 0..LANES {
            for c in 0..d {
                assert_eq!(
                    layout[g * LANES * d + c * LANES + l],
                    centroids[(g * LANES + l) * d + c]
                );
            }
        }
    }
}

#[test]
#[should_panic(expected = "not whole groups of LANES")]
fn transpose_into_lanes_rejects_a_partial_group() {
    let mut layout = vec![0.0f32; 2 * (LANES - 1)];
    transpose_into_lanes(&[0.0f32; 2 * (LANES - 1)], 2, &mut layout);
}
