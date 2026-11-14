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

// The common-offset case of the design's Kernel choice section: the direct kernel gets it right.
#[test]
fn direct_has_no_common_offset_cancellation() {
    let (labels, distances) = direct_nearest(&[10000.0], &[10002.0, 10001.0], 1);
    assert_eq!(labels, [1]);
    assert_eq!(distances, [1.0]);
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
