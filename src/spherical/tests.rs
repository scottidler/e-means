use super::*;
use std::f32::consts::FRAC_1_SQRT_2;

fn assert_close(got: &[f32], want: &[f32], atol: f32, context: &str) {
    assert_eq!(got.len(), want.len(), "{context}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= atol,
            "{context}: value {i} is {g}, want {w} (atol {atol})"
        );
    }
}

// Design criterion: `normalize_into` on [[3e38, -3e38], [1e-38, 0], [0, 0]] -> [[FRAC_1_SQRT_2, -FRAC_1_SQRT_2], [1, 0],
// [0, 0]] at atol 1e-7.
// Faiss tests/e_means_test_utils.py:228-243 `test_spherical_projection_extremes_and_wide_closing`
#[test]
fn projection_extremes_prescale_and_keep_zero() {
    let source = [3e38, -3e38, 1e-38, 0.0, 0.0, 0.0];
    let mut out = [f32::NAN; 6];
    normalize_into(&source, &mut out, 2);
    assert_close(
        &out,
        &[FRAC_1_SQRT_2, -FRAC_1_SQRT_2, 1.0, 0.0, 0.0, 0.0],
        1e-7,
        "tests/e_means_test_utils.py:238",
    );
    assert_eq!(
        source,
        [3e38, -3e38, 1e-38, 0.0, 0.0, 0.0],
        "tests/e_means_test_utils.py:243 source untouched"
    );
    assert_eq!(&out[4..], &[0.0, 0.0], "a zero row stays exactly zero");
}

// Without the max-abs prescale, 3e38^2 overflows to inf and the row collapses to zero.
#[test]
fn prescale_is_what_keeps_huge_rows_finite() {
    let mut row = [f32::MAX, f32::MAX, f32::MAX];
    normalize_rows(&mut row, 3);
    let third = (1.0f32 / 3.0).sqrt();
    assert_close(&row, &[third; 3], 1e-7, "f32::MAX row");
}

#[test]
fn rows_are_independent_and_unit_norm() {
    let mut rows = [3.0, 4.0, 0.0, -2.0, -0.5, 0.0];
    normalize_rows(&mut rows, 2);
    assert_close(&rows, &[0.6, 0.8, 0.0, -1.0, -1.0, 0.0], 1e-7, "3-4-5 row, axis rows");
}

#[test]
#[should_panic(expected = "source and out disagree")]
fn mismatched_lengths_panic() {
    normalize_into(&[1.0, 2.0], &mut [0.0], 1);
}
