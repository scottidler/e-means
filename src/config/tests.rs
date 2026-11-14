#![allow(clippy::unwrap_used)]

use super::*;

fn invalid(builder: &EMeans, n: usize, init: &[f32]) -> String {
    match builder.begin(n, init) {
        Err(Error::InvalidParameter(m)) => m,
        other => panic!("want InvalidParameter, got {other:?}"),
    }
}

#[test]
fn begin_resolves_the_recipe_for_the_run() {
    let trainer = EMeans::new(2, 2)
        .epochs(2.5)
        .batch_size(4)
        .alpha0(0.3)
        .begin(31, &[0.0; 4])
        .unwrap();
    assert_eq!(trainer.batch_rows(), 4);
    assert_eq!(trainer.recipe().alpha0, 0.3);
    assert_eq!(trainer.recipe().epoch_rows, vec![31, 16, 31]);
    assert_eq!(trainer.recipe().total_steps, 8 + 4 + 8);
    assert_eq!(trainer.recipe().unused_threshold, 0.0, "d <= 2 disables revival");
}

#[test]
fn defaults_follow_the_reference_constructor() {
    let builder = EMeans::new(32, 256);
    assert_eq!(builder.epochs, 5.0);
    assert_eq!(builder.batch_law, BatchLaw::Auto);
    let trainer = builder.begin(100_000, &vec![0.0; 32 * 256]).unwrap();
    assert_eq!(trainer.recipe().batch_size, 8_192, "throughput law above d = 16");
    assert_eq!(trainer.recipe().unused_threshold, 0.1);
    let trainer = EMeans::new(32, 256)
        .batch_law(BatchLaw::Quality)
        .unused_threshold(0.0)
        .begin(100_000, &vec![0.0; 32 * 256])
        .unwrap();
    assert_eq!(trainer.recipe().batch_size, 1_024);
    assert_eq!(trainer.recipe().unused_threshold, 0.0);
}

// Edge-case table: d == 0, k == 0, k > u32::MAX, batch_size == 0, bad epochs, alpha0 and threshold
#[test]
fn invalid_parameters_are_rejected_at_begin() {
    assert!(invalid(&EMeans::new(0, 2), 4, &[]).contains("d must be"));
    assert!(invalid(&EMeans::new(1, 0), 4, &[]).contains("k must be"));
    let huge = u32::MAX as usize + 1;
    assert!(invalid(&EMeans::new(1, huge), huge, &[]).contains("does not fit a u32"));
    let init = [0.0, 1.0];
    assert!(invalid(&EMeans::new(1, 2).batch_size(0), 4, &init).contains("batch_size"));
    for epochs in [0.0, -1.0, f64::NAN, f64::INFINITY, 0.01] {
        assert!(invalid(&EMeans::new(1, 2).epochs(epochs), 4, &init).contains("epochs"));
    }
    for alpha0 in [0.0, 1.5, f64::NAN] {
        assert!(invalid(&EMeans::new(1, 2).alpha0(alpha0), 4, &init).contains("alpha0"));
    }
    for tau in [-0.1, f32::NAN] {
        assert!(invalid(&EMeans::new(1, 2).unused_threshold(tau), 4, &init).contains("unused_threshold"));
    }
}

#[test]
fn shape_errors_are_typed() {
    assert_eq!(
        EMeans::new(1, 3).begin(2, &[0.0; 3]).unwrap_err(),
        Error::TooFewRows { rows: 2, k: 3 }
    );
    assert!(matches!(
        EMeans::new(2, 2).begin(4, &[0.0; 3]),
        Err(Error::DimensionMismatch(ref m)) if m.contains("want k*d = 4")
    ));
}

// Design criterion: init [[0], [NaN]] is NonFiniteInput from begin
#[test]
fn non_finite_init_is_rejected_at_begin() {
    let err = EMeans::new(1, 2).begin(4, &[0.0, f32::NAN]).unwrap_err();
    assert!(
        matches!(err, Error::NonFiniteInput(ref m) if m.contains("centroid 1 coordinate 0")),
        "{err}"
    );
    let err = EMeans::new(2, 1).begin(4, &[f32::INFINITY, 0.0]).unwrap_err();
    assert!(matches!(err, Error::NonFiniteInput(_)), "{err}");
}
