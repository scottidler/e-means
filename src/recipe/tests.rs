#![allow(clippy::unwrap_used)]

use super::*;

const F: &str = "tests/test_e_means.py";

/// `(batch, alpha0)` of a law-resolved recipe, the Rust analog of Faiss `_resolve_recipe(n, k, T, law, None, None)`.
fn resolve_law(n: usize, k: usize, epochs: f64, law: BatchLaw) -> (usize, f64) {
    let batch = batch_from_law(law, n, k, epochs);
    (batch, step_law(n, k, epochs, batch))
}

fn spec(n: usize, k: usize, d: usize, epochs: f64) -> RecipeSpec {
    RecipeSpec {
        n,
        k,
        d,
        epochs,
        batch_law: BatchLaw::Auto,
        batch_size: None,
        alpha0: None,
        unused_threshold: None,
    }
}

fn assert_close(got: f64, want: f64, tol: f64) {
    assert!((got - want).abs() <= tol, "got {got}, want {want}");
}

// Faiss tests/test_e_means.py:99-106
#[test]
fn fractional_epoch_rounding() {
    assert_eq!(
        epoch_rows(10_000_000_000, 1.9999999995),
        vec![9_999_999_995, 10_000_000_000],
        "{F}:99"
    );
    assert_eq!(epoch_rows(3, 1.5), vec![2, 3]);
    assert_eq!(epoch_rows(5, 1.5), vec![2, 5]);
    assert_eq!(epoch_rows(3, 1.01), vec![3]);
}

// Design criteria: pass layout (REF/base.py:156-166)
#[test]
fn epoch_rows_places_fraction_before_closing_pass() {
    assert_eq!(epoch_rows(31, 2.5), vec![31, 16, 31]);
    assert_eq!(epoch_rows(31, 0.5), vec![16]);
    assert_eq!(epoch_rows(31, 3.0), vec![31, 31, 31]);
}

// Faiss tests/test_e_means.py:92-98 `test_pass_counts_keep_small_fractions` (rows side): a tiny fraction rounds to 0 rows
#[test]
fn epoch_rows_tiny_fraction_rounds_to_zero_rows() {
    assert_eq!(epoch_rows(10, 0.01), vec![0]);
    assert_eq!(epoch_rows(10, 1.01), vec![10]);
}

// Faiss tests/test_e_means.py:108-129 `test_reference_values`
#[test]
fn reference_values() {
    let (b, a) = resolve_law(1_000_000, 16_384, 25.0, BatchLaw::Throughput);
    assert_eq!((b, a), (262_144, 1.0));
    let (b, a) = resolve_law(1_000_000, 16_384, 25.0, BatchLaw::Quality);
    assert_eq!(b, 32_768);
    assert_close(a, 0.15453882813046269, 1e-12);
    let (b, a) = resolve_law(100_000, 256, 5.0, BatchLaw::Throughput);
    assert_eq!((b, a), (8_192, 1.0));
    let (b, a) = resolve_law(100_000, 256, 5.0, BatchLaw::Quality);
    assert_eq!(b, 1_024);
    assert_close(a, 0.15543381717430582, 1e-12);
    let (b, a) = resolve_law(100_000_000, 262_144, 25.0, BatchLaw::Throughput);
    assert_eq!((b, a), (33_554_432, 1.0));
    let (b, a) = resolve_law(100_000_000, 262_144, 25.0, BatchLaw::Quality);
    assert_eq!(b, 4_194_304);
    assert_close(a, 0.42718555053293517, 1e-12);
}

// Faiss tests/test_e_means.py:131-142 `test_batch_size_alone_takes_the_step_law`
#[test]
fn batch_size_alone_takes_the_step_law() {
    let mut s = spec(1_000_000, 16_384, 32, 3.0);
    s.batch_size = Some(8_192);
    let r = Recipe::resolve(&s).unwrap();
    assert_eq!(r.batch_size, 8_192);
    assert_eq!(r.alpha0, step_law(1_000_000, 16_384, 3.0, 8_192));
    assert_eq!(r.batch_law, None);
    // Faiss :138-142: n=32, k=4, T=1, batch_size=8
    let mut s = spec(32, 4, 2, 1.0);
    s.batch_size = Some(8);
    let r = Recipe::resolve(&s).unwrap();
    assert_eq!(r.alpha0, step_law(32, 4, 1.0, 8));
}

#[test]
fn batch_size_is_capped_at_n_and_zero_is_rejected() {
    let mut s = spec(10, 2, 2, 1.0);
    s.batch_size = Some(1000);
    assert_eq!(Recipe::resolve(&s).unwrap().batch_size, 10);
    s.batch_size = Some(0);
    assert!(matches!(Recipe::resolve(&s), Err(Error::InvalidParameter(_))));
}

// Faiss tests/test_e_means.py:144-156 `test_auto_batch_law`
#[test]
fn auto_batch_law() {
    use BatchLaw::*;
    for (law, d, want) in [
        (Auto, 2, Quality),
        (Auto, 16, Quality),
        (Auto, 17, Throughput),
        (Auto, 1024, Throughput),
        (Throughput, 2, Throughput),
        (Quality, 128, Quality),
    ] {
        assert_eq!(resolve_batch_law(law, d), want, "{law:?} d={d}");
    }
}

// Faiss tests/test_e_means.py:158-165 `test_auto_unused_threshold`
#[test]
fn auto_unused_threshold() {
    assert_eq!(resolve_unused_threshold(None, 1).unwrap(), 0.0);
    assert_eq!(resolve_unused_threshold(None, 2).unwrap(), 0.0);
    assert_eq!(resolve_unused_threshold(None, 3).unwrap(), 0.1);
    assert_eq!(resolve_unused_threshold(Some(0.05), 2).unwrap(), 0.05);
    assert_eq!(resolve_unused_threshold(Some(0.0), 64).unwrap(), 0.0);
    assert!(matches!(
        resolve_unused_threshold(Some(-0.1), 8),
        Err(Error::InvalidParameter(_))
    ));
    assert!(matches!(
        resolve_unused_threshold(Some(f32::NAN), 8),
        Err(Error::InvalidParameter(_))
    ));
}

// Faiss tests/test_e_means.py:167-186 `test_recipe_records_resolved_defaults`
#[test]
fn recipe_records_resolved_defaults() {
    let r = Recipe::resolve(&spec(2048, 4, 2, 2.0)).unwrap();
    assert_eq!(r.batch_law, Some(BatchLaw::Quality));
    assert_eq!(r.unused_threshold, 0.0);
    assert_eq!(r.batch_size, batch_from_law(BatchLaw::Quality, 2048, 4, 2.0));

    let mut s = spec(2048, 4, 32, 2.0);
    s.batch_law = BatchLaw::Throughput;
    let r = Recipe::resolve(&s).unwrap();
    assert_eq!(r.batch_law, Some(BatchLaw::Throughput));
    assert_eq!(r.unused_threshold, 0.1);
    assert_eq!(r.batch_size, batch_from_law(BatchLaw::Throughput, 2048, 4, 2.0));
}

// Faiss tests/test_e_means.py:194-206 `test_derived_recipe_metadata`: n=31, T=3, steps_per_pass = ceil(n / B)
#[test]
fn derived_recipe_metadata() {
    let r = Recipe::resolve(&spec(31, 3, 4, 3.0)).unwrap();
    let steps_per_pass = 31usize.div_ceil(r.batch_size);
    assert_eq!(r.epoch_rows, vec![31, 31, 31]);
    assert_eq!(r.total_steps, 3 * steps_per_pass);
}

#[test]
fn total_steps_keeps_tail_batches() {
    let mut s = spec(31, 3, 4, 2.5);
    s.batch_size = Some(10);
    let r = Recipe::resolve(&s).unwrap();
    assert_eq!(r.epoch_rows, vec![31, 16, 31]);
    assert_eq!(r.total_steps, 4 + 2 + 4);
}

// Faiss tests/test_e_means.py:188-192 `test_default_nepochs` (pass count of T = 5)
#[test]
fn five_epochs_give_five_passes() {
    assert_eq!(Recipe::resolve(&spec(32, 4, 2, 5.0)).unwrap().epoch_rows.len(), 5);
}

// Faiss tests/test_e_means.py:208-214 `test_schedule_uses_logical_steps`
#[test]
fn schedule_uses_logical_steps() {
    let alpha0 = 0.4;
    assert_eq!(scheduled_alpha(alpha0, 0, 10), alpha0);
    assert_close(scheduled_alpha(alpha0, 10, 10), 0.04, 1e-12);
    let last = scheduled_alpha(alpha0, 9, 10);
    assert!(last > 0.04);
    assert_close(last, 0.04880982706687237, 1e-12);
    assert_eq!(scheduled_alpha(alpha0, 99, 10), scheduled_alpha(alpha0, 10, 10));
    assert_eq!(scheduled_alpha(alpha0, 0, 0), alpha0);
}

// Design criterion: round_pow2 (REF/base.py:67-73)
#[test]
fn round_pow2_nearest_power() {
    let got: Vec<usize> = [0.5, 1.0, 3.0, 5.6, 5.7, 6.0].into_iter().map(round_pow2).collect();
    assert_eq!(got, vec![1, 1, 4, 4, 8, 8]);
}

#[test]
fn resolve_rejects_bad_epochs_and_alpha0() {
    for epochs in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(
                Recipe::resolve(&spec(10, 2, 2, epochs)),
                Err(Error::InvalidParameter(_))
            ),
            "{epochs}"
        );
    }
    assert!(matches!(
        Recipe::resolve(&spec(10, 2, 2, 0.01)),
        Err(Error::InvalidParameter(_))
    ));
    for alpha0 in [0.0, -0.5, 1.5, f64::NAN] {
        let mut s = spec(10, 2, 2, 1.0);
        s.alpha0 = Some(alpha0);
        assert!(
            matches!(Recipe::resolve(&s), Err(Error::InvalidParameter(_))),
            "{alpha0}"
        );
    }
    let mut s = spec(10, 2, 2, 1.0);
    s.alpha0 = Some(1.0);
    assert_eq!(Recipe::resolve(&s).unwrap().alpha0, 1.0);
}

#[test]
fn constants_match_reference() {
    assert_eq!(INIT_USAGE, 2f32.powi(-16));
    assert_eq!(CHECK_UNUSED_EVERY, 5);
    assert_eq!(NO_REVIVAL_MAX_D, 2);
    assert_eq!(AUTO_QUALITY_MAX_D, 16);
    assert_eq!(SPLIT_EPSILON, 1e-4);
    assert_eq!(UNUSED_THRESHOLD, 0.1);
    assert_eq!(ALPHA_MIN_FACTOR, 0.1);
}
