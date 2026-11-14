#![allow(clippy::unwrap_used)]

use super::*;
use crate::assign::Kernel;
use crate::rng::{Stream, StreamId};

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

const TEST: &str = "tests/test_e_means.py";
const UTILS: &str = "tests/e_means_test_utils.py";

/// Deterministic values in `[-scale, scale)` from a 64-bit LCG; tests need data, not a seeded RNG contract.
fn values(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn objectives(model: &Model) -> Vec<u64> {
    model.epoch_stats().iter().map(|s| s.objective.to_bits()).collect()
}

fn rows_of(x: &[f32], ids: &[usize], d: usize) -> Vec<f32> {
    ids.iter()
        .flat_map(|&i| x[i * d..(i + 1) * d].iter().copied())
        .collect()
}

/// The pass orders `train` draws: pass `i` is a permutation of `0..n` from stream `(seed, Pass(i))`.
fn pass_order(seed: u64, pass: usize, n: usize) -> Vec<usize> {
    Stream::new(seed, StreamId::Pass(pass as u64)).permutation(n)
}

/// A caller-driven run fed `init` and `train`'s pass orders, each batch in two chunks (2 rows, then the
/// rest). Returns the model (labels in feed order) and the last pass's order.
fn caller_run(builder: &EMeans, x: &[f32], d: usize, init: &[f32], seed: u64) -> (Model, Vec<usize>) {
    let n = x.len() / d;
    let mut trainer = builder.begin(n, init).unwrap();
    let batch = trainer.batch_rows();
    let mut last = Vec::new();
    for (pass, rows) in trainer.recipe().epoch_rows.clone().into_iter().enumerate() {
        let order = pass_order(seed, pass, n);
        for ids in order[..rows].chunks(batch) {
            let chunk = rows_of(x, ids, d);
            let split = 2.min(ids.len()) * d;
            trainer.accumulate(&chunk[..split]).unwrap();
            trainer.accumulate(&chunk[split..]).unwrap();
            trainer.update().unwrap();
        }
        trainer.end_epoch().unwrap();
        last = order;
    }
    (trainer.finalize().unwrap(), last)
}

/// Feed-order labels moved to row order, `None` on rows the last pass did not visit.
fn row_order(model: &Model, order: &[usize], n: usize) -> Vec<Option<u32>> {
    let mut labels = vec![None; n];
    for (&row, &label) in order.iter().zip(model.labels().unwrap()) {
        labels[row] = label;
    }
    labels
}

// Design criterion: a caller-driven loop fed the same init and orders is bitwise equal to `train`.
// Faiss tests/test_e_means.py:391-415 `test_caller_batches_match_matrix_training`
#[test]
fn caller_batches_match_matrix_training() {
    let d = 3;
    let x = values(17 * d, 123, 2.0);
    let init = x[..4 * d].to_vec();
    for epochs in [3.0, 2.5] {
        let builder = EMeans::new(d, 4).epochs(epochs).batch_size(4).alpha0(0.2).seed(7);
        let matrix = builder.train_from(&x, &init).unwrap();
        let (caller, order) = caller_run(&builder, &x, d, &init, 7);
        assert_eq!(
            bits(caller.centroids()),
            bits(matrix.centroids()),
            "{TEST}:412 epochs={epochs}"
        );
        assert_eq!(objectives(&caller), objectives(&matrix), "{TEST}:413 epochs={epochs}");
        let splits = |m: &Model| m.epoch_stats().iter().map(|s| s.splits).collect::<Vec<_>>();
        assert_eq!(splits(&caller), splits(&matrix), "{TEST}:414 epochs={epochs}");
        let labels = row_order(&caller, &order, 17);
        assert!(labels.iter().all(Option::is_some));
        assert_eq!(matrix.labels().unwrap(), &labels[..], "epochs={epochs}");
    }
}

// Design criterion, seeded-init half: `train` draws init as the first k rows of the Init stream's permutation
#[test]
fn train_matches_a_caller_loop_from_the_seeded_init() {
    let d = 3;
    let x = values(50 * d, 5, 3.0);
    let builder = EMeans::new(d, 5).epochs(2.0).batch_size(8).alpha0(0.3).seed(11);
    let model = builder.train(&x).unwrap();
    let init = rows_of(&x, &Stream::new(11, StreamId::Init).permutation(50)[..5], d);
    let (caller, order) = caller_run(&builder, &x, d, &init, 11);
    assert_eq!(bits(model.centroids()), bits(caller.centroids()));
    assert_eq!(objectives(&model), objectives(&caller));
    assert_eq!(model.labels().unwrap(), &row_order(&caller, &order, 50)[..]);
}

// `train` against a caller loop through the public `accumulate`, inside explicit 1-thread and 8-thread
// pools: bitwise centroids, epoch stats (objective included) and captured labels. Batches span several
// 256-row blocks (B=1024 is 4 blocks, 16 tasks at 8 threads), n is not a multiple of B (a short final
// batch), and T is whole and fractional. Pins the fused `accumulate_indexed` path (Phase 4) and the
// pool-thread loop with its permutation overlap (Phase 7) to `accumulate`.
#[test]
fn train_matches_caller_loop_in_pools() {
    // (name, d, k, batch_size, n, spherical)
    let cases = [
        ("L2 Direct", 8, 16, 1024, 5000, false),
        ("L2 Gemm", 32, 64, 4096, 10_000, false),
        ("spherical Direct", 8, 16, 1024, 5000, true),
        ("spherical Gemm", 32, 64, 4096, 10_000, true),
    ];
    let pools: Vec<_> = [1, 8]
        .into_iter()
        .map(|threads| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            (threads, pool)
        })
        .collect();
    for (name, d, k, batch, n, spherical) in cases {
        let kernel = Kernel::for_batch(batch, d);
        assert_eq!(
            kernel,
            if name.ends_with("Gemm") { Kernel::Gemm } else { Kernel::Direct },
            "{name}: the case's B * d picks the kernel its name says"
        );
        assert_ne!(n % batch, 0, "{name}: n is not a multiple of B");
        let seed = 21;
        let mut x = values(n * d, 33, 3.0);
        if spherical {
            x = unit_rows(x, d);
        }
        let init = rows_of(&x, &Stream::new(seed, StreamId::Init).permutation(n)[..k], d);
        for epochs in [5.0, 2.5] {
            let builder = EMeans::new(d, k)
                .epochs(epochs)
                .batch_size(batch)
                .alpha0(0.2)
                .seed(seed)
                .spherical(spherical);
            for (threads, pool) in &pools {
                let (model, caller, order) = pool.install(|| {
                    let model = builder.train(&x).unwrap();
                    let (caller, order) = caller_run(&builder, &x, d, &init, seed);
                    (model, caller, order)
                });
                let what = format!("{name} T={epochs} threads={threads}");
                assert_eq!(bits(model.centroids()), bits(caller.centroids()), "{what}: centroids");
                assert_eq!(model.epoch_stats(), caller.epoch_stats(), "{what}: epoch stats");
                assert_eq!(objectives(&model), objectives(&caller), "{what}: objective bits");
                assert_eq!(
                    model.labels().unwrap(),
                    &row_order(&caller, &order, n)[..],
                    "{what}: captured labels"
                );
            }
        }
    }
}

// `train`'s fused region scans each task for NaN/inf and keeps the minimum offending offset (`fetch_min`).
// NaN at gathered positions 5 and 900 of the first batch land in blocks 0 and 3 (64-row tasks 0 and 14 at
// 8 threads); the error
// names batch row 5, the message `accumulate` gives for the gathered batch, in 1- and 8-thread pools. One
// case per gather: a batch under `PARALLEL_REDUCE_VALUES` (batch-wide buffer) and one at or over it
// (task-local buffers, the value read back from `x`).
#[test]
fn train_non_finite_rows_in_two_tasks_report_the_first_batch_row() {
    use crate::trainer::PARALLEL_REDUCE_VALUES;
    // (d, batch_size, n, the batch reduces in parallel)
    for (d, batch, n, parallel) in [(8, 1024, 3000, false), (32, 4096, 5000, true)] {
        assert_eq!(
            batch * d >= PARALLEL_REDUCE_VALUES,
            parallel,
            "d={d}: the case's side of the line"
        );
        let (k, seed) = (16, 4);
        let clean = values(n * d, 17, 3.0);
        let init = clean[..k * d].to_vec();
        let builder = EMeans::new(d, k).epochs(2.0).batch_size(batch).alpha0(0.2).seed(seed);
        let first_batch = pass_order(seed, 0, n);
        let poisoned = |marks: &[(usize, usize, f32)]| {
            let mut x = clean.clone();
            for &(position, column, value) in marks {
                x[first_batch[position] * d + column] = value;
            }
            x
        };
        let both = poisoned(&[(5, 3, f32::NAN), (900, 6, f32::NAN)]);
        let late = poisoned(&[(900, 6, f32::INFINITY)]);
        for threads in [1, 8] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| {
                assert_eq!(
                    builder.train_from(&both, &init).unwrap_err(),
                    Error::NonFiniteInput("chunk row 5 column 3 is NaN".to_string()),
                    "d={d} threads={threads}"
                );
                assert_eq!(
                    builder.train_from(&late, &init).unwrap_err(),
                    Error::NonFiniteInput("chunk row 900 column 6 is inf".to_string()),
                    "d={d} threads={threads}"
                );
                builder.train_from(&clean, &init).unwrap();
            });
        }
    }
}

// Design criterion: for T in {0.5, 0.1}, labels are None on exactly the unvisited rows.
// Faiss tests/e_means_test_utils.py:389-484 `test_fractional_epochs_preserve_batches_and_closing_labels`
#[test]
fn fractional_epochs_preserve_batches_and_closing_labels() {
    let cases: [(f64, &[usize]); 5] = [
        (2.0, &[31, 31]),
        (2.5, &[31, 16, 31]),
        (1.25, &[8, 31]),
        (0.5, &[16]),
        (0.1, &[3]),
    ];
    let x: Vec<f32> = (0..31).map(|i| i as f32 / 4.0).collect();
    let initial = rows_of(&x, &[0, 10, 20, 30], 1);
    for (budget, epoch_rows) in cases {
        let builder = EMeans::new(1, 4).epochs(budget).batch_size(7).alpha0(0.2).seed(7);
        let matrix = builder.train_from(&x, &initial).unwrap();
        let steps: Vec<usize> = epoch_rows.iter().map(|n| n.div_ceil(7)).collect();
        assert_eq!(matrix.recipe().epoch_rows, epoch_rows, "{UTILS}:425 T={budget}");
        assert_eq!(
            matrix.recipe().total_steps,
            steps.iter().sum::<usize>(),
            "{UTILS}:427 T={budget}"
        );
        let (caller, order) = caller_run(&builder, &x, 1, &initial, 7);
        assert_eq!(
            bits(caller.centroids()),
            bits(matrix.centroids()),
            "{UTILS}:452 T={budget}"
        );
        assert_eq!(objectives(&caller), objectives(&matrix), "{UTILS}:458 T={budget}");

        let captured = matrix.labels().unwrap();
        assert_eq!(captured.len(), 31);
        let last = *epoch_rows.last().unwrap();
        let visited = &order[..last];
        for (row, label) in captured.iter().enumerate() {
            assert_eq!(
                label.is_some(),
                visited.contains(&row),
                "{UTILS}:468 T={budget} row {row}"
            );
        }
        assert_eq!(captured.iter().filter(|l| l.is_none()).count(), 31 - last);
        assert_eq!(captured, &row_order(&caller, &order, 31)[..], "T={budget}");

        let mut counts = [0f64; 4];
        for centroid in 0..4u32 {
            let assigned: Vec<f64> = visited
                .iter()
                .filter(|&&row| captured[row] == Some(centroid))
                .map(|&row| f64::from(x[row]))
                .collect();
            counts[centroid as usize] = assigned.len() as f64;
            if !assigned.is_empty() {
                let mean = assigned.iter().sum::<f64>() / assigned.len() as f64;
                let got = f64::from(matrix.centroids()[centroid as usize]);
                assert!(
                    (got - mean).abs() <= 1e-6 + 1e-6 * mean.abs(),
                    "{UTILS}:474 T={budget} centroid {centroid}: {got} vs {mean}"
                );
            }
        }
        let square: f64 = counts.iter().map(|c| c * c).sum();
        assert_eq!(
            matrix.epoch_stats().last().unwrap().imbalance_factor,
            4.0 * square / (last * last) as f64,
            "{UTILS}:481 T={budget}"
        );
    }
}

/// 16 Gaussian-ish blobs: centers uniform in `[-scale, scale)^d`, rows at unit spread around them.
fn blobs(n: usize, d: usize, blobs: usize, scale: f32) -> Vec<f32> {
    let centers = values(blobs * d, 1, scale);
    let noise = values(n * d, 2, 1.0);
    (0..n * d)
        .map(|at| {
            let (row, col) = (at / d, at % d);
            centers[(row % blobs) * d + col] + noise[at]
        })
        .collect()
}

// Design criterion: blobs n=2000, d=8, 16 blobs at scale 10, k=16; seed 1 twice is bitwise equal, seed 2 differs
#[test]
fn seeded_runs_on_blobs_are_reproducible() {
    let x = blobs(2000, 8, 16, 10.0);
    let first = EMeans::new(8, 16).seed(1).train(&x).unwrap();
    let again = EMeans::new(8, 16).seed(1).train(&x).unwrap();
    assert_eq!(bits(first.centroids()), bits(again.centroids()));
    assert_eq!(first.labels(), again.labels());
    assert_eq!(objectives(&first), objectives(&again));
    let other = EMeans::new(8, 16).seed(2).train(&x).unwrap();
    assert_ne!(bits(other.centroids()), bits(first.centroids()));
}

// Faiss tests/test_e_means.py:474-486 `test_fresh_objects_are_deterministic`
// (`:488-498`, the global RNG check, has no counterpart: the crate has no global RNG state.)
#[test]
fn fresh_runs_are_deterministic() {
    let x = values(40 * 4, 123, 2.0);
    let builder = EMeans::new(4, 3).epochs(2.0).batch_size(11).alpha0(0.2);
    let first = builder.train(&x).unwrap();
    let second = EMeans::new(4, 3)
        .epochs(2.0)
        .batch_size(11)
        .alpha0(0.2)
        .train(&x)
        .unwrap();
    assert_eq!(bits(second.centroids()), bits(first.centroids()), "{TEST}:484");
    assert_eq!(objectives(&second), objectives(&first), "{TEST}:485");
}

// Faiss tests/e_means_test_utils.py:615-664 `test_max_points_per_centroid_trains_on_a_seeded_subset`
#[test]
fn max_points_per_centroid_trains_on_a_seeded_subset() {
    let data: Vec<f32> = values(400 * 4, 0, 0.5).iter().map(|v| v + 0.5).collect();
    let builder = EMeans::new(4, 8).epochs(2.0).batch_size(32).alpha0(0.3).seed(5);
    let capped = builder.clone().max_points_per_centroid(10).train(&data).unwrap();
    let labels = capped.labels().unwrap();
    assert_eq!(labels.len(), 400, "{UTILS}:627");
    let kept: Vec<usize> = (0..400).filter(|&row| labels[row].is_some()).collect();
    assert_eq!(kept.len(), 80, "{UTILS}:629");
    assert_eq!(capped.recipe().epoch_rows.iter().max(), Some(&80), "{UTILS}:630");
    let reference = builder.train(&rows_of(&data, &kept, 4)).unwrap();
    assert_eq!(bits(capped.centroids()), bits(reference.centroids()), "{UTILS}:632");
    let kept_labels: Vec<Option<u32>> = kept.iter().map(|&row| labels[row]).collect();
    assert_eq!(&kept_labels[..], reference.labels().unwrap(), "{UTILS}:634");

    let loose = builder.clone().max_points_per_centroid(50).train(&data).unwrap();
    let plain = builder.train(&data).unwrap();
    assert_eq!(bits(loose.centroids()), bits(plain.centroids()), "{UTILS}:637");
    assert_eq!(loose.labels(), plain.labels());

    let err = builder.clone().max_points_per_centroid(0).train(&data).unwrap_err();
    assert!(
        matches!(err, Error::InvalidParameter(ref m) if m.contains("max_points_per_centroid")),
        "{UTILS}:638 {err}"
    );
    let other_seed = builder
        .clone()
        .seed(6)
        .max_points_per_centroid(10)
        .train(&data)
        .unwrap();
    let other_kept: Vec<usize> = (0..400)
        .filter(|&row| other_seed.labels().unwrap()[row].is_some())
        .collect();
    assert_eq!(other_kept.len(), 80);
    assert_ne!(other_kept, kept, "the subset follows the seed");
}

// Faiss tests/e_means_test_utils.py:666-703 `test_last_pass_labels_are_owned_and_not_final_nearest`
#[test]
fn last_pass_labels_are_not_final_nearest() {
    let rows = [0.0, 1.0, 49.0, 51.0, 52.0, 53.0];
    let initial = [0.0, 100.0];
    let builder = EMeans::new(1, 2).epochs(1.0).batch_size(6).alpha0(0.5).seed(7);
    let model = builder.train_from(&rows, &initial).unwrap();
    let labels = model.labels().unwrap();
    assert_eq!(
        labels,
        &[Some(0), Some(0), Some(0), Some(1), Some(1), Some(1)],
        "{UTILS}:676"
    );
    let want = [50.0 / 3.0, 52.0];
    for (got, want) in model.centroids().iter().zip(want) {
        assert!(
            (f64::from(*got) - want).abs() <= 1e-6 * want,
            "{UTILS}:681 {got} vs {want}"
        );
    }
    let nearest = model.assign(&rows).unwrap();
    assert_eq!(nearest.labels[2], 1, "{UTILS}:684");
    assert_eq!(labels[2], Some(0), "{UTILS}:685");

    let disabled = builder
        .clone()
        .capture_labels(false)
        .train_from(&rows, &initial)
        .unwrap();
    assert_eq!(disabled.labels(), None, "{UTILS}:688");
    assert_eq!(bits(disabled.centroids()), bits(model.centroids()), "{UTILS}:689");
    assert_eq!(objectives(&disabled), objectives(&model), "{UTILS}:692");
    let again = builder.train_from(&rows, &initial).unwrap();
    assert_eq!(again.labels(), model.labels(), "{UTILS}:701");
}

#[test]
fn capture_labels_false_skips_caller_driven_labels() {
    let mut trainer = EMeans::new(1, 2)
        .epochs(1.0)
        .batch_size(4)
        .capture_labels(false)
        .begin(4, &[0.0, 10.0])
        .unwrap();
    trainer.accumulate(&[0.0, 2.0, 10.0, 12.0]).unwrap();
    trainer.update().unwrap();
    trainer.end_epoch().unwrap();
    assert_eq!(trainer.finalize().unwrap().labels(), None);
}

// Edge-case table: n == k in `train` uses the rows as given, no draw (REF/e_means.py:593-600)
#[test]
fn n_equal_k_uses_the_rows_as_given() {
    for seed in 0..8 {
        let model = EMeans::new(1, 2)
            .epochs(1.0)
            .batch_size(1)
            .alpha0(1.0)
            .seed(seed)
            .train(&[0.0, 10.0])
            .unwrap();
        assert_eq!(model.centroids(), &[0.0, 10.0], "seed {seed}");
    }
    let x = values(3 * 2, 9, 1.0);
    let builder = EMeans::new(2, 3).epochs(1.5).batch_size(2).alpha0(0.4).seed(3);
    let (caller, _) = caller_run(&builder, &x, 2, &x, 3);
    assert_eq!(bits(builder.train(&x).unwrap().centroids()), bits(caller.centroids()));
}

#[test]
fn train_errors_are_typed() {
    assert!(matches!(
        EMeans::new(0, 2).train(&[0.0]),
        Err(Error::InvalidParameter(_))
    ));
    assert!(matches!(
        EMeans::new(1, 0).train(&[0.0]),
        Err(Error::InvalidParameter(_))
    ));
    assert!(matches!(
        EMeans::new(2, 1).train(&[0.0; 3]),
        Err(Error::DimensionMismatch(ref m)) if m.contains("d=2")
    ));
    assert_eq!(
        EMeans::new(1, 3).train(&[0.0, 1.0]).unwrap_err(),
        Error::TooFewRows { rows: 2, k: 3 }
    );
    assert!(matches!(
        EMeans::new(1, 2).train_from(&[0.0, 1.0, 2.0], &[0.0]),
        Err(Error::DimensionMismatch(ref m)) if m.contains("want k*d = 2")
    ));
    let err = EMeans::new(1, 2)
        .train_from(&[0.0, 1.0, f32::NAN, 3.0], &[0.0, 3.0])
        .unwrap_err();
    assert!(matches!(err, Error::NonFiniteInput(_)), "{err}");
    assert!(matches!(
        EMeans::new(1, 1).epochs(0.0).train(&[0.0]),
        Err(Error::InvalidParameter(ref m)) if m.contains("epochs")
    ));
}

fn unit_rows(mut x: Vec<f32>, d: usize) -> Vec<f32> {
    for row in x.chunks_exact_mut(d) {
        let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        row.iter_mut().for_each(|v| *v /= norm);
    }
    x
}

/// Rows whose norm is not within 1e-6 of 1 and not exactly zero.
fn off_sphere(centroids: &[f32], d: usize) -> Vec<(usize, f32)> {
    centroids
        .chunks_exact(d)
        .map(|c| c.iter().map(|v| v * v).sum::<f32>().sqrt())
        .enumerate()
        .filter(|&(_, norm)| norm != 0.0 && (norm - 1.0).abs() > 1e-6)
        .collect()
}

// Design criterion: every final centroid has norm within 1e-6 of 1, or is exactly zero.
#[test]
fn spherical_final_centroids_are_unit_or_zero() {
    let x = unit_rows(blobs(2000, 8, 16, 10.0), 8);
    let model = EMeans::new(8, 16).spherical(true).seed(3).train(&x).unwrap();
    assert_eq!(off_sphere(model.centroids(), 8), [], "blobs on the sphere");
    assert_eq!(model.recipe().unused_threshold, 0.1, "revival is live at d = 8");

    // The UTILS:118 fixture through `train_from`: centroid 0 only receives rows that sum to zero.
    let rows = [
        0.0,
        0.0,
        0.25,
        15f32.sqrt() / 4.0,
        1.0,
        0.0,
        -1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        -1.0,
    ];
    let model = EMeans::new(2, 3)
        .epochs(1.0)
        .batch_size(6)
        .alpha0(1.0)
        .spherical(true)
        .train_from(&rows, &[0.0, 0.0, 1.0, 0.0, -1.0, 0.0])
        .unwrap();
    assert_eq!(off_sphere(model.centroids(), 2), []);
    assert_eq!(
        &model.centroids()[..2],
        &[0.0, 0.0],
        "the exactly-zero branch is exercised"
    );
}

// Faiss tests/e_means_test_utils.py:327-388 `test_spherical_fractional_caller_and_matrix`. The reference
// compares at atol 2e-7; with the crate's own pass orders (D1) the caller loop is bitwise equal.
#[test]
fn spherical_fractional_caller_and_matrix() {
    let d = 5;
    let rows = unit_rows(values(31 * d, 73, 1.0), d);
    let init = rows[..4 * d].to_vec();
    for budget in [0.5, 1.5] {
        let builder = EMeans::new(d, 4)
            .epochs(budget)
            .spherical(true)
            .batch_size(7)
            .alpha0(0.2)
            .seed(7);
        let matrix = builder.train_from(&rows, &init).unwrap();
        let (caller, order) = caller_run(&builder, &rows, d, &init, 7);
        assert_eq!(
            bits(matrix.centroids()),
            bits(caller.centroids()),
            "{UTILS}:377 T={budget}"
        );
        assert_eq!(
            matrix.labels().unwrap(),
            &row_order(&caller, &order, 31)[..],
            "{UTILS}:382 T={budget}"
        );
        assert_eq!(objectives(&matrix), objectives(&caller), "{UTILS}:385 T={budget}");
        assert_eq!(off_sphere(matrix.centroids(), d), [], "T={budget}");
    }
}

#[test]
fn spherical_defaults_off() {
    assert!(!EMeans::new(2, 2).spherical);
    assert!(EMeans::new(2, 2).spherical(true).spherical);
}
