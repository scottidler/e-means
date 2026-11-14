#![allow(clippy::unwrap_used)]

use super::*;
use crate::assign::{Kernel, Metric};
use crate::recipe::{BatchLaw, RecipeSpec};

fn recipe() -> Recipe {
    Recipe::resolve(&RecipeSpec {
        n: 4,
        k: 2,
        d: 1,
        epochs: 1.0,
        batch_law: BatchLaw::Auto,
        batch_size: Some(4),
        alpha0: Some(0.5),
        unused_threshold: None,
    })
    .expect("valid spec")
}

fn stats() -> EpochStats {
    EpochStats {
        objective: 8.0,
        imbalance_factor: 1.0,
        splits: 0,
        rows: 4,
    }
}

#[test]
fn accessors_return_what_the_trainer_handed_over() {
    let labels = vec![Some(0), Some(1)];
    let model = Model::new(
        1,
        Kernel::Direct,
        Metric::L2,
        vec![1.0, 11.0],
        vec![stats()],
        recipe(),
        Some(labels.clone()),
    );
    assert_eq!(model.centroids(), &[1.0, 11.0]);
    assert_eq!(model.epoch_stats(), &[stats()]);
    assert_eq!(model.recipe().batch_size, 4);
    assert_eq!(model.labels(), Some(&labels[..]));
}

#[test]
fn labels_are_none_when_not_captured() {
    let model = Model::new(1, Kernel::Direct, Metric::L2, vec![1.0], vec![], recipe(), None);
    assert_eq!(model.labels(), None);
}

fn two_cluster_model() -> Model {
    Model::new(
        2,
        Kernel::Direct,
        Metric::L2,
        vec![0.0, 0.0, 10.0, 10.0],
        vec![stats()],
        recipe(),
        None,
    )
}

#[test]
fn assign_returns_the_nearest_final_centroid_and_its_squared_distance() {
    let assignment = two_cluster_model()
        .assign(&[1.0, 0.0, 9.0, 10.0, 5.0, 5.0, 4.0, 4.0])
        .unwrap();
    assert_eq!(assignment.labels, [0, 1, 0, 0], "an exact tie goes to the lowest index");
    assert_eq!(assignment.distances, [1.0, 1.0, 50.0, 32.0]);
}

#[test]
fn assign_of_no_rows_is_empty() {
    let assignment = two_cluster_model().assign(&[]).unwrap();
    assert_eq!(
        assignment,
        Assignment {
            labels: vec![],
            distances: vec![]
        }
    );
}

#[test]
fn assign_rejects_partial_and_non_finite_rows() {
    let model = two_cluster_model();
    assert!(matches!(
        model.assign(&[1.0, 2.0, 3.0]),
        Err(Error::DimensionMismatch(ref m)) if m.contains("d=2")
    ));
    assert!(matches!(
        model.assign(&[1.0, 2.0, 3.0, f32::NAN]),
        Err(Error::NonFiniteInput(ref m)) if m.contains("row 1 column 1")
    ));
}

// `Model::assign` scans each 256-row task for NaN/inf and keeps the minimum offending offset. 300 rows are
// two tasks, the second short (44 rows); NaN at rows 5 and 290 names row 5, in 1- and 8-thread pools.
#[test]
fn assign_non_finite_rows_in_two_tasks_report_the_first_row() {
    let model = two_cluster_model();
    let clean: Vec<f32> = (0..300 * 2).map(|i| (i % 23) as f32 * 0.5).collect();
    let mut both = clean.clone();
    both[5 * 2 + 1] = f32::NAN;
    both[290 * 2] = f32::NAN;
    let mut late = clean.clone();
    late[290 * 2] = f32::NEG_INFINITY;
    for threads in [1, 8] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| {
            assert_eq!(
                model.assign(&both).unwrap_err(),
                Error::NonFiniteInput("row 5 column 1 is NaN".to_string()),
                "threads={threads}"
            );
            assert_eq!(
                model.assign(&late).unwrap_err(),
                Error::NonFiniteInput("row 290 column 0 is -inf".to_string()),
                "threads={threads}"
            );
            assert_eq!(model.assign(&clean).unwrap().labels.len(), 300);
        });
    }
}

#[test]
fn scatter_labels_moves_feed_positions_to_rows() {
    let labels = vec![Some(1), Some(0)];
    let mut model = Model::new(1, Kernel::Direct, Metric::L2, vec![1.0], vec![], recipe(), Some(labels));
    model.scatter_labels(&[3, 0, 2], 4);
    assert_eq!(model.labels().unwrap(), &[Some(0), None, None, Some(1)]);

    let mut uncaptured = Model::new(1, Kernel::Direct, Metric::L2, vec![1.0], vec![], recipe(), None);
    uncaptured.scatter_labels(&[0], 1);
    assert_eq!(uncaptured.labels(), None);
}

// Deviation D10: in spherical mode `Assignment.distances` holds `2 - 2 * ip`, and the label is the largest ip.
#[test]
fn spherical_assign_returns_two_minus_two_ip() {
    let model = Model::new(
        2,
        Kernel::Direct,
        Metric::InnerProduct,
        vec![1.0, 0.0, 0.0, 1.0],
        vec![stats()],
        recipe(),
        None,
    );
    let assignment = model.assign(&[0.6, 0.8, -1.0, 0.0, 0.0, 0.0]).unwrap();
    assert_eq!(
        assignment.labels,
        [1, 1, 0],
        "zero beats negative; a zero row ties at 0 and takes index 0"
    );
    assert_eq!(assignment.distances, [2.0 - 2.0 * 0.8f32, 2.0, 2.0]);
}

// Phase 8: blocks run in parallel; the result matches a serial pass and does not depend on the thread count
#[test]
fn assign_over_many_blocks_is_thread_count_invariant() {
    let model = two_cluster_model();
    let n = 3 * BLOCK_ROWS + 17;
    let rows: Vec<f32> = (0..n * 2).map(|i| ((i * 7919) % 23) as f32 * 0.5).collect();
    let mut serial_labels = vec![0; n];
    let mut serial_distances = vec![0.0; n];
    crate::assign::nearest(
        Kernel::Direct,
        Metric::L2,
        &rows,
        model.centroids(),
        &[],
        2,
        &mut serial_labels,
        &mut serial_distances,
        &mut Vec::new(),
    );
    for threads in [1, 8] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        let assignment = pool.install(|| model.assign(&rows)).unwrap();
        assert_eq!(assignment.labels, serial_labels, "threads {threads}");
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&assignment.distances),
            bits(&serial_distances),
            "threads {threads}"
        );
    }
}

/// `n` rows of `d` around `k` well separated centers: row `i` sits near center `i % k`, which is `+8` on
/// coordinate `i % k`; small deterministic noise elsewhere.
fn separated_rows(n: usize, d: usize, k: usize) -> Vec<f32> {
    let mut state = 5u64;
    let mut x = Vec::with_capacity(n * d);
    for i in 0..n {
        for c in 0..d {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let noise = ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2;
            x.push(if c == i % k { 8.0 + noise } else { noise });
        }
    }
    x
}

// The kernel is picked once from the recipe's `B * d` and the model keeps it for `assign`.
#[test]
fn model_keeps_the_kernel_picked_from_batch_times_d() {
    let (n, d, k) = (4000, 32, 4);
    let x = separated_rows(n, d, k);
    let init = x[..k * d].to_vec();
    for (batch, kernel) in [(3999, Kernel::Direct), (4000, Kernel::Gemm)] {
        for spherical in [false, true] {
            let model = crate::EMeans::new(d, k)
                .epochs(2.0)
                .batch_size(batch)
                .spherical(spherical)
                .train_from(&x, &init)
                .unwrap();
            assert_eq!(model.kernel, kernel, "B={batch} d={d} spherical={spherical}");
            let assignment = model.assign(&x).unwrap();
            let expected: Vec<u32> = (0..n).map(|i| (i % k) as u32).collect();
            assert_eq!(assignment.labels, expected, "B={batch} spherical={spherical}");
            if !spherical {
                assert!(assignment.distances.iter().all(|&v| v >= 0.0), "B={batch}");
            }
        }
    }
}
