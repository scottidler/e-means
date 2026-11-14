#![allow(clippy::unwrap_used)]

use super::*;
use crate::config::EMeans;
use std::f32::consts::FRAC_1_SQRT_2;

const TEST: &str = "tests/test_e_means.py";
const UTILS: &str = "tests/e_means_test_utils.py";

/// Feed every pass of the schedule in row order: batches of B, each cut into chunks whose sizes cycle
/// through `chunks` (clipped at the batch end), then `update` per batch and `end_epoch` per pass.
fn feed(trainer: &mut Trainer, x: &[f32], d: usize, chunks: &[usize]) {
    let passes = trainer.recipe().epoch_rows.clone();
    let batch = trainer.batch_rows();
    let mut sizes = chunks.iter().cycle();
    for rows in passes {
        let mut fed = 0;
        while fed < rows {
            let want = batch.min(rows - fed);
            let mut got = 0;
            while got < want {
                let size = (*sizes.next().unwrap()).min(want - got);
                let start = fed + got;
                trainer.accumulate(&x[start * d..(start + size) * d]).unwrap();
                got += size;
            }
            trainer.update().unwrap();
            fed += want;
        }
        trainer.end_epoch().unwrap();
    }
}

fn train(builder: &EMeans, x: &[f32], d: usize, init: &[f32], chunks: &[usize]) -> Model {
    let mut trainer = builder.begin(x.len() / d, init).unwrap();
    feed(&mut trainer, x, d, chunks);
    trainer.finalize().unwrap()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn objectives(model: &Model) -> Vec<u64> {
    model.epoch_stats().iter().map(|s| s.objective.to_bits()).collect()
}

/// Deterministic values in `[-scale, scale)` from a 64-bit LCG; tests need data, not a seeded RNG contract.
fn lcg_values(len: usize, seed: u64, scale: f32) -> Vec<f32> {
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

/// A fixture whose batch spans 3 blocks: B = 2 * BLOCK_ROWS + 37, n not a multiple of B (tail batches).
fn three_block_fixture() -> (EMeans, Vec<f32>, Vec<f32>, usize) {
    let d = 3;
    let k = 5;
    let batch = 2 * BLOCK_ROWS + 37;
    let n = 2 * batch + 101;
    let mut x = lcg_values(n * d, 11, 1.0);
    for (i, row) in x.chunks_exact_mut(d).enumerate() {
        row[0] += (i % k) as f32 * 4.0;
    }
    let init = x[..k * d].to_vec();
    let builder = EMeans::new(d, k).epochs(2.5).batch_size(batch).alpha0(0.3);
    (builder, x, init, d)
}

// Faiss tests/test_e_means.py:300-316 `test_closing_mean_and_assign`, with the captured labels of :339
#[test]
fn closing_mean_four_point_fixture() {
    let x = [0.0, 2.0, 10.0, 12.0];
    let builder = EMeans::new(1, 2).epochs(1.0).batch_size(4).alpha0(0.5);
    let model = train(&builder, &x, 1, &[0.0, 10.0], &[4]);
    assert_eq!(model.centroids(), &[1.0, 11.0], "{TEST}:307");
    assert_eq!(model.epoch_stats()[0].objective, 8.0, "{TEST}:308");
    assert_eq!(model.recipe().batch_law, None, "{TEST}:312");
    assert_eq!(
        model.labels().unwrap(),
        &[Some(0), Some(0), Some(1), Some(1)],
        "{TEST}:339"
    );
    let mut labels = [0u32; 4];
    let mut distances = [0f32; 4];
    assign::nearest(
        Kernel::Direct,
        Metric::L2,
        &x,
        model.centroids(),
        &[],
        1,
        &mut labels,
        &mut distances,
    );
    assert_eq!(labels, [0, 0, 1, 1], "{TEST}:315");
    assert_eq!(distances, [1.0, 1.0, 1.0, 1.0], "{TEST}:316");
    let stats = &model.epoch_stats()[0];
    assert_eq!((stats.imbalance_factor, stats.splits, stats.rows), (1.0, 0, 4));
}

// Faiss tests/test_e_means.py:379-389 `test_untouched_centroid_retains_its_position`
#[test]
fn untouched_centroid_retains_its_position() {
    let x = [0.0, 1.0, 2.0, 3.0];
    let init = [0.0, 100.0];
    for epochs in [1.0, 2.0] {
        let builder = EMeans::new(1, 2).epochs(epochs).batch_size(4).alpha0(1.0);
        let model = train(&builder, &x, 1, &init, &[4]);
        assert_eq!(model.centroids(), &[1.5, 100.0], "{TEST}:384 epochs={epochs}");
    }
}

// Faiss tests/test_e_means.py:444-453 `test_each_pass_uses_tail`
#[test]
fn each_pass_uses_tail() {
    let x = [0.0, 0.0, 0.0, 0.0, 10.0];
    let builder = EMeans::new(1, 1).epochs(2.0).batch_size(3).alpha0(0.5);
    let trainer = builder.begin(5, &[0.0]).unwrap();
    assert_eq!(trainer.recipe().total_steps, 4, "{TEST}:452");
    let model = train(&builder, &x, 1, &[0.0], &[3]);
    assert_eq!(model.centroids(), &[2.0], "{TEST}:450");
}

// Faiss tests/e_means_test_utils.py:909-951 `test_initial_mass_first_update_and_zero_mass_reentry`
#[test]
fn initial_mass_first_update_and_zero_mass_reentry() {
    for (alpha, zero_mass, position) in [(0.5, false, 3.0f32), (1.0, false, 1e8), (0.5, true, 1e8)] {
        let mut trainer = EMeans::new(1, 2)
            .epochs(2.0)
            .batch_size(2)
            .alpha0(alpha)
            .begin(2, &[position, 8.25])
            .unwrap();
        assert_eq!(trainer.usage, vec![INIT_USAGE; 2], "{UTILS}:922");
        if zero_mass {
            trainer.usage[0] = 0.0;
        }
        let mass = if zero_mass { 0.0 } else { INIT_USAGE as f64 };
        let weight = alpha * 2.0 / ((1.0 - alpha) * mass + alpha * 2.0);
        let expected = if weight == 1.0 {
            1.0
        } else {
            position as f64 + weight * (1.0 - position as f64)
        };
        trainer
            .update_with(BatchStats {
                sums: &[2.0, 0.0],
                counts: &[2, 0],
                objective: 7.0,
            })
            .unwrap();
        let got = trainer.centroids[0] as f64;
        assert!(
            (got - expected).abs() <= 2e-7 * expected.abs(),
            "{UTILS}:945 alpha={alpha} zero_mass={zero_mass}: got {got}, want {expected}"
        );
        assert_eq!(trainer.centroids[1], 8.25, "{UTILS}:947");
        let unused = if alpha == 1.0 { INIT_USAGE } else { (1.0 - alpha) as f32 * INIT_USAGE };
        assert_eq!(trainer.usage[1], unused, "{UTILS}:951");
    }
}

// Faiss tests/e_means_test_utils.py:954-990 `test_untouched_underflow_and_independent_ema`
#[test]
fn ema_matches_f64_scalar_oracle_over_400_steps() {
    let mut trainer = EMeans::new(1, 2)
        .epochs(1.0)
        .batch_size(1)
        .alpha0(0.9)
        .begin(400, &[0.0, 8.25])
        .unwrap();
    let mut mass = INIT_USAGE as f64;
    let mut position = 0.0f64;
    for step in 0..400usize {
        let value = ((step % 7) as f64 - 3.0) / 8.0;
        let alpha = scheduled_alpha(0.9, step, 400);
        mass = (1.0 - alpha) * mass + alpha;
        position += alpha / mass * (value - position);
        trainer
            .update_with(BatchStats {
                sums: &[value, 0.0],
                counts: &[1, 0],
                objective: 0.0,
            })
            .unwrap();
        let got = trainer.centroids[0] as f64;
        assert!(
            (got - position).abs() <= 3e-7 + 3e-6 * position.abs(),
            "{UTILS}:979 step {step}: got {got}, oracle {position}"
        );
        assert_eq!(trainer.centroids[1], 8.25, "{UTILS}:985 step {step}");
    }
    assert!(trainer.usage[1] <= f32::MIN_POSITIVE, "{UTILS}:986");
    trainer.end_epoch().unwrap();
    assert_eq!(trainer.finalize().unwrap().centroids()[1], 8.25, "{UTILS}:990");
}

// Faiss tests/e_means_test_utils.py:486-613 `test_supplied_chunks_share_one_frozen_batch`
#[test]
fn supplied_chunks_share_one_frozen_batch() {
    let rows: [f32; 26] = [
        0.25, 0.0, 10.0, 10.25, 0.5, 0.25, 10.25, 10.0, 0.0, 0.25, 10.5, 10.25, 0.5, 0.0, 10.0, 10.5, 0.25, 0.5, 10.25,
        10.5, 0.5, 0.5, 10.5, 10.5, 0.0, 0.0,
    ];
    let builder = EMeans::new(2, 2).epochs(2.0).batch_size(4).alpha0(0.3);
    let init = [0.0, 0.0, 10.0, 10.0];
    let mut whole = builder.begin(13, &init).unwrap();
    let mut chunked = builder.begin(13, &init).unwrap();
    chunked.accumulate(&rows[..0]).unwrap();
    assert_eq!(chunked.received(), 0, "{UTILS}:520");
    for epoch in 0..2 {
        for start in (0..13).step_by(4) {
            let end = (start + 4).min(13);
            let batch = &rows[start * 2..end * 2];
            whole.accumulate(batch).unwrap();
            let frozen = chunked.centroids.clone();
            for part in batch.chunks(4) {
                chunked.accumulate(part).unwrap();
                assert_eq!(chunked.centroids, frozen, "{UTILS}:534");
                assert_eq!(chunked.step, whole.step, "{UTILS}:537");
            }
            assert_eq!(chunked.batch.counts, whole.batch.counts, "{UTILS}:586");
            assert_eq!(chunked.batch.sums, whole.batch.sums, "{UTILS}:586");
            assert_eq!(chunked.batch.objective, whole.batch.objective, "{UTILS}:586");
            if epoch == 1 {
                let labels = chunked.pass_labels.as_ref().unwrap();
                let captured = &labels[labels.len() - (end - start)..];
                for (row, &label) in batch.as_chunks::<2>().0.iter().zip(captured) {
                    let distance = |j: usize| {
                        (0..2)
                            .map(|c| (row[c] as f64 - frozen[j * 2 + c] as f64).powi(2))
                            .sum::<f64>()
                    };
                    let expected = if distance(1) < distance(0) { 1 } else { 0 };
                    assert_eq!(label, expected, "{UTILS}:558");
                }
            }
            whole.update().unwrap();
            chunked.update().unwrap();
        }
        whole.end_epoch().unwrap();
        chunked.end_epoch().unwrap();
    }
    let whole = whole.finalize().unwrap();
    let chunked = chunked.finalize().unwrap();
    assert_eq!(bits(chunked.centroids()), bits(whole.centroids()), "{UTILS}:600");
    assert_eq!(objectives(&chunked), objectives(&whole), "{UTILS}:606");
    assert_eq!(chunked.labels(), whole.labels());
}

// Design criterion: one chunk, row-at-a-time, and chunks of 3, 5, 7 rows give bitwise equal results
#[test]
fn chunking_is_invisible_on_a_three_block_batch() {
    let (builder, x, init, d) = three_block_fixture();
    let one = train(&builder, &x, d, &init, &[usize::MAX]);
    assert_eq!(
        one.recipe().batch_size.div_ceil(BLOCK_ROWS),
        3,
        "the batch spans 3 blocks"
    );
    assert_eq!(one.recipe().epoch_rows.len(), 3);
    for chunks in [&[1usize][..], &[3, 5, 7]] {
        let other = train(&builder, &x, d, &init, chunks);
        assert_eq!(bits(other.centroids()), bits(one.centroids()), "chunks {chunks:?}");
        assert_eq!(objectives(&other), objectives(&one), "chunks {chunks:?}");
        assert_eq!(other.labels(), one.labels(), "chunks {chunks:?}");
    }
}

// Design criterion: BatchIncomplete is recoverable, and the finished run is bitwise equal to an uninterrupted one
#[test]
fn short_batch_update_is_recoverable() {
    let (builder, x, init, d) = three_block_fixture();
    let uninterrupted = train(&builder, &x, d, &init, &[usize::MAX]);

    let mut trainer = builder.begin(x.len() / d, &init).unwrap();
    let batch = trainer.batch_rows();
    let short = BLOCK_ROWS + 10;
    trainer.accumulate(&x[..short * d]).unwrap();
    assert_eq!(
        trainer.update(),
        Err(Error::BatchIncomplete {
            have: short,
            want: batch
        })
    );
    assert_eq!(trainer.step, 0);
    trainer.accumulate(&x[short * d..batch * d]).unwrap();
    trainer.update().unwrap();
    let rows = trainer.recipe().epoch_rows[0];
    let mut fed = batch;
    while fed < rows {
        let want = batch.min(rows - fed);
        trainer.accumulate(&x[fed * d..(fed + want) * d]).unwrap();
        trainer.update().unwrap();
        fed += want;
    }
    trainer.end_epoch().unwrap();
    let rest = trainer.recipe().epoch_rows[1..].to_vec();
    for rows in rest {
        let mut fed = 0;
        while fed < rows {
            let want = batch.min(rows - fed);
            trainer.accumulate(&x[fed * d..(fed + want) * d]).unwrap();
            trainer.update().unwrap();
            fed += want;
        }
        trainer.end_epoch().unwrap();
    }
    let resumed = trainer.finalize().unwrap();
    assert_eq!(bits(resumed.centroids()), bits(uninterrupted.centroids()));
    assert_eq!(objectives(&resumed), objectives(&uninterrupted));
}

// Design criterion: a NaN row is NonFiniteInput from accumulate, found before anything changes
#[test]
fn non_finite_row_is_rejected_before_anything_changes() {
    let builder = EMeans::new(2, 2).epochs(1.0).batch_size(4).alpha0(0.5);
    let mut trainer = builder.begin(4, &[0.0, 0.0, 10.0, 10.0]).unwrap();
    trainer.accumulate(&[1.0, 1.0]).unwrap();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let err = trainer.accumulate(&[2.0, 2.0, 3.0, bad]).unwrap_err();
        assert!(
            matches!(err, Error::NonFiniteInput(ref m) if m.contains("row 1 column 1")),
            "{err}"
        );
    }
    assert_eq!(trainer.received(), 1);
    trainer.accumulate(&[2.0, 2.0, 9.0, 9.0, 11.0, 11.0]).unwrap();
    trainer.update().unwrap();
    trainer.end_epoch().unwrap();
    let model = trainer.finalize().unwrap();
    assert_eq!(model.centroids(), &[1.5, 1.5, 10.0, 10.0]);
}

#[test]
fn overflowing_or_ragged_chunk_is_rejected_unchanged() {
    let builder = EMeans::new(2, 2).epochs(1.0).batch_size(2).alpha0(0.5);
    let mut trainer = builder.begin(3, &[0.0, 0.0, 10.0, 10.0]).unwrap();
    assert!(matches!(
        trainer.accumulate(&[1.0, 2.0, 3.0]),
        Err(Error::DimensionMismatch(_))
    ));
    trainer.accumulate(&[1.0, 1.0]).unwrap();
    assert_eq!(
        trainer.accumulate(&[1.0, 1.0, 2.0, 2.0]),
        Err(Error::BatchOverflow { chunk: 2, remaining: 1 })
    );
    trainer.accumulate(&[2.0, 2.0]).unwrap();
    assert_eq!(
        trainer.accumulate(&[3.0, 3.0]),
        Err(Error::BatchOverflow { chunk: 1, remaining: 0 })
    );
    trainer.update().unwrap();
    // The tail batch of the pass is one row.
    assert_eq!(
        trainer.accumulate(&[3.0, 3.0, 4.0, 4.0]),
        Err(Error::BatchOverflow { chunk: 2, remaining: 1 })
    );
    assert_eq!(trainer.received(), 0);
}

#[test]
fn epoch_and_schedule_sequencing_errors() {
    let builder = EMeans::new(1, 1).epochs(2.0).batch_size(2).alpha0(0.5);
    let mut trainer = builder.begin(2, &[0.0]).unwrap();
    trainer.accumulate(&[1.0, 3.0]).unwrap();
    let err = trainer.end_epoch().unwrap_err();
    assert!(
        matches!(err, Error::EpochIncomplete(ref m) if m.contains("0 of 2 rows")),
        "{err}"
    );
    trainer.update().unwrap();
    assert!(matches!(trainer.update(), Err(Error::ScheduleComplete(ref m)) if m.contains("call end_epoch")));
    assert!(matches!(trainer.accumulate(&[1.0]), Err(Error::ScheduleComplete(_))));
    trainer.end_epoch().unwrap();
    // Twice in a row: the next pass has no rows applied.
    let err = trainer.end_epoch().unwrap_err();
    assert!(
        matches!(err, Error::EpochIncomplete(ref m) if m.contains("pass 2 of 2")),
        "{err}"
    );
    trainer.accumulate(&[1.0, 3.0]).unwrap();
    trainer.update().unwrap();
    trainer.end_epoch().unwrap();
    assert!(matches!(trainer.accumulate(&[1.0]), Err(Error::ScheduleComplete(ref m)) if m.contains("all 2")));
    assert!(matches!(trainer.update(), Err(Error::ScheduleComplete(_))));
    assert!(matches!(trainer.end_epoch(), Err(Error::ScheduleComplete(_))));
    assert_eq!(trainer.epoch_stats.len(), 2);
    assert_eq!(trainer.finalize().unwrap().centroids(), &[2.0]);
}

#[test]
fn finalize_before_the_last_pass_ends_is_epoch_incomplete() {
    let builder = EMeans::new(1, 1).epochs(1.0).batch_size(2).alpha0(0.5);
    let mut trainer = builder.begin(2, &[0.0]).unwrap();
    trainer.accumulate(&[1.0, 3.0]).unwrap();
    trainer.update().unwrap();
    let err = trainer.finalize().unwrap_err();
    assert!(
        matches!(err, Error::EpochIncomplete(ref m) if m.contains("pass 1 of 1 has not ended")),
        "{err}"
    );
}

#[test]
fn non_finite_state_poisons_the_trainer() {
    let builder = EMeans::new(1, 1).epochs(2.0).batch_size(1).alpha0(0.5);
    let mut trainer = builder.begin(1, &[-3e38]).unwrap();
    trainer.accumulate(&[3e38]).unwrap();
    assert!(matches!(trainer.update(), Err(Error::NonFiniteState(_))));
    assert!(trainer.poisoned);
    assert_eq!(trainer.accumulate(&[0.0]), Err(Error::Poisoned));
    assert_eq!(trainer.update(), Err(Error::Poisoned));
    assert_eq!(trainer.end_epoch(), Err(Error::Poisoned));
    assert_eq!(trainer.current_centroids(), Err(Error::Poisoned));
    assert_eq!(trainer.finalize(), Err(Error::Poisoned));
}

#[test]
fn update_with_equals_update_on_the_same_statistics() {
    let (builder, x, init, d) = three_block_fixture();
    let mut fed = builder.begin(x.len() / d, &init).unwrap();
    let mut given = builder.begin(x.len() / d, &init).unwrap();
    let b = fed.batch_rows();
    fed.accumulate(&x[..b * d]).unwrap();
    let counts = fed.batch.counts.clone();
    let sums = fed.batch.sums.clone();
    let objective = fed.batch.objective;
    fed.update().unwrap();
    given
        .update_with(BatchStats {
            sums: &sums,
            counts: &counts,
            objective,
        })
        .unwrap();
    assert_eq!(bits(&given.centroids), bits(&fed.centroids));
    assert_eq!(bits(&given.usage), bits(&fed.usage));
    assert_eq!((given.step, given.pass_rows), (fed.step, fed.pass_rows));
    assert_eq!(given.pass_objective, fed.pass_objective);
}

#[test]
fn update_with_rejects_inconsistent_statistics_unchanged() {
    let builder = EMeans::new(1, 2).epochs(1.0).batch_size(2).alpha0(0.5);
    let mut trainer = builder.begin(2, &[0.0, 10.0]).unwrap();
    let cases: [(&[f64], &[u64], f64, &str); 7] = [
        (&[1.0], &[2, 0], 0.0, "sums has 1"),
        (&[1.0, 0.0], &[2], 0.0, "counts has 1"),
        (&[1.0, 0.0], &[1, 0], 0.0, "add up to Some(1)"),
        (&[1.0, 0.0], &[u64::MAX, 3], 0.0, "add up to None"),
        (&[f64::NAN, 0.0], &[2, 0], 0.0, "hold NaN"),
        (&[1.0, 5.0], &[2, 0], 0.0, "no rows but a sum"),
        (&[1.0, 0.0], &[2, 0], f64::INFINITY, "objective is inf"),
    ];
    for (sums, counts, objective, want) in cases {
        let err = trainer
            .update_with(BatchStats {
                sums,
                counts,
                objective,
            })
            .unwrap_err();
        assert!(
            matches!(err, Error::InvalidStatistics(ref m) if m.contains(want)),
            "{err} !~ {want}"
        );
    }
    let err = trainer
        .update_with(BatchStats {
            sums: &[1e300, 0.0],
            counts: &[2, 0],
            objective: 0.0,
        })
        .unwrap_err();
    assert!(
        matches!(err, Error::InvalidStatistics(ref m) if m.contains("overflows f32")),
        "{err}"
    );
    trainer.accumulate(&[1.0]).unwrap();
    let err = trainer
        .update_with(BatchStats {
            sums: &[2.0, 0.0],
            counts: &[2, 0],
            objective: 0.0,
        })
        .unwrap_err();
    assert!(
        matches!(err, Error::InvalidStatistics(ref m) if m.contains("already accumulated")),
        "{err}"
    );
    assert_eq!((trainer.step, trainer.centroids.clone()), (0, vec![0.0, 10.0]));
}

#[test]
fn update_with_in_the_last_pass_drops_labels() {
    let builder = EMeans::new(1, 1).epochs(1.0).batch_size(1).alpha0(0.5);
    let mut trainer = builder.begin(2, &[0.0]).unwrap();
    trainer.accumulate(&[1.0]).unwrap();
    trainer.update().unwrap();
    trainer
        .update_with(BatchStats {
            sums: &[3.0],
            counts: &[1],
            objective: 9.0,
        })
        .unwrap();
    trainer.end_epoch().unwrap();
    let model = trainer.finalize().unwrap();
    assert_eq!(model.labels(), None);
    assert_eq!(model.centroids(), &[2.0]);
    assert_eq!(model.epoch_stats()[0].objective, 10.0);
}

fn splits(model: &Model) -> usize {
    model.epoch_stats().iter().map(|s| s.splits).sum()
}

// Faiss tests/test_e_means.py:417-429 `test_revival_starts_on_fifth_update`. Replaces Phase 3's
// `revival_hook_runs_without_splitting`, which pinned the no-op hook on this same fixture.
#[test]
fn revival_starts_on_fifth_update() {
    let x = [0.0f32; 10];
    let builder = |epochs| {
        EMeans::new(1, 2)
            .epochs(epochs)
            .batch_size(10)
            .alpha0(0.5)
            .unused_threshold(0.1)
    };
    let before = train(&builder(4.0), &x, 1, &[0.0, 10.0], &[10]);
    assert_eq!(splits(&before), 0, "{TEST}:424");
    let on_fifth = train(&builder(5.0), &x, 1, &[0.0, 10.0], &[10]);
    assert_eq!(splits(&on_fifth), 1, "{TEST}:429");
    assert_eq!(on_fifth.epoch_stats()[4].splits, 1, "the split lands in pass 5");
}

// Faiss tests/test_e_means.py:431-442 `test_revival_follows_current_batch_update`: the 10.0 row arrives
// in the fifth batch, so the fifth update makes cluster 1 live before revival checks usage. The
// reference relies on the d = 1 default tau of 0 (no revival at all); the explicit tau = 0.1 run makes
// the order observable: revival before the update would see cluster 1 dead and split.
#[test]
fn revival_follows_current_batch_update() {
    let x = [0.0f32, 0.0, 0.0, 0.0, 10.0];
    let builder = EMeans::new(1, 2).epochs(1.0).batch_size(1).alpha0(0.5);
    let model = train(&builder, &x, 1, &[0.0, 10.0], &[1]);
    assert_eq!(splits(&model), 0, "{TEST}:442");
    let model = train(&builder.unused_threshold(0.1), &x, 1, &[0.0, 10.0], &[1]);
    assert_eq!(splits(&model), 0, "{TEST}:442 with tau = 0.1");
}

// Faiss tests/e_means_test_utils.py:1112-1148 `test_closing_statistics_survive_revival_and_object_deletion`
#[test]
fn closing_statistics_survive_revival() {
    let mut trainer = EMeans::new(1, 2)
        .epochs(1.0)
        .batch_size(1)
        .alpha0(0.5)
        .unused_threshold(0.1)
        .begin(5, &[0.0, 100.0])
        .unwrap();
    for value in 0..5 {
        trainer
            .update_with(BatchStats {
                sums: &[f64::from(value), 0.0],
                counts: &[1, 0],
                objective: 1.0,
            })
            .unwrap();
    }
    assert_eq!(trainer.pass_splits, 1, "{UTILS}:1132");
    let revived = trainer.centroids[1];
    assert_ne!(revived, 100.0);
    trainer.end_epoch().unwrap();
    let model = trainer.finalize().unwrap();
    assert_eq!(model.centroids(), &[2.0, revived], "{UTILS}:1137");
    assert_eq!(model.epoch_stats()[0].objective, 5.0, "{UTILS}:1139");
    assert_eq!(model.epoch_stats()[0].imbalance_factor, 2.0, "{UTILS}:1141");
}

// Revival step 8 and the error contract: a split child past f32::MAX poisons the trainer.
#[test]
fn split_overflow_is_non_finite_state_then_poisoned() {
    let builder = EMeans::new(1, 2)
        .epochs(1.0)
        .batch_size(1)
        .alpha0(0.5)
        .unused_threshold(0.1);
    let mut trainer = builder.begin(5, &[0.0, f32::MAX]).unwrap();
    for _ in 0..4 {
        trainer.accumulate(&[f32::MAX]).unwrap();
        trainer.update().unwrap();
    }
    trainer.accumulate(&[f32::MAX]).unwrap();
    let err = trainer.update().unwrap_err();
    assert!(
        matches!(err, Error::NonFiniteState(ref m) if m.contains("centroid 1 into 0")),
        "{err}"
    );
    assert_eq!(trainer.centroids, vec![0.0, f32::MAX], "nothing of the split committed");
    assert_eq!(trainer.update(), Err(Error::Poisoned));
    assert_eq!(trainer.accumulate(&[0.0]), Err(Error::Poisoned));
    assert_eq!(trainer.finalize(), Err(Error::Poisoned));
}

#[test]
fn the_revival_stream_follows_the_seed() {
    let first = |builder: EMeans| builder.begin(2, &[0.0, 1.0]).unwrap().split_rng.next_u64();
    let default = first(EMeans::new(1, 2));
    assert_eq!(default, Stream::new(1234, StreamId::Revival).next_u64());
    assert_eq!(first(EMeans::new(1, 2).seed(1234)), default);
    assert_ne!(first(EMeans::new(1, 2).seed(1235)), default);
}

// EMA step 2: with alpha == 1 there is no decay, so an untouched cluster keeps its mass
#[test]
fn alpha_one_keeps_untouched_mass() {
    let builder = EMeans::new(1, 2).epochs(2.0).batch_size(4).alpha0(1.0);
    let mut trainer = builder.begin(4, &[0.0, 100.0]).unwrap();
    trainer.accumulate(&[0.0, 1.0, 2.0, 3.0]).unwrap();
    trainer.update().unwrap();
    assert_eq!(trainer.usage, vec![4.0, INIT_USAGE]);
    assert_eq!(trainer.centroids, vec![1.5, 100.0]);
}

fn assert_close(got: &[f32], want: &[f32], atol: f32, context: &str) {
    assert_eq!(got.len(), want.len(), "{context}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= atol,
            "{context}: value {i} is {g}, want {w} (atol {atol})"
        );
    }
}

// Design criterion: the reference spherical fixture gives objective 7.5 and labels [0, 1, 1, 2, 0, 0].
// Faiss tests/e_means_test_utils.py:118-166 `test_spherical_ip_zero_objective_and_represented_statistics`
#[test]
fn spherical_ip_zero_objective_and_represented_statistics() {
    let initial = [0.0, 0.0, 1.0, 0.0, -1.0, 0.0];
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
    let expected_labels = [0u32, 1, 1, 2, 0, 0];
    let builder = EMeans::new(2, 3).epochs(1.0).batch_size(6).alpha0(1.0).spherical(true);
    let mut trainer = builder.begin(6, &initial).unwrap();
    for chunk in rows.chunks(2 * 2) {
        trainer.accumulate(chunk).unwrap();
    }
    assert_eq!(trainer.pass_labels.as_deref().unwrap(), expected_labels, "{UTILS}:142");
    assert_eq!(trainer.batch.counts, [3, 2, 1], "{UTILS}:148");
    let mut sums = [0f64; 6];
    for (row, &label) in rows.as_chunks::<2>().0.iter().zip(&expected_labels) {
        let j = label as usize;
        sums[2 * j] += f64::from(row[0]);
        sums[2 * j + 1] += f64::from(row[1]);
    }
    assert_eq!(trainer.batch.sums, sums, "{UTILS}:149");
    assert!(
        (trainer.batch.objective - 7.5).abs() < 1e-6,
        "{UTILS}:150: {}",
        trainer.batch.objective
    );
    trainer.update().unwrap();
    trainer.end_epoch().unwrap();
    let model = trainer.finalize().unwrap();
    let labels: Vec<u32> = model.labels().unwrap().iter().map(|l| l.unwrap()).collect();
    assert_eq!(labels, expected_labels);
    let fin = model.centroids();
    assert_eq!(&fin[..2], &[0.0, 0.0], "{UTILS}:154 zero mean stays zero");

    let assignment = model.assign(&rows).unwrap();
    for (i, row) in rows.as_chunks::<2>().0.iter().enumerate() {
        let scores: Vec<f32> = fin
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| row[0] * c[0] + row[1] * c[1])
            .collect();
        let best = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let argmax = scores.iter().position(|&s| s == best).unwrap() as u32;
        assert_eq!(assignment.labels[i], argmax, "{UTILS}:157 row {i}");
        let want = 2.0 - 2.0 * best;
        assert!(
            (assignment.distances[i] - want).abs() <= 4e-7,
            "{UTILS}:160 (D10: 2 - 2 * ip) row {i}: {} vs {want}",
            assignment.distances[i]
        );
    }
    assert!((model.epoch_stats()[0].objective - 7.5).abs() < 1e-6, "{UTILS}:164");
}

// Faiss tests/e_means_test_utils.py:273-290 `test_spherical_zero_beats_negative_similarity`
#[test]
fn spherical_zero_beats_negative_similarity() {
    let builder = EMeans::new(2, 2).epochs(1.0).batch_size(2).alpha0(1.0).spherical(true);
    let mut trainer = builder.begin(2, &[0.0, 0.0, 1.0, 0.0]).unwrap();
    trainer.accumulate(&[-1.0, 0.0, 0.0, 0.0]).unwrap();
    assert_eq!(trainer.pass_labels.as_deref().unwrap(), [0, 0], "{UTILS}:289");
    assert_eq!(trainer.batch.objective, 4.0, "{UTILS}:290");
}

/// The spherical fixture of `UTILS:168-226`: d = 2, k = 1, two batches of 3 rows, alpha0 0.25.
fn deferred_ema_trainer() -> Trainer {
    EMeans::new(2, 1)
        .epochs(1.0)
        .batch_size(3)
        .alpha0(0.25)
        .spherical(true)
        .begin(6, &[3.0, 4.0])
        .unwrap()
}

// Faiss tests/e_means_test_utils.py:168-226 `test_spherical_deferred_ema_borrowed_cache_and_chunks`.
// The bite check for the readout refresh: skipping it leaves the readout at the initial direction.
#[test]
fn spherical_deferred_ema_keeps_raw_state_and_refreshes_readout() {
    let rows = [1.0, 0.0, 0.0, 1.0, -1.0, 0.0];
    let mut model = deferred_ema_trainer();
    let mut control = deferred_ema_trainer();
    assert_close(
        &model.centroids,
        &[0.6, 0.8],
        1e-7,
        "begin normalizes the initial centroids",
    );
    let raw = model.centroids.clone();
    let usage = model.usage[0];
    let before = model.snapshot.clone();
    model.accumulate(&rows[..2]).unwrap();
    assert_eq!(model.snapshot, before, "{UTILS}:190 readout frozen inside the batch");
    model.accumulate(&rows[2..]).unwrap();
    assert_eq!(model.snapshot, before, "{UTILS}:198");
    assert!(
        (model.batch.objective - 4.4).abs() < 5e-6,
        "{UTILS}:199: {}",
        model.batch.objective
    );

    let mass = 0.75f32 * usage + 0.25 * 3.0;
    let weight = 0.25f32 * 3.0 / mass;
    let mean = [0.0f32, 1.0 / 3.0];
    let expected: Vec<f32> = raw.iter().zip(mean).map(|(&c, m)| c + weight * (m - c)).collect();
    model.update().unwrap();
    control.accumulate(&rows).unwrap();
    control.update().unwrap();
    assert_close(
        &model.centroids,
        &expected,
        1e-7,
        "tests/e_means_test_utils.py:206 raw EMA state",
    );
    let norm = expected.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        model.centroids.iter().map(|v| v * v).sum::<f32>().sqrt() < 0.999,
        "{UTILS}:209 EMA state stays un-normalized"
    );
    let unit: Vec<f32> = expected.iter().map(|v| v / norm).collect();
    assert_close(
        &model.snapshot,
        &unit,
        1e-7,
        "tests/e_means_test_utils.py:212 readout refreshed after the update",
    );

    for trainer in [&mut model, &mut control] {
        trainer.accumulate(&rows).unwrap();
        trainer.update().unwrap();
        trainer.end_epoch().unwrap();
    }
    let (model, control) = (model.finalize().unwrap(), control.finalize().unwrap());
    assert_eq!(bits(model.centroids()), bits(control.centroids()), "{UTILS}:221");
    assert_eq!(objectives(&model), objectives(&control), "{UTILS}:224");
}

// Faiss tests/e_means_test_utils.py:292-325 `test_spherical_torch_version_and_inference_invalidation`, the
// half that is not about Torch version counters: the readout follows the update.
#[test]
fn spherical_readout_follows_the_update() {
    let builder = EMeans::new(2, 1).epochs(1.0).batch_size(2).alpha0(1.0).spherical(true);
    let mut trainer = builder.begin(2, &[1.0, 0.0]).unwrap();
    assert_eq!(trainer.snapshot, [1.0, 0.0]);
    trainer.accumulate(&[0.0, -1.0, 0.0, -1.0]).unwrap();
    trainer.update().unwrap();
    assert_eq!(trainer.snapshot, [0.0, -1.0], "{UTILS}:319");
}

// Faiss tests/e_means_test_utils.py:260-271 (`test_spherical_projection_extremes_and_wide_closing`, closing
// half), through the same direct seam: pass statistics set by hand. Closing takes the mean in f64, rounds it to
// f32, then projects in f32 with the max-abs prescale; a cluster with no rows keeps its EMA position, normalized.
#[test]
fn spherical_closing_projects_means_and_empty_clusters() {
    let rows = [1.0, 0.0, 0.0, 1.0, -1.0, 0.0];
    let builder = EMeans::new(2, 3).epochs(1.0).batch_size(3).alpha0(1.0).spherical(true);
    let mut trainer = builder.begin(3, &rows).unwrap();
    trainer.accumulate(&rows).unwrap();
    trainer.update().unwrap();
    trainer.end_epoch().unwrap();
    trainer.pass_sums = vec![6e38, -6e38, 0.0, 0.0, 0.0, 0.0];
    trainer.pass_counts = vec![2, 1, 0];
    trainer.centroids[4..].copy_from_slice(&[-3.0, 0.0]);
    let model = trainer.finalize().unwrap();
    assert_close(
        model.centroids(),
        &[FRAC_1_SQRT_2, -FRAC_1_SQRT_2, 0.0, 0.0, -1.0, 0.0],
        1e-7,
        "tests/e_means_test_utils.py:266",
    );
}

// Euclidean mode is untouched by the spherical machinery: the snapshot is the raw centroids, no projection.
#[test]
fn euclidean_mode_does_not_normalize() {
    let builder = EMeans::new(2, 1).epochs(1.0).batch_size(2).alpha0(1.0);
    let mut trainer = builder.begin(2, &[3.0, 4.0]).unwrap();
    assert_eq!(trainer.centroids, [3.0, 4.0]);
    trainer.accumulate(&[0.0, 2.0, 0.0, 2.0]).unwrap();
    trainer.update().unwrap();
    trainer.end_epoch().unwrap();
    assert_eq!(trainer.finalize().unwrap().centroids(), &[0.0, 2.0]);
}

// Faiss `current_centroids` (`REF/e_means.py:243-258`): the EMA state in Euclidean mode, the unit-norm readout
// in spherical mode, current after each update. This is what a caller of `update_with` assigns against.
#[test]
fn current_centroids_track_updates_in_both_modes() {
    let builder = EMeans::new(2, 1).epochs(2.0).batch_size(2).alpha0(1.0);
    let mut trainer = builder.begin(2, &[3.0, 4.0]).unwrap();
    assert_eq!(trainer.current_centroids().unwrap(), &[3.0, 4.0]);
    trainer.accumulate(&[0.0, 2.0, 0.0, 2.0]).unwrap();
    trainer.update().unwrap();
    assert_eq!(
        trainer.current_centroids().unwrap(),
        &[0.0, 2.0],
        "Euclidean: the moved EMA state"
    );

    let builder = EMeans::new(2, 1).epochs(2.0).batch_size(2).alpha0(1.0).spherical(true);
    let mut trainer = builder.begin(2, &[3.0, 4.0]).unwrap();
    assert_close(
        trainer.current_centroids().unwrap(),
        &[0.6, 0.8],
        1e-7,
        "spherical: normalized init",
    );
    trainer.accumulate(&[0.0, 2.0, 0.0, 2.0]).unwrap();
    trainer.update().unwrap();
    assert_eq!(trainer.centroids, [0.0, 2.0], "the EMA state stays un-normalized");
    assert_eq!(
        trainer.current_centroids().unwrap(),
        &[0.0, 1.0],
        "spherical: the refreshed readout"
    );
}
