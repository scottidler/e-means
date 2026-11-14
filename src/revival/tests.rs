#![allow(clippy::unwrap_used)]

use super::*;
use crate::rng::StreamId;

const UTILS: &str = "tests/e_means_test_utils.py";

fn stream(seed: u64) -> Stream {
    Stream::new(seed, StreamId::Revival)
}

/// u64 words drawn from a fresh revival stream of `seed` so far.
fn draws(rng: &Stream, seed: u64) -> usize {
    let mut fresh = stream(seed);
    let mut probe = rng.clone();
    let next = probe.next_u64();
    (0..).find(|_| fresh.next_u64() == next).unwrap_or(usize::MAX)
}

fn assert_rel(got: f32, want: f32, rtol: f32, at: &str) {
    assert!((got - want).abs() <= rtol * want.abs(), "{at}: got {got}, want {want}");
}

// Faiss tests/e_means_test_utils.py:992-1036 `test_revival_threshold_separation_and_conservation`
#[test]
fn threshold_bit_patterns() {
    let before = [27.715767f32, 0.38280505, 803.37445];
    let actual = threshold(&before, 0.1, 3);
    let expected = (before[0] + before[1] + before[2]) * 0.1f32 / 3.0f32;
    assert_eq!(actual, expected, "{UTILS}:1025");
    assert_eq!(actual.to_bits(), 1105050084, "{UTILS}:1027");
    assert_eq!(
        threshold(&[69938690.0], 0.1, 52359).to_bits(),
        1124438855,
        "{UTILS}:1033"
    );
    let dead: Vec<bool> = before.iter().map(|&u| u < actual).collect();
    assert_eq!(dead, [false, true, false], "{UTILS}:1036");
}

// Faiss tests/e_means_test_utils.py:1037-1084: one split, inferred donor, 1e-4 displacement, mass dropped
#[test]
fn split_revives_the_dead_cluster_and_drops_its_mass() {
    let before = [27.715767f32, 0.38280505, 803.37445];
    let mut usage = before;
    let mut centroids = [10.0f32, 20.0, 30.0];
    let weighted: f32 = centroids.iter().zip(&before).map(|(c, u)| c * u).sum();
    let discarded = centroids[1] * before[1];
    assert_eq!(
        revive(&mut usage, &mut centroids, 1, 0.1, &mut stream(123)),
        Ok(1),
        "{UTILS}:1050"
    );
    let sources: Vec<usize> = (0..3).filter(|&j| j != 1 && usage[j] != before[j]).collect();
    assert_eq!(sources.len(), 1, "{UTILS}:1055");
    let source = sources[0];
    let initial = [10.0f32, 20.0, 30.0][source];
    assert_rel(centroids[source], initial * 0.9999, 1e-6, "{UTILS}:1061");
    assert_rel(centroids[1], initial * 1.0001, 1e-6, "{UTILS}:1067");
    let other = if source == 0 { 2 } else { 0 };
    assert_eq!(centroids[other], [10.0, 20.0, 30.0][other], "{UTILS}:1073");
    let total: f32 = usage.iter().sum();
    let want: f32 = before.iter().sum::<f32>() - before[1];
    assert_rel(total, want, 1e-6, "{UTILS}:1076");
    let after: f32 = centroids.iter().zip(&usage).map(|(c, u)| c * u).sum();
    assert_rel(after, weighted - discarded, 1e-6, "{UTILS}:1083");
}

// Faiss tests/e_means_test_utils.py:1085-1098: equal donor and child separate by one ULP of coordinate 0
#[test]
fn separation_moves_collapsed_children_one_ulp() {
    let rows = [
        ([0.0f32, 0.0], [0.0f32, 0.0]),
        ([2.0, 3.0], [2.0002, 2.9997]),
        ([1.0, 0.0], [1.0, 0.0]),
    ];
    let mut children = Vec::new();
    for (donor, child) in rows {
        let mut child = child;
        separate(&donor, &mut child);
        children.push(child);
    }
    assert_eq!(children[0][0], 0.0f32.next_up(), "{UTILS}:1092");
    assert_eq!(children[0][0].to_bits(), 1, "smallest subnormal");
    assert_eq!(children[2][0], 1.0f32.next_up(), "{UTILS}:1096");
    assert_eq!(children[1], [2.0002, 2.9997], "{UTILS}:1098");
    assert_eq!(children[2][1], 0.0);
}

#[test]
fn separation_steps_down_from_f32_max() {
    let mut child = [f32::MAX];
    separate(&[f32::MAX], &mut child);
    assert_eq!(child, [f32::MAX.next_down()]);
}

// Faiss tests/e_means_test_utils.py:1099-1110: one donor, two dead, one revived; zero donor separates
#[test]
fn one_donor_revives_one_of_two_dead() {
    let mut usage = [10.0f32, 0.0, 0.0];
    let mut centroids = [0.0f32, 0.0, 10.0, 10.0, 20.0, 20.0];
    assert_eq!(
        revive(&mut usage, &mut centroids, 2, 0.1, &mut stream(123)),
        Ok(1),
        "{UTILS}:1104"
    );
    assert_eq!(usage, [5.0, 5.0, 0.0], "{UTILS}:1106");
    assert_ne!(centroids[..2], centroids[2..4], "{UTILS}:1109");
    assert_eq!(centroids[4..], [20.0, 20.0], "{UTILS}:1111");
}

// Phase 4 criterion: usage [10, 0, 0] -> [5, 5, 0]; donor 0.9999x / child 1.0001x on alternating coordinates
#[test]
fn split_offsets_alternate_by_coordinate() {
    let mut usage = [10.0f32, 0.0, 0.0];
    let mut centroids = [10.0f32, 20.0, 30.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0];
    assert_eq!(revive(&mut usage, &mut centroids, 3, 0.1, &mut stream(1234)), Ok(1));
    assert_eq!(usage, [5.0, 5.0, 0.0]);
    let donor = [10.0f32 * 0.9999, 20.0 * 1.0001, 30.0 * 0.9999];
    let child = [10.0f32 * 1.0001, 20.0 * 0.9999, 30.0 * 1.0001];
    for i in 0..3 {
        assert_rel(centroids[i], donor[i], 1e-6, "donor");
        assert_rel(centroids[3 + i], child[i], 1e-6, "child");
    }
    assert_eq!(centroids[6..], [20.0, 20.0, 20.0]);
}

// Revival step 1: `dead = usage < threshold` is strict, so a mass exactly at the threshold is alive.
#[test]
fn mass_at_the_threshold_is_alive() {
    let mut usage = [2.0f32, 1.0, 0.0];
    let mut centroids = [5.0f32, 10.0, 15.0];
    assert_eq!(threshold(&usage, 1.0, 3), 1.0);
    assert_eq!(
        revive(&mut usage, &mut centroids, 1, 1.0, &mut stream(1234)),
        Ok(1),
        "only cluster 2 is dead; cluster 1 sits exactly at the threshold"
    );
    assert!(usage[2] > 0.0, "cluster 2 was revived");
}

#[test]
fn several_splits_pair_largest_keys_with_ascending_dead() {
    let mut usage = [0.0f32, 8.0, 0.0, 4.0, 0.0];
    let mut centroids = [0.0f32, 1.0, 2.0, 3.0, 4.0];
    assert_eq!(revive(&mut usage, &mut centroids, 1, 0.5, &mut stream(1234)), Ok(2));
    let mut received = [usage[0], usage[2]];
    received.sort_by(f32::total_cmp);
    assert_eq!(received, [2.0, 4.0], "targets are the first two dead clusters");
    assert_eq!((usage[4], centroids[4]), (0.0, 4.0), "third dead cluster untouched");
    assert_eq!((usage[1], usage[3]), (4.0, 2.0));
}

#[test]
fn no_pair_draws_no_random_number() {
    let cases: [([f32; 3], f32); 3] = [
        ([1.0, 0.0, 5.0], 0.0),  // tau 0 disables revival
        ([1.0, 1.0, 1.0], 0.1),  // no dead cluster
        ([1.0, 1.0, 1.0], 10.0), // every cluster dead, no donor
    ];
    for (usage, tau) in cases {
        let (mut u, mut c) = (usage, [1.0f32, 2.0, 3.0]);
        let mut rng = stream(5);
        assert_eq!(revive(&mut u, &mut c, 1, tau, &mut rng), Ok(0));
        assert_eq!((u, c), (usage, [1.0, 2.0, 3.0]));
        assert_eq!(draws(&rng, 5), 0, "{usage:?} tau {tau}");
    }
}

#[test]
fn a_revival_event_draws_one_uniform_per_cluster() {
    let mut usage = [10.0f32, 0.0, 3.0, 4.0];
    let mut centroids = [1.0f32, 2.0, 3.0, 4.0];
    let mut rng = stream(5);
    assert_eq!(revive(&mut usage, &mut centroids, 1, 0.1, &mut rng), Ok(1));
    assert_eq!(draws(&rng, 5), 4);
}

// Gumbel top-k samples donors in proportion to mass: P(donor 0) = 3 / (3 + 1).
#[test]
fn donors_are_drawn_in_proportion_to_mass() {
    let trials = 4000;
    let mut rng = stream(42);
    let mut heavy = 0;
    for _ in 0..trials {
        let mut usage = [3.0f32, 1.0, 0.0];
        let mut centroids = [1.0f32, 2.0, 3.0];
        assert_eq!(revive(&mut usage, &mut centroids, 1, 0.01, &mut rng), Ok(1));
        if usage[0] == 1.5 {
            heavy += 1;
        }
    }
    assert!((2850..=3150).contains(&heavy), "heavy donor chosen {heavy} of {trials}");
}

// Revival step 8: a child outside f32 range commits nothing.
#[test]
fn non_finite_child_commits_nothing() {
    let mut usage = [10.0f32, 0.0];
    let mut centroids = [f32::MAX, 1.0];
    let err = revive(&mut usage, &mut centroids, 1, 0.1, &mut stream(1)).unwrap_err();
    assert!(
        matches!(err, Error::NonFiniteState(ref m) if m.contains("centroid 0 into 1")),
        "{err}"
    );
    assert_eq!((usage, centroids), ([10.0, 0.0], [f32::MAX, 1.0]));
}
