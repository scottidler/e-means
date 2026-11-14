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
