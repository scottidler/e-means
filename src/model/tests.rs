use super::*;
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
    let model = Model::new(vec![1.0, 11.0], vec![stats()], recipe(), Some(labels.clone()));
    assert_eq!(model.centroids(), &[1.0, 11.0]);
    assert_eq!(model.epoch_stats(), &[stats()]);
    assert_eq!(model.recipe().batch_size, 4);
    assert_eq!(model.labels(), Some(&labels[..]));
}

#[test]
fn labels_are_none_when_not_captured() {
    let model = Model::new(vec![1.0], vec![], recipe(), None);
    assert_eq!(model.labels(), None);
}
