//! e-means: a Rust port of the e-means streaming k-means trainer from Faiss.

#![deny(clippy::unwrap_used)]
#![deny(dead_code)]
#![deny(unused_variables)]

mod assign;
mod config;
mod error;
mod model;
pub mod recipe;
mod revival;
mod rng;
mod trainer;

pub use config::EMeans;
pub use error::{Error, Result};
pub use model::{Assignment, EpochStats, Model};
pub use recipe::{BatchLaw, Recipe};
pub use trainer::{BatchStats, Trainer};
