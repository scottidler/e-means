//! e-means: a Rust port of the e-means streaming k-means trainer from Faiss.
//!
//! ```
//! use e_means::EMeans;
//!
//! // Two well-separated 1-d blobs, rows alternating between them.
//! let x: Vec<f32> = (0..200).map(|i| if i % 2 == 0 { (i % 7) as f32 * 0.1 } else { 50.0 + (i % 7) as f32 * 0.1 }).collect();
//! let model = EMeans::new(1, 2).epochs(5.0).seed(1).train(&x)?;
//! assert_eq!(model.centroids().len(), 2);
//! let found = model.assign(&[0.2, 50.2])?;
//! assert_ne!(found.labels[0], found.labels[1]);
//! # Ok::<(), e_means::Error>(())
//! ```

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
mod spherical;
mod trainer;

pub use config::EMeans;
pub use error::{Error, Result};
pub use model::{Assignment, EpochStats, Model};
pub use recipe::{BatchLaw, Recipe};
pub use trainer::{BatchStats, Trainer};
