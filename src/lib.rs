//! e-means: a Rust port of the e-means streaming k-means trainer from Faiss.

#![deny(clippy::unwrap_used)]
#![deny(dead_code)]
#![deny(unused_variables)]

mod error;

pub use error::{Error, Result};
