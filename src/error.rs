use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum Error {
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),
    #[error("too few rows: {rows} rows for {k} centroids")]
    TooFewRows { rows: usize, k: usize },
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("non-finite input: {0}")]
    NonFiniteInput(String),
    #[error("non-finite state: a split child overflowed")]
    NonFiniteState,
    #[error("trainer is poisoned by an earlier non-finite state")]
    Poisoned,
    #[error("batch incomplete: {have} of {want} rows")]
    BatchIncomplete { have: usize, want: usize },
    #[error("batch overflow: chunk of {chunk} rows exceeds the {remaining} remaining in the batch")]
    BatchOverflow { chunk: usize, remaining: usize },
    #[error("epoch incomplete: {missing} rows missing")]
    EpochIncomplete { missing: usize },
    #[error("schedule complete: no steps remain")]
    ScheduleComplete,
    #[error("invalid statistics: {0}")]
    InvalidStatistics(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fails() -> Result<()> {
        Err(Error::Poisoned)
    }

    #[test]
    fn display_carries_the_cause() {
        let e = Error::TooFewRows { rows: 3, k: 5 };
        assert_eq!(e.to_string(), "too few rows: 3 rows for 5 centroids");
    }

    #[test]
    fn result_alias_propagates_error() {
        assert_eq!(fails(), Err(Error::Poisoned));
    }

    #[test]
    fn unit_variants_render_fixed_messages() {
        assert!(Error::ScheduleComplete.to_string().contains("no steps remain"));
        assert!(Error::NonFiniteState.to_string().contains("overflowed"));
    }
}
