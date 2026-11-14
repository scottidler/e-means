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
    #[error("non-finite state: {0}")]
    NonFiniteState(String),
    #[error("trainer is poisoned by an earlier non-finite state")]
    Poisoned,
    #[error("batch incomplete: {have} of {want} rows")]
    BatchIncomplete { have: usize, want: usize },
    #[error("batch overflow: chunk of {chunk} rows exceeds the {remaining} remaining in the batch")]
    BatchOverflow { chunk: usize, remaining: usize },
    #[error("epoch incomplete: {0}")]
    EpochIncomplete(String),
    #[error("schedule complete: {0}")]
    ScheduleComplete(String),
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
    fn sequencing_variants_carry_their_reason() {
        let e = Error::ScheduleComplete("all 3 passes have ended".to_string());
        assert_eq!(e.to_string(), "schedule complete: all 3 passes have ended");
        let e = Error::EpochIncomplete("pass 1 of 2 has 4 of 31 rows applied".to_string());
        assert!(e.to_string().starts_with("epoch incomplete: pass 1"));
        assert!(
            Error::NonFiniteState("x".to_string())
                .to_string()
                .starts_with("non-finite state")
        );
    }
}
