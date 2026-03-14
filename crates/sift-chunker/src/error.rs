use thiserror::Error;

#[derive(Error, Debug)]
pub enum ChunkerError {
    #[error("Invalid chunk configuration: {0}")]
    InvalidConfig(String),
    #[error("AST parse failed for {language}: {detail}")]
    AstParseFailed { language: String, detail: String },
}

impl From<ChunkerError> for sift_core::SiftError {
    fn from(e: ChunkerError) -> Self {
        sift_core::SiftError::Parse {
            path: String::new(),
            message: e.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunker_error_converts_to_sift_error() {
        let chunker_err = ChunkerError::InvalidConfig("bad config".into());
        let sift_err: sift_core::SiftError = chunker_err.into();
        match sift_err {
            sift_core::SiftError::Parse { path, message } => {
                assert!(path.is_empty(), "path should be empty");
                assert!(
                    message.contains("bad config"),
                    "message should contain original error"
                );
            }
            other => panic!("expected Parse variant, got: {other:?}"),
        }
    }
}
