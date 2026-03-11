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
