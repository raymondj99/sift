use std::path::PathBuf;
use thiserror::Error;

pub type SiftResult<T> = Result<T, SiftError>;

#[derive(Error, Debug)]
pub enum SiftError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Config error: {0}")]
    Config(String),

    #[error("Parse error for {path}: {message}")]
    Parse { path: String, message: String },

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Search error: {0}")]
    Search(String),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Source error: {0}")]
    Source(String),

    #[error("{0}")]
    Other(#[from] anyhow::Error),

    #[error("Error processing {path}: {source}")]
    WithPath {
        path: PathBuf,
        #[source]
        source: Box<SiftError>,
    },

    #[error("Multiple errors ({} total)", .0.len())]
    Partial(Vec<SiftError>),
}

impl SiftError {
    pub fn with_path(self, path: impl Into<PathBuf>) -> Self {
        SiftError::WithPath {
            path: path.into(),
            source: Box::new(self),
        }
    }

    pub fn is_partial(&self) -> bool {
        matches!(self, SiftError::Partial(_))
    }

    pub fn exit_code(&self) -> i32 {
        match self {
            SiftError::Config(_) => 3,
            SiftError::Model(_) => 4,
            SiftError::Storage(_) => 5,
            _ => 1,
        }
    }
}
