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
}
