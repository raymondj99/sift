use thiserror::Error;

#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("Model not found: {name}")]
    ModelNotFound { name: String },
    #[error("ONNX runtime error: {0}")]
    OnnxRuntime(String),
    #[error("Tokenization failed: {0}")]
    Tokenization(String),
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<EmbeddingError> for sift_core::SiftError {
    fn from(e: EmbeddingError) -> Self {
        sift_core::SiftError::Embedding(e.to_string())
    }
}
