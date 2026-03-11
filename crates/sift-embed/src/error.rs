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

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Display impls (derived by thiserror)
    // -----------------------------------------------------------------------

    #[test]
    fn display_model_not_found() {
        let err = EmbeddingError::ModelNotFound {
            name: "some-model".into(),
        };
        assert_eq!(err.to_string(), "Model not found: some-model");
    }

    #[test]
    fn display_onnx_runtime() {
        let err = EmbeddingError::OnnxRuntime("session init failed".into());
        assert_eq!(err.to_string(), "ONNX runtime error: session init failed");
    }

    #[test]
    fn display_tokenization() {
        let err = EmbeddingError::Tokenization("invalid token".into());
        assert_eq!(err.to_string(), "Tokenization failed: invalid token");
    }

    #[test]
    fn display_dimension_mismatch() {
        let err = EmbeddingError::DimensionMismatch {
            expected: 768,
            actual: 512,
        };
        assert_eq!(err.to_string(), "Dimension mismatch: expected 768, got 512");
    }

    #[test]
    fn display_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file gone");
        let err = EmbeddingError::Io(io_err);
        assert!(err.to_string().contains("IO error"));
        assert!(err.to_string().contains("file gone"));
    }

    // -----------------------------------------------------------------------
    // From<EmbeddingError> for SiftError
    // -----------------------------------------------------------------------

    #[test]
    fn from_embedding_error_model_not_found() {
        let embed_err = EmbeddingError::ModelNotFound {
            name: "test".into(),
        };
        let sift_err: sift_core::SiftError = embed_err.into();
        match sift_err {
            sift_core::SiftError::Embedding(msg) => {
                assert!(msg.contains("Model not found: test"));
            }
            other => panic!("expected SiftError::Embedding, got {other:?}"),
        }
    }

    #[test]
    fn from_embedding_error_onnx_runtime() {
        let embed_err = EmbeddingError::OnnxRuntime("broken".into());
        let sift_err: sift_core::SiftError = embed_err.into();
        match sift_err {
            sift_core::SiftError::Embedding(msg) => {
                assert!(msg.contains("ONNX runtime error: broken"));
            }
            other => panic!("expected SiftError::Embedding, got {other:?}"),
        }
    }

    #[test]
    fn from_embedding_error_tokenization() {
        let embed_err = EmbeddingError::Tokenization("bad input".into());
        let sift_err: sift_core::SiftError = embed_err.into();
        match sift_err {
            sift_core::SiftError::Embedding(msg) => {
                assert!(msg.contains("Tokenization failed: bad input"));
            }
            other => panic!("expected SiftError::Embedding, got {other:?}"),
        }
    }

    #[test]
    fn from_embedding_error_dimension_mismatch() {
        let embed_err = EmbeddingError::DimensionMismatch {
            expected: 1024,
            actual: 256,
        };
        let sift_err: sift_core::SiftError = embed_err.into();
        match sift_err {
            sift_core::SiftError::Embedding(msg) => {
                assert!(msg.contains("expected 1024, got 256"));
            }
            other => panic!("expected SiftError::Embedding, got {other:?}"),
        }
    }

    #[test]
    fn from_embedding_error_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let embed_err = EmbeddingError::Io(io_err);
        let sift_err: sift_core::SiftError = embed_err.into();
        match sift_err {
            sift_core::SiftError::Embedding(msg) => {
                assert!(msg.contains("IO error"));
                assert!(msg.contains("denied"));
            }
            other => panic!("expected SiftError::Embedding, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // From<std::io::Error> for EmbeddingError
    // -----------------------------------------------------------------------

    #[test]
    fn from_io_error_to_embedding_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::BrokenPipe, "pipe");
        let embed_err: EmbeddingError = io_err.into();
        assert!(matches!(embed_err, EmbeddingError::Io(_)));
    }
}
