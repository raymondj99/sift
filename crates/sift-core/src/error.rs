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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_io_error() {
        let err = SiftError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "gone"));
        assert!(err.to_string().contains("IO error"));
        assert!(err.to_string().contains("gone"));
    }

    #[test]
    fn display_config_error() {
        let err = SiftError::Config("bad key".into());
        assert_eq!(err.to_string(), "Config error: bad key");
    }

    #[test]
    fn display_parse_error() {
        let err = SiftError::Parse {
            path: "foo.txt".into(),
            message: "invalid utf8".into(),
        };
        assert_eq!(err.to_string(), "Parse error for foo.txt: invalid utf8");
    }

    #[test]
    fn display_embedding_error() {
        let err = SiftError::Embedding("dim mismatch".into());
        assert_eq!(err.to_string(), "Embedding error: dim mismatch");
    }

    #[test]
    fn display_storage_error() {
        let err = SiftError::Storage("disk full".into());
        assert_eq!(err.to_string(), "Storage error: disk full");
    }

    #[test]
    fn display_search_error() {
        let err = SiftError::Search("no index".into());
        assert_eq!(err.to_string(), "Search error: no index");
    }

    #[test]
    fn display_model_error() {
        let err = SiftError::Model("not found".into());
        assert_eq!(err.to_string(), "Model error: not found");
    }

    #[test]
    fn display_source_error() {
        let err = SiftError::Source("unreachable".into());
        assert_eq!(err.to_string(), "Source error: unreachable");
    }

    #[test]
    fn display_partial_error() {
        let errs = vec![SiftError::Config("a".into()), SiftError::Config("b".into())];
        let err = SiftError::Partial(errs);
        assert_eq!(err.to_string(), "Multiple errors (2 total)");
    }

    #[test]
    fn display_with_path_error() {
        let inner = SiftError::Storage("corrupt".into());
        let err = inner.with_path("/tmp/bad");
        assert!(err.to_string().contains("/tmp/bad"));
        assert!(err.to_string().contains("Storage error: corrupt"));
    }

    #[test]
    fn with_path_wraps_error() {
        let inner = SiftError::Config("oops".into());
        let err = inner.with_path("/a/b/c");
        match &err {
            SiftError::WithPath { path, source } => {
                assert_eq!(path, &PathBuf::from("/a/b/c"));
                assert!(matches!(source.as_ref(), SiftError::Config(_)));
            }
            _ => panic!("expected WithPath variant"),
        }
    }

    #[test]
    fn is_partial_returns_true_for_partial() {
        let err = SiftError::Partial(vec![SiftError::Config("x".into())]);
        assert!(err.is_partial());
    }

    #[test]
    fn is_partial_returns_false_for_non_partial() {
        assert!(!SiftError::Config("x".into()).is_partial());
        assert!(!SiftError::Storage("x".into()).is_partial());
        assert!(!SiftError::Model("x".into()).is_partial());
        assert!(!SiftError::Search("x".into()).is_partial());
        assert!(!SiftError::Source("x".into()).is_partial());
        assert!(!SiftError::Embedding("x".into()).is_partial());
    }

    #[test]
    fn exit_code_config() {
        assert_eq!(SiftError::Config("x".into()).exit_code(), 3);
    }

    #[test]
    fn exit_code_model() {
        assert_eq!(SiftError::Model("x".into()).exit_code(), 4);
    }

    #[test]
    fn exit_code_storage() {
        assert_eq!(SiftError::Storage("x".into()).exit_code(), 5);
    }

    #[test]
    fn exit_code_others_return_one() {
        assert_eq!(SiftError::Search("x".into()).exit_code(), 1);
        assert_eq!(SiftError::Embedding("x".into()).exit_code(), 1);
        assert_eq!(SiftError::Source("x".into()).exit_code(), 1);
        assert_eq!(SiftError::Partial(vec![]).exit_code(), 1);
    }

    #[test]
    fn from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "nope");
        let sift_err: SiftError = io_err.into();
        assert!(matches!(sift_err, SiftError::Io(_)));
        assert_eq!(sift_err.exit_code(), 1);
    }

    #[test]
    fn from_anyhow_error() {
        let anyhow_err = anyhow::anyhow!("something went wrong");
        let sift_err: SiftError = anyhow_err.into();
        assert!(matches!(sift_err, SiftError::Other(_)));
        assert!(sift_err.to_string().contains("something went wrong"));
    }
}
