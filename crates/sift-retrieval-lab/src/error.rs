//! Typed errors for the retrieval lab.
//!
//! Library functions return [`RetrievalLabResult<T>`]. Binaries use
//! `anyhow::Result` and propagate via `?`, matching the workspace
//! convention: typed errors at library boundaries, untyped at the CLI
//! edge where output is for humans.

use std::path::PathBuf;
use thiserror::Error;

pub type RetrievalLabResult<T> = Result<T, RetrievalLabError>;

#[derive(Error, Debug)]
pub enum RetrievalLabError {
    #[error("i/o error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("json parse error at {path}: {source}")]
    JsonParse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("json serialize error: {0}")]
    JsonSerialize(#[from] serde_json::Error),

    /// Corpus file declared a `schema_version` the lab does not understand.
    /// Refusing to load is the correct response: silent coercion of qrels
    /// would invalidate every committed baseline.
    #[error("corpus schema version mismatch at {path}: file is v{found}, lab expects v{expected}")]
    CorpusSchemaMismatch {
        path: PathBuf,
        found: u32,
        expected: u32,
    },

    /// Structural defects in a loaded corpus: duplicate doc ids, qrels
    /// referencing absent docs, empty queries list. Detected by
    /// [`crate::corpus::RetrievalCorpus::validate`] so downstream cells
    /// never run against a broken substrate.
    #[error("corpus validation failed at {path}: {reason}")]
    CorpusInvalid { path: PathBuf, reason: String },

    #[error("invalid cell config: {0}")]
    InvalidCellConfig(String),

    /// The configured model is not present in the local model directory.
    /// Suggests the precise remediation rather than failing on a generic
    /// "file not found" later.
    #[error("model {model_name} not downloaded; run `sift models download {model_name}`")]
    ModelNotDownloaded { model_name: String },

    #[error("model manager init failed: {0}")]
    ModelManager(#[source] sift_core::SiftError),

    #[error("embedder load failed for {model_name}: {source}")]
    EmbedderLoad {
        model_name: String,
        #[source]
        source: sift_core::SiftError,
    },

    #[error("embedding failed: {0}")]
    EmbeddingFailed(#[source] sift_core::SiftError),

    #[error("storage operation failed: {0}")]
    Storage(#[source] sift_core::SiftError),

    #[error("search failed: {0}")]
    Search(#[source] sift_core::SiftError),
}

impl RetrievalLabError {
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    pub fn parse(path: impl Into<PathBuf>, source: serde_json::Error) -> Self {
        Self::JsonParse {
            path: path.into(),
            source,
        }
    }
}
