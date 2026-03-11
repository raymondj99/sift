use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("Index not found: {path}")]
    IndexNotFound { path: PathBuf },
    #[error("Corrupt index at {path}: {detail}")]
    CorruptIndex { path: PathBuf, detail: String },
    #[error("Lock contention on {path}")]
    LockContention { path: PathBuf },
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<StorageError> for sift_core::SiftError {
    fn from(e: StorageError) -> Self {
        sift_core::SiftError::Storage(e.to_string())
    }
}
