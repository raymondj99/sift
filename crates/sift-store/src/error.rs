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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_sqlite_error() {
        // rusqlite::Error::InvalidQuery is a simple variant that doesn't need a connection
        let err = StorageError::Sqlite(rusqlite::Error::InvalidQuery);
        let msg = err.to_string();
        assert!(msg.contains("SQLite error"), "got: {msg}");
    }

    #[test]
    fn display_index_not_found() {
        let err = StorageError::IndexNotFound {
            path: PathBuf::from("/tmp/missing"),
        };
        assert_eq!(err.to_string(), "Index not found: /tmp/missing");
    }

    #[test]
    fn display_corrupt_index() {
        let err = StorageError::CorruptIndex {
            path: PathBuf::from("/data/idx"),
            detail: "bad magic bytes".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Corrupt index at /data/idx: bad magic bytes"
        );
    }

    #[test]
    fn display_lock_contention() {
        let err = StorageError::LockContention {
            path: PathBuf::from("/data/lock"),
        };
        assert_eq!(err.to_string(), "Lock contention on /data/lock");
    }

    #[test]
    fn display_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file gone");
        let err = StorageError::Io(io_err);
        let msg = err.to_string();
        assert!(msg.contains("IO error"), "got: {msg}");
        assert!(msg.contains("file gone"), "got: {msg}");
    }

    #[test]
    fn from_storage_error_to_sift_error() {
        let storage_err = StorageError::IndexNotFound {
            path: PathBuf::from("/missing"),
        };
        let sift_err: sift_core::SiftError = storage_err.into();
        match sift_err {
            sift_core::SiftError::Storage(msg) => {
                assert!(msg.contains("Index not found"), "got: {msg}");
                assert!(msg.contains("/missing"), "got: {msg}");
            }
            other => panic!("Expected SiftError::Storage, got: {other:?}"),
        }
    }

    #[test]
    fn from_corrupt_index_to_sift_error() {
        let storage_err = StorageError::CorruptIndex {
            path: PathBuf::from("/bad"),
            detail: "header invalid".to_string(),
        };
        let sift_err: sift_core::SiftError = storage_err.into();
        match sift_err {
            sift_core::SiftError::Storage(msg) => {
                assert!(msg.contains("Corrupt index"), "got: {msg}");
            }
            other => panic!("Expected SiftError::Storage, got: {other:?}"),
        }
    }
}
