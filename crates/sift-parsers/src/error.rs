use thiserror::Error;

#[derive(Error, Debug)]
pub enum ParseError {
    #[error("Unsupported format: {mime_type}")]
    UnsupportedFormat { mime_type: String },
    #[error("Extraction failed for {path}: {detail}")]
    ExtractionFailed { path: String, detail: String },
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<ParseError> for sift_core::SiftError {
    fn from(e: ParseError) -> Self {
        match &e {
            ParseError::ExtractionFailed { path, .. } => sift_core::SiftError::Parse {
                path: path.clone(),
                message: e.to_string(),
            },
            _ => sift_core::SiftError::Parse {
                path: String::new(),
                message: e.to_string(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_error_display_unsupported() {
        let err = ParseError::UnsupportedFormat {
            mime_type: "application/x-unknown".to_string(),
        };
        let display = format!("{err}");
        assert!(display.contains("Unsupported format"));
        assert!(display.contains("application/x-unknown"));
    }

    #[test]
    fn test_parse_error_display_extraction_failed() {
        let err = ParseError::ExtractionFailed {
            path: "/some/file.txt".to_string(),
            detail: "corrupt data".to_string(),
        };
        let display = format!("{err}");
        assert!(display.contains("Extraction failed"));
        assert!(display.contains("/some/file.txt"));
        assert!(display.contains("corrupt data"));
    }

    #[test]
    fn test_parse_error_display_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = ParseError::Io(io_err);
        let display = format!("{err}");
        assert!(display.contains("IO error"));
        assert!(display.contains("file not found"));
    }

    #[test]
    fn test_parse_error_into_sift_error_extraction() {
        let err = ParseError::ExtractionFailed {
            path: "/test/path.bin".to_string(),
            detail: "bad bytes".to_string(),
        };
        let sift_err: sift_core::SiftError = err.into();
        match sift_err {
            sift_core::SiftError::Parse { path, message } => {
                assert_eq!(path, "/test/path.bin");
                assert!(message.contains("bad bytes"));
            }
            other => panic!("Expected Parse variant, got: {other:?}"),
        }
    }

    #[test]
    fn test_parse_error_into_sift_error_unsupported() {
        let err = ParseError::UnsupportedFormat {
            mime_type: "video/mp4".to_string(),
        };
        let sift_err: sift_core::SiftError = err.into();
        match sift_err {
            sift_core::SiftError::Parse { path, message } => {
                assert!(path.is_empty());
                assert!(message.contains("video/mp4"));
            }
            other => panic!("Expected Parse variant, got: {other:?}"),
        }
    }

    #[test]
    fn test_parse_error_into_sift_error_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
        let err = ParseError::Io(io_err);
        let sift_err: sift_core::SiftError = err.into();
        match sift_err {
            sift_core::SiftError::Parse { path, message } => {
                assert!(path.is_empty());
                assert!(message.contains("access denied"));
            }
            other => panic!("Expected Parse variant, got: {other:?}"),
        }
    }

    #[test]
    fn test_parse_error_from_io_error() {
        let io_err = std::io::Error::other("disk failure");
        let parse_err: ParseError = io_err.into();
        match parse_err {
            ParseError::Io(e) => assert_eq!(e.to_string(), "disk failure"),
            other => panic!("Expected Io variant, got: {other:?}"),
        }
    }
}
