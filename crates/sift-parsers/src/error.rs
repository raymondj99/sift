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
