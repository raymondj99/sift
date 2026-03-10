use sift_core::{ScanOptions, SiftResult, SourceItem};

/// A source of data to be indexed.
pub trait Source: Send + Sync {
    /// Discover items from this source.
    fn discover(&self, options: &ScanOptions) -> SiftResult<Vec<SourceItem>>;
}
