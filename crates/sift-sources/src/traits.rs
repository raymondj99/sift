use sift_core::{ScanOptions, SiftResult, SourceItem};

/// A source of data to be indexed.
pub trait Source: Send + Sync {
    /// Discover items from this source, returning them via callback.
    fn discover(
        &self,
        options: &ScanOptions,
        callback: &mut dyn FnMut(SourceItem) -> SiftResult<()>,
    ) -> SiftResult<u64>;
}
