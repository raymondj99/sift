use std::path::Path;

/// Progress reporting trait for pipeline consumers.
///
/// Each method has a default no-op implementation so callers only need
/// to override the events they care about.
pub trait ProgressSink: Send + Sync {
    fn on_file_discovered(&self, _path: &Path) {}
    fn on_file_skipped(&self, _path: &Path) {}
    fn on_file_parsed(&self, _path: &Path, _chunks: usize) {}
    fn on_file_embedded(&self, _path: &Path) {}
    fn on_file_stored(&self, _path: &Path) {}
    fn on_file_error(&self, _path: &Path, _error: &str) {}
    fn on_scan_complete(&self, _stats: &ScanStats) {}
}

/// No-op progress sink for library usage / testing.
pub struct NoopProgress;
impl ProgressSink for NoopProgress {}

/// Summary statistics from a scan operation.
#[derive(Debug, Default, Clone)]
pub struct ScanStats {
    pub discovered: u64,
    pub skipped: u64,
    pub indexed: u64,
    pub chunks: u64,
    pub errors: u64,
    pub cache_hits: u64,
    pub file_types: std::collections::HashMap<String, u64>,
}
