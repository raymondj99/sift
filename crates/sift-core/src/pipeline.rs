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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn scan_stats_default_all_zeros() {
        let stats = ScanStats::default();
        assert_eq!(stats.discovered, 0);
        assert_eq!(stats.skipped, 0);
        assert_eq!(stats.indexed, 0);
        assert_eq!(stats.chunks, 0);
        assert_eq!(stats.errors, 0);
        assert_eq!(stats.cache_hits, 0);
        assert!(stats.file_types.is_empty());
    }

    #[test]
    fn noop_progress_does_not_panic() {
        let p = NoopProgress;
        let path = Path::new("/tmp/test.txt");
        let stats = ScanStats::default();

        p.on_file_discovered(path);
        p.on_file_skipped(path);
        p.on_file_parsed(path, 5);
        p.on_file_embedded(path);
        p.on_file_stored(path);
        p.on_file_error(path, "boom");
        p.on_scan_complete(&stats);
    }
}
