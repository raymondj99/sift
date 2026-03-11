use sift_core::{EmbeddedChunk, SearchResult, SiftResult};

use crate::flat::ExportEntry;

/// Push-model search result consumer.
/// Return `true` to continue receiving results, `false` to stop early.
pub trait SearchSink {
    fn on_result(&mut self, result: &SearchResult) -> bool;
    fn on_complete(&mut self, _total: usize) {}
}

/// Convenience sink that collects results into a Vec.
pub struct CollectSink {
    pub results: Vec<SearchResult>,
}

impl CollectSink {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }
}

impl Default for CollectSink {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchSink for CollectSink {
    fn on_result(&mut self, result: &SearchResult) -> bool {
        self.results.push(result.clone());
        true
    }
}

/// A vector similarity search store.
pub trait VectorStore: Send + Sync {
    /// Insert embedded chunks into the store.
    fn insert(&self, chunks: &[EmbeddedChunk]) -> SiftResult<()>;

    /// Search by vector similarity. Returns top-k results with scores.
    fn search(&self, query_vector: &[f32], top_k: usize) -> SiftResult<Vec<SearchResult>>;

    /// Delete all chunks from a given source URI.
    fn delete_by_uri(&self, uri: &str) -> SiftResult<u64>;

    /// Total number of chunks stored.
    fn count(&self) -> SiftResult<u64>;

    /// Streaming search — pushes results to a sink one at a time.
    fn search_streaming(
        &self,
        query_vector: &[f32],
        top_k: usize,
        sink: &mut dyn SearchSink,
    ) -> SiftResult<()> {
        let results = self.search(query_vector, top_k)?;
        let total = results.len();
        for r in &results {
            if !sink.on_result(r) {
                break;
            }
        }
        sink.on_complete(total);
        Ok(())
    }
}

/// Trait for vector similarity index implementations.
///
/// Extends [`VectorStore`] with persistence and export capabilities.
pub trait VectorIndex: VectorStore {
    /// Persist to disk.
    fn save(&self, path: &std::path::Path) -> SiftResult<()>;

    /// Export all entries.
    fn export_all(&self) -> SiftResult<Vec<ExportEntry>>;
}

/// A full-text (BM25) search store.
pub trait FullTextStore: Send + Sync {
    /// Index chunk text for full-text search.
    fn insert(&self, chunks: &[sift_core::EmbeddedChunk]) -> SiftResult<()>;

    /// Search by keywords (BM25). Returns top-k results with scores.
    fn search(&self, query: &str, top_k: usize) -> SiftResult<Vec<SearchResult>>;

    /// Delete all chunks from a given source URI.
    fn delete_by_uri(&self, uri: &str) -> SiftResult<u64>;

    /// Persist any buffered state to disk. Default is a no-op for stores
    /// that commit on each mutation (e.g. SQLite-backed FTS5, Tantivy).
    fn flush(&self) -> SiftResult<()> {
        Ok(())
    }

    /// Streaming search — pushes results to a sink one at a time.
    fn search_streaming(
        &self,
        query: &str,
        top_k: usize,
        sink: &mut dyn SearchSink,
    ) -> SiftResult<()> {
        let results = self.search(query, top_k)?;
        let total = results.len();
        for r in &results {
            if !sink.on_result(r) {
                break;
            }
        }
        sink.on_complete(total);
        Ok(())
    }
}
