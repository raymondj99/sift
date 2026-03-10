use sift_core::{EmbeddedChunk, SearchResult, SiftResult};

use crate::flat::ExportEntry;

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
}
