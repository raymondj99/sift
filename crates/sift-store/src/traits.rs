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

#[cfg(test)]
mod tests {
    use super::*;
    use sift_core::{Chunk, ContentType};

    fn make_chunk(uri: &str, text: &str, idx: u32, vector: Vec<f32>) -> EmbeddedChunk {
        EmbeddedChunk {
            chunk: Chunk {
                text: text.to_string(),
                source_uri: uri.to_string(),
                chunk_index: idx,
                content_type: ContentType::Text,
                file_type: "txt".to_string(),
                title: None,
                language: None,
                byte_range: None,
            },
            vector,
        }
    }

    #[test]
    fn collect_sink_new_starts_empty() {
        let sink = CollectSink::new();
        assert!(sink.results.is_empty());
    }

    #[test]
    fn collect_sink_default_starts_empty() {
        let sink = CollectSink::default();
        assert!(sink.results.is_empty());
    }

    #[test]
    fn collect_sink_on_result_collects_and_returns_true() {
        let mut sink = CollectSink::new();
        let result = SearchResult {
            uri: "file:///a.txt".to_string(),
            text: "hello".to_string(),
            score: 0.9,
            chunk_index: 0,
            content_type: ContentType::Text,
            file_type: "txt".to_string(),
            title: None,
            byte_range: None,
        };

        let cont = sink.on_result(&result);
        assert!(cont, "on_result should return true to continue");
        assert_eq!(sink.results.len(), 1);
        assert_eq!(sink.results[0].uri, "file:///a.txt");

        // Push a second result
        let result2 = SearchResult {
            uri: "file:///b.txt".to_string(),
            text: "world".to_string(),
            score: 0.8,
            chunk_index: 1,
            content_type: ContentType::Text,
            file_type: "txt".to_string(),
            title: None,
            byte_range: None,
        };
        let cont2 = sink.on_result(&result2);
        assert!(cont2);
        assert_eq!(sink.results.len(), 2);
    }

    #[test]
    fn collect_sink_on_complete_is_noop() {
        let mut sink = CollectSink::new();
        // Should not panic or mutate state
        sink.on_complete(42);
        assert!(sink.results.is_empty());
    }

    #[test]
    fn vector_store_search_streaming_default_impl() {
        use crate::flat::FlatVectorIndex;

        let store = FlatVectorIndex::new();
        store
            .insert(&[
                make_chunk("file:///a.txt", "hello", 0, vec![1.0, 0.0, 0.0]),
                make_chunk("file:///b.txt", "world", 0, vec![0.0, 1.0, 0.0]),
                make_chunk("file:///c.txt", "foo", 0, vec![0.5, 0.5, 0.0]),
            ])
            .unwrap();

        let mut sink = CollectSink::new();
        store
            .search_streaming(&[1.0, 0.0, 0.0], 2, &mut sink)
            .unwrap();

        assert_eq!(sink.results.len(), 2);
        // First result should be closest to the query vector
        assert_eq!(sink.results[0].uri, "file:///a.txt");
    }

    #[test]
    fn vector_store_search_streaming_empty_index() {
        use crate::flat::FlatVectorIndex;

        let store = FlatVectorIndex::new();
        let mut sink = CollectSink::new();
        store
            .search_streaming(&[1.0, 0.0, 0.0], 5, &mut sink)
            .unwrap();

        assert!(sink.results.is_empty());
    }

    /// A custom sink that stops after collecting N results, to test early termination.
    struct LimitSink {
        results: Vec<SearchResult>,
        limit: usize,
    }

    impl SearchSink for LimitSink {
        fn on_result(&mut self, result: &SearchResult) -> bool {
            self.results.push(result.clone());
            self.results.len() < self.limit
        }
    }

    #[test]
    fn vector_store_search_streaming_early_stop() {
        use crate::flat::FlatVectorIndex;

        let store = FlatVectorIndex::new();
        store
            .insert(&[
                make_chunk("file:///a.txt", "a", 0, vec![1.0, 0.0, 0.0]),
                make_chunk("file:///b.txt", "b", 0, vec![0.9, 0.1, 0.0]),
                make_chunk("file:///c.txt", "c", 0, vec![0.8, 0.2, 0.0]),
            ])
            .unwrap();

        let mut sink = LimitSink {
            results: Vec::new(),
            limit: 1,
        };
        store
            .search_streaming(&[1.0, 0.0, 0.0], 3, &mut sink)
            .unwrap();

        // The sink returned false after the first result, so only 1 collected.
        assert_eq!(sink.results.len(), 1);
    }

    #[cfg(all(not(feature = "fulltext"), feature = "fts5"))]
    #[test]
    fn fts5_search_streaming_default_impl() {
        use crate::fts5::Fts5Store;

        let store = Fts5Store::open_in_memory().unwrap();

        let chunks = vec![
            make_chunk("file:///a.txt", "the quick brown fox", 0, vec![]),
            make_chunk("file:///b.txt", "the lazy dog sleeps", 0, vec![]),
        ];
        FullTextStore::insert(&store, &chunks).unwrap();

        let mut sink = CollectSink::new();
        FullTextStore::search_streaming(&store, "fox", 10, &mut sink).unwrap();

        assert!(!sink.results.is_empty());
        assert_eq!(sink.results[0].uri, "file:///a.txt");
    }

    #[cfg(feature = "fulltext")]
    #[test]
    fn fulltext_store_flush_default_impl_tantivy() {
        use crate::tantivy_store::TantivyStore;

        let dir = tempfile::TempDir::new().unwrap();
        let store = TantivyStore::open(dir.path()).unwrap();
        // flush() uses the default no-op impl from the FullTextStore trait
        FullTextStore::flush(&store).unwrap();
    }

    #[cfg(all(not(feature = "fulltext"), feature = "fts5"))]
    #[test]
    fn fulltext_store_flush_default_impl_fts5() {
        use crate::fts5::Fts5Store;

        let store = Fts5Store::open_in_memory().unwrap();
        FullTextStore::flush(&store).unwrap();
    }

    #[cfg(feature = "fulltext")]
    #[test]
    fn fulltext_store_search_streaming_tantivy() {
        use crate::tantivy_store::TantivyStore;

        let dir = tempfile::TempDir::new().unwrap();
        let store = TantivyStore::open(dir.path()).unwrap();

        let chunks = vec![
            make_chunk("file:///a.txt", "the quick brown fox", 0, vec![]),
            make_chunk("file:///b.txt", "the lazy dog sleeps", 0, vec![]),
        ];
        FullTextStore::insert(&store, &chunks).unwrap();

        let mut sink = CollectSink::new();
        FullTextStore::search_streaming(&store, "fox", 10, &mut sink).unwrap();

        assert!(!sink.results.is_empty());
        assert_eq!(sink.results[0].uri, "file:///a.txt");
    }

    #[cfg(feature = "fulltext")]
    #[test]
    fn fulltext_store_search_streaming_early_stop_tantivy() {
        use crate::tantivy_store::TantivyStore;

        let dir = tempfile::TempDir::new().unwrap();
        let store = TantivyStore::open(dir.path()).unwrap();

        let chunks = vec![
            make_chunk("file:///a.txt", "rust programming language", 0, vec![]),
            make_chunk("file:///b.txt", "rust systems programming", 0, vec![]),
            make_chunk("file:///c.txt", "rust embedded programming", 0, vec![]),
        ];
        FullTextStore::insert(&store, &chunks).unwrap();

        let mut sink = LimitSink {
            results: Vec::new(),
            limit: 1,
        };
        FullTextStore::search_streaming(&store, "rust programming", 10, &mut sink).unwrap();

        // The sink returned false after the first result, so only 1 collected.
        assert_eq!(sink.results.len(), 1);
    }
}
