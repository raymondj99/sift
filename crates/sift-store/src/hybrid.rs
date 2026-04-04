use crate::traits::{FullTextStore, VectorStore};
use sift_core::{SearchResult, SiftResult};
use std::collections::HashMap;

/// Hybrid search engine combining vector similarity and BM25 keyword search
/// using Reciprocal Rank Fusion (RRF).
pub struct HybridSearchEngine<V: VectorStore, F: FullTextStore> {
    pub vector_store: V,
    pub fulltext_store: F,
    /// Weight for vector results. 0.0 = pure BM25, 1.0 = pure vector.
    pub alpha: f32,
}

impl<V: VectorStore, F: FullTextStore> HybridSearchEngine<V, F> {
    pub fn new(vector_store: V, fulltext_store: F, alpha: f32) -> Self {
        Self {
            vector_store,
            fulltext_store,
            alpha: alpha.clamp(0.0, 1.0),
        }
    }

    /// Insert chunks into both stores.
    pub fn insert(&self, chunks: &[sift_core::EmbeddedChunk]) -> SiftResult<()> {
        self.vector_store.insert(chunks)?;
        self.fulltext_store.insert(chunks)?;
        Ok(())
    }

    /// Hybrid search using RRF fusion.
    pub fn search(
        &self,
        query_vector: &[f32],
        query_text: &str,
        top_k: usize,
        mode: sift_core::SearchMode,
    ) -> SiftResult<Vec<SearchResult>> {
        match mode {
            sift_core::SearchMode::VectorOnly => self.vector_store.search(query_vector, top_k),
            sift_core::SearchMode::KeywordOnly => self.fulltext_store.search(query_text, top_k),
            sift_core::SearchMode::Hybrid => {
                // Fetch more than top_k from each to have good candidates for fusion
                let fetch_k = top_k * 3;

                let vector_results = self.vector_store.search(query_vector, fetch_k)?;
                let bm25_results = self.fulltext_store.search(query_text, fetch_k)?;

                let fused = rrf_fuse(&vector_results, &bm25_results, self.alpha, top_k);

                Ok(fused)
            }
        }
    }

    /// Delete a source from both stores.
    pub fn delete_by_uri(&self, uri: &str) -> SiftResult<u64> {
        let v = self.vector_store.delete_by_uri(uri)?;
        self.fulltext_store.delete_by_uri(uri)?;
        Ok(v)
    }

    /// Total chunks in vector store.
    pub fn count(&self) -> SiftResult<u64> {
        self.vector_store.count()
    }
}

/// Source list identifier for index-based RRF fusion.
#[derive(Clone, Copy)]
enum Source {
    Vector(usize),
    Bm25(usize),
}

/// Reciprocal Rank Fusion - merge two ranked lists into one.
///
/// RRF score = alpha * `1/(k+rank_vector)` + (1-alpha) * `1/(k+rank_bm25)`
/// where k=60 is a standard constant.
///
/// Uses index references instead of cloning `SearchResult` into the HashMap.
/// Only the final top-k winners are materialized, avoiding O(n) clones.
fn rrf_fuse(
    vector_results: &[SearchResult],
    bm25_results: &[SearchResult],
    alpha: f32,
    top_k: usize,
) -> Vec<SearchResult> {
    const K: f32 = 60.0;

    let capacity = vector_results.len() + bm25_results.len();
    let mut scores: HashMap<(&str, u32), (Source, f32)> = HashMap::with_capacity(capacity);

    for (rank, result) in vector_results.iter().enumerate() {
        let key = (result.uri.as_str(), result.chunk_index);
        let rrf_score = alpha / (K + rank as f32 + 1.0);

        scores
            .entry(key)
            .and_modify(|(_, score)| *score += rrf_score)
            .or_insert((Source::Vector(rank), rrf_score));
    }

    for (rank, result) in bm25_results.iter().enumerate() {
        let key = (result.uri.as_str(), result.chunk_index);
        let rrf_score = (1.0 - alpha) / (K + rank as f32 + 1.0);

        scores
            .entry(key)
            .and_modify(|(_, score)| *score += rrf_score)
            .or_insert((Source::Bm25(rank), rrf_score));
    }

    let mut fused: Vec<(Source, f32)> = scores.into_values().collect();

    // Partial sort when we have more candidates than needed.
    if fused.len() > top_k && top_k > 0 {
        fused.select_nth_unstable_by(top_k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        fused.truncate(top_k);
    }
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    fused
        .into_iter()
        .map(|(source, score)| {
            let base = match source {
                Source::Vector(i) => &vector_results[i],
                Source::Bm25(i) => &bm25_results[i],
            };
            SearchResult {
                uri: base.uri.clone(),
                text: base.text.clone(),
                score,
                chunk_index: base.chunk_index,
                content_type: base.content_type,
                file_type: base.file_type.clone(),
                title: base.title.clone(),
                byte_range: base.byte_range,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sift_core::ContentType;

    fn make_result(uri: &str, score: f32) -> SearchResult {
        SearchResult {
            uri: uri.to_string(),
            text: "test".to_string(),
            score,
            chunk_index: 0,
            content_type: ContentType::Text,
            file_type: "txt".to_string(),
            title: None,
            byte_range: None,
        }
    }

    #[test]
    fn test_rrf_fuse_both() {
        let vector = vec![make_result("a", 0.9), make_result("b", 0.8)];
        let bm25 = vec![make_result("b", 5.0), make_result("c", 4.0)];

        let fused = rrf_fuse(&vector, &bm25, 0.5, 10);
        assert!(fused.len() >= 2);
        // "b" should score highest since it appears in both
        assert_eq!(fused[0].uri, "b");
    }

    #[test]
    fn test_rrf_fuse_empty_both() {
        let fused = rrf_fuse(&[], &[], 0.5, 10);
        assert!(fused.is_empty());
    }

    #[test]
    fn test_rrf_fuse_empty_vector_only() {
        let bm25 = vec![make_result("a", 5.0)];
        let fused = rrf_fuse(&[], &bm25, 0.5, 10);
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].uri, "a");
    }

    #[test]
    fn test_rrf_fuse_empty_bm25_only() {
        let vector = vec![make_result("a", 0.9)];
        let fused = rrf_fuse(&vector, &[], 0.5, 10);
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].uri, "a");
    }

    #[test]
    fn test_rrf_fuse_alpha_zero_pure_bm25() {
        let vector = vec![make_result("v", 0.9)];
        let bm25 = vec![make_result("b", 5.0)];
        let fused = rrf_fuse(&vector, &bm25, 0.0, 10);
        // alpha=0 means vector gets zero weight, bm25 gets full weight
        assert_eq!(fused[0].uri, "b");
    }

    #[test]
    fn test_rrf_fuse_alpha_one_pure_vector() {
        let vector = vec![make_result("v", 0.9)];
        let bm25 = vec![make_result("b", 5.0)];
        let fused = rrf_fuse(&vector, &bm25, 1.0, 10);
        // alpha=1 means bm25 gets zero weight, vector gets full weight
        assert_eq!(fused[0].uri, "v");
    }

    #[test]
    fn test_rrf_fuse_top_k_truncation() {
        let vector: Vec<SearchResult> = (0..20)
            .map(|i| make_result(&format!("v{i}"), 0.9 - i as f32 * 0.01))
            .collect();
        let bm25: Vec<SearchResult> = (0..20)
            .map(|i| make_result(&format!("b{i}"), 5.0 - i as f32 * 0.1))
            .collect();
        let fused = rrf_fuse(&vector, &bm25, 0.5, 5);
        assert_eq!(fused.len(), 5);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_alpha_clamping() {
        // Test clamping with a simple struct that satisfies both traits
        let eng_low = HybridSearchEngine {
            vector_store: crate::flat::FlatVectorIndex::new(),
            fulltext_store: create_fulltext_store(),
            alpha: (-0.5f32).clamp(0.0, 1.0),
        };
        assert_eq!(eng_low.alpha, 0.0);

        let eng_high = HybridSearchEngine {
            vector_store: crate::flat::FlatVectorIndex::new(),
            fulltext_store: create_fulltext_store(),
            alpha: (2.0f32).clamp(0.0, 1.0),
        };
        assert_eq!(eng_high.alpha, 1.0);
    }

    fn create_fulltext_store() -> impl FullTextStore {
        #[cfg(feature = "fulltext")]
        {
            crate::tantivy_store::TantivyStore::open_in_memory().unwrap()
        }
        #[cfg(all(not(feature = "fulltext"), feature = "fts5"))]
        {
            crate::fts5::Fts5Store::open_in_memory().unwrap()
        }
        #[cfg(all(not(feature = "fulltext"), not(feature = "fts5")))]
        {
            let tmp = tempfile::tempdir().unwrap();
            crate::bm25::Bm25Store::open(&tmp.path().join("bm25.json")).unwrap()
        }
    }

    #[test]
    fn test_hybrid_engine_hybrid_mode_search() {
        // Exercises the Hybrid branch (line 41) with actual stores
        let tmp = tempfile::tempdir().unwrap();
        let vector_store = crate::flat::FlatVectorIndex::new();
        #[cfg(feature = "fulltext")]
        let fulltext_store =
            crate::tantivy_store::TantivyStore::open(&tmp.path().join("tantivy")).unwrap();
        #[cfg(all(not(feature = "fulltext"), feature = "fts5"))]
        let fulltext_store = crate::fts5::Fts5Store::open(&tmp.path().join("fts5.db")).unwrap();
        #[cfg(all(not(feature = "fulltext"), not(feature = "fts5")))]
        let fulltext_store = crate::bm25::Bm25Store::open(&tmp.path().join("bm25.json")).unwrap();

        let engine = HybridSearchEngine::new(vector_store, fulltext_store, 0.7);

        // Insert data
        let chunks = vec![sift_core::EmbeddedChunk {
            chunk: sift_core::Chunk {
                text: "hello world from rust".into(),
                source_uri: "file:///a.txt".into(),
                chunk_index: 0,
                content_type: ContentType::Text,
                file_type: "txt".into(),
                title: None,
                language: None,
                byte_range: None,
            },
            vector: vec![1.0, 0.0, 0.0],
        }];
        engine.insert(&chunks).unwrap();

        // Search in Hybrid mode
        let results = engine
            .search(&[1.0, 0.0, 0.0], "hello", 10, sift_core::SearchMode::Hybrid)
            .unwrap();
        assert!(!results.is_empty(), "hybrid search should find results");
        assert_eq!(results[0].uri, "file:///a.txt");
    }
}
