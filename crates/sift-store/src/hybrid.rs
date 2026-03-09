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

/// Reciprocal Rank Fusion - merge two ranked lists into one.
///
/// RRF score = alpha * 1/(k+rank_vector) + (1-alpha) * 1/(k+rank_bm25)
/// where k=60 is a standard constant.
fn rrf_fuse(
    vector_results: &[SearchResult],
    bm25_results: &[SearchResult],
    alpha: f32,
    top_k: usize,
) -> Vec<SearchResult> {
    const K: f32 = 60.0;

    // Build a map of URI+chunk_index → (best SearchResult, RRF score)
    let mut scores: HashMap<String, (SearchResult, f32)> = HashMap::new();

    // Process vector results
    for (rank, result) in vector_results.iter().enumerate() {
        let key = format!("{}:{}", result.uri, result.chunk_index);
        let rrf_score = alpha / (K + rank as f32 + 1.0);

        scores
            .entry(key)
            .and_modify(|(_, score)| *score += rrf_score)
            .or_insert_with(|| (result.clone(), rrf_score));
    }

    // Process BM25 results
    for (rank, result) in bm25_results.iter().enumerate() {
        let key = format!("{}:{}", result.uri, result.chunk_index);
        let rrf_score = (1.0 - alpha) / (K + rank as f32 + 1.0);

        scores
            .entry(key)
            .and_modify(|(_, score)| *score += rrf_score)
            .or_insert_with(|| (result.clone(), rrf_score));
    }

    // Sort by fused score descending
    let mut fused: Vec<(SearchResult, f32)> = scores.into_values().collect();
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Return top-k with fused scores
    fused
        .into_iter()
        .take(top_k)
        .map(|(mut result, score)| {
            result.score = score;
            result
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
    fn test_rrf_fuse_vector_only() {
        let vector = vec![make_result("a", 0.9), make_result("b", 0.8)];
        let bm25 = vec![];

        let fused = rrf_fuse(&vector, &bm25, 1.0, 10);
        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].uri, "a");
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
    fn test_rrf_fuse_respects_top_k() {
        let vector = vec![
            make_result("a", 0.9),
            make_result("b", 0.8),
            make_result("c", 0.7),
        ];
        let bm25 = vec![];

        let fused = rrf_fuse(&vector, &bm25, 1.0, 2);
        assert_eq!(fused.len(), 2);
    }
}
