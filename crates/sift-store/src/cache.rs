use lru::LruCache;
use sift_core::{SearchMode, SearchResult, SiftResult};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::hybrid::HybridSearchEngine;
use crate::traits::{FullTextStore, VectorStore};

struct CacheEntry {
    results: Vec<SearchResult>,
    inserted_at: Instant,
}

/// A caching wrapper around [`HybridSearchEngine`] that stores recent search
/// results in an LRU cache with a configurable TTL.
pub struct CachedSearchEngine<V: VectorStore, F: FullTextStore> {
    inner: HybridSearchEngine<V, F>,
    cache: Mutex<LruCache<u64, CacheEntry>>,
    ttl: Duration,
}

fn cache_key(query_text: &str, top_k: usize, mode: SearchMode) -> u64 {
    let mut hasher = DefaultHasher::new();
    query_text.hash(&mut hasher);
    top_k.hash(&mut hasher);
    let mode_tag: u8 = match mode {
        SearchMode::Hybrid => 0,
        SearchMode::VectorOnly => 1,
        SearchMode::KeywordOnly => 2,
    };
    mode_tag.hash(&mut hasher);
    hasher.finish()
}

impl<V: VectorStore, F: FullTextStore> CachedSearchEngine<V, F> {
    pub fn new(inner: HybridSearchEngine<V, F>, capacity: usize, ttl: Duration) -> Self {
        Self {
            inner,
            cache: Mutex::new(LruCache::new(
                NonZeroUsize::new(capacity).expect("capacity must be > 0"),
            )),
            ttl,
        }
    }

    pub fn search(
        &self,
        query_vector: &[f32],
        query_text: &str,
        top_k: usize,
        mode: SearchMode,
    ) -> SiftResult<Vec<SearchResult>> {
        let key = cache_key(query_text, top_k, mode);

        {
            let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(entry) = cache.get(&key) {
                if entry.inserted_at.elapsed() < self.ttl {
                    return Ok(entry.results.clone());
                }
                // TTL expired
            }
            // Remove expired entry outside the immutable borrow
            cache.pop(&key);
        }

        let results = self.inner.search(query_vector, query_text, top_k, mode)?;

        {
            let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            cache.put(
                key,
                CacheEntry {
                    results: results.clone(),
                    inserted_at: Instant::now(),
                },
            );
        }

        Ok(results)
    }

    pub fn insert(&self, chunks: &[sift_core::EmbeddedChunk]) -> SiftResult<()> {
        self.inner.insert(chunks)
    }

    pub fn delete_by_uri(&self, uri: &str) -> SiftResult<u64> {
        self.inner.delete_by_uri(uri)
    }

    pub fn count(&self) -> SiftResult<u64> {
        self.inner.count()
    }

    pub fn inner(&self) -> &HybridSearchEngine<V, F> {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sift_core::ContentType;

    fn create_test_engine() -> CachedSearchEngine<crate::flat::FlatVectorIndex, impl FullTextStore>
    {
        let vector_store = crate::flat::FlatVectorIndex::new();

        #[cfg(feature = "fulltext")]
        let fulltext_store = crate::tantivy_store::TantivyStore::open_in_memory().unwrap();
        #[cfg(all(not(feature = "fulltext"), feature = "fts5"))]
        let fulltext_store = crate::fts5::Fts5Store::open_in_memory().unwrap();
        #[cfg(all(not(feature = "fulltext"), not(feature = "fts5")))]
        let fulltext_store = {
            let tmp = tempfile::tempdir().unwrap();
            crate::bm25::Bm25Store::open(&tmp.path().join("bm25.json")).unwrap()
        };

        let engine = HybridSearchEngine::new(vector_store, fulltext_store, 0.7);
        CachedSearchEngine::new(engine, 50, Duration::from_secs(60))
    }

    fn insert_test_data<V: VectorStore, F: FullTextStore>(engine: &CachedSearchEngine<V, F>) {
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
    }

    #[test]
    fn cache_hit_returns_same_results() {
        let engine = create_test_engine();
        insert_test_data(&engine);

        let r1 = engine
            .search(&[1.0, 0.0, 0.0], "hello", 10, SearchMode::KeywordOnly)
            .unwrap();
        let r2 = engine
            .search(&[1.0, 0.0, 0.0], "hello", 10, SearchMode::KeywordOnly)
            .unwrap();

        assert_eq!(r1.len(), r2.len());
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.uri, b.uri);
            assert!((a.score - b.score).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn different_queries_do_not_collide() {
        let engine = create_test_engine();
        insert_test_data(&engine);

        let r1 = engine
            .search(&[1.0, 0.0, 0.0], "hello", 10, SearchMode::KeywordOnly)
            .unwrap();
        let r2 = engine
            .search(&[1.0, 0.0, 0.0], "nonexistent", 10, SearchMode::KeywordOnly)
            .unwrap();

        assert!(!r1.is_empty());
        assert!(r2.is_empty());
    }

    #[test]
    fn ttl_expiration() {
        let vector_store = crate::flat::FlatVectorIndex::new();

        #[cfg(feature = "fulltext")]
        let fulltext_store = crate::tantivy_store::TantivyStore::open_in_memory().unwrap();
        #[cfg(all(not(feature = "fulltext"), feature = "fts5"))]
        let fulltext_store = crate::fts5::Fts5Store::open_in_memory().unwrap();
        #[cfg(all(not(feature = "fulltext"), not(feature = "fts5")))]
        let fulltext_store = {
            let tmp = tempfile::tempdir().unwrap();
            crate::bm25::Bm25Store::open(&tmp.path().join("bm25.json")).unwrap()
        };

        let engine = HybridSearchEngine::new(vector_store, fulltext_store, 0.7);
        // Zero TTL so entries expire immediately
        let cached = CachedSearchEngine::new(engine, 50, Duration::from_secs(0));
        insert_test_data(&cached);

        let r1 = cached
            .search(&[1.0, 0.0, 0.0], "hello", 10, SearchMode::KeywordOnly)
            .unwrap();
        // With zero TTL, the entry is expired on the next lookup, forcing a fresh query
        let r2 = cached
            .search(&[1.0, 0.0, 0.0], "hello", 10, SearchMode::KeywordOnly)
            .unwrap();

        assert_eq!(r1.len(), r2.len());
        // Both should return valid results (cache miss both times)
        assert!(!r1.is_empty());
    }

    #[test]
    fn different_modes_produce_different_cache_keys() {
        let key_hybrid = cache_key("hello", 10, SearchMode::Hybrid);
        let key_keyword = cache_key("hello", 10, SearchMode::KeywordOnly);
        let key_vector = cache_key("hello", 10, SearchMode::VectorOnly);

        assert_ne!(key_hybrid, key_keyword);
        assert_ne!(key_hybrid, key_vector);
        assert_ne!(key_keyword, key_vector);
    }

    #[test]
    fn different_top_k_produce_different_cache_keys() {
        let key_5 = cache_key("hello", 5, SearchMode::Hybrid);
        let key_10 = cache_key("hello", 10, SearchMode::Hybrid);

        assert_ne!(key_5, key_10);
    }
}
