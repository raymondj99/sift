//! Hybrid-search engine bootstrap.
//!
//! Both `sift-cli` and `sift-mcp` need the same "open or create the index"
//! ceremony: pick the right vector backend (HNSW vs Flat), the right
//! full-text backend (Tantivy / FTS5 / BM25), and the right metadata store
//! (SQLite / JSON), then assemble them into a [`HybridSearchEngine`].
//!
//! Centralising the cfg-gate ladder here means a new backend or a renamed
//! on-disk file only changes one place.

use crate::{DefaultFullTextStore, HybridSearchEngine, MetadataStore, SimpleVectorStore};
use sift_core::{Config, SiftResult};

/// Open or create the hybrid search engine and its metadata store for the
/// configured index directory.
///
/// On first run this creates the on-disk layout. On later runs it loads the
/// existing one. The returned engine wraps whichever vector / full-text
/// backends are compiled in via cargo features.
pub fn open_engine(
    config: &Config,
) -> SiftResult<(
    HybridSearchEngine<SimpleVectorStore, DefaultFullTextStore>,
    MetadataStore,
)> {
    config.ensure_dirs()?;
    let index_dir = config.index_dir()?;

    #[cfg(feature = "hnsw")]
    let vector_store = {
        let precision = crate::VectorPrecision::from_config(&config.search.vector_quantization);
        SimpleVectorStore::load_or_create_with_precision(&index_dir, precision)?
    };
    #[cfg(not(feature = "hnsw"))]
    let vector_store = SimpleVectorStore::load_or_migrate(&index_dir)?;

    #[cfg(feature = "fulltext")]
    let fulltext_store = DefaultFullTextStore::open(&index_dir.join("tantivy"))?;
    #[cfg(all(not(feature = "fulltext"), feature = "fts5"))]
    let fulltext_store = DefaultFullTextStore::open(&index_dir.join("fts5.db"))?;
    #[cfg(all(not(feature = "fulltext"), not(feature = "fts5")))]
    let fulltext_store = DefaultFullTextStore::open(&index_dir.join("bm25.json"))?;

    #[cfg(feature = "sqlite")]
    let metadata_path = index_dir.join("metadata.db");
    #[cfg(not(feature = "sqlite"))]
    let metadata_path = index_dir.join("metadata.json");
    let metadata = MetadataStore::open(&metadata_path)?;

    let engine = HybridSearchEngine::new(vector_store, fulltext_store, config.search.hybrid_alpha);
    Ok((engine, metadata))
}
