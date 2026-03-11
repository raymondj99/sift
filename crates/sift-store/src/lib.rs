//! Storage backends for vectors, full-text search, and metadata.
//!
//! Provides [`FlatVectorIndex`] (brute-force cosine similarity), optional
//! [`HnswIndex`] (approximate nearest-neighbor), [`MetadataStore`] (`SQLite` or JSON),
//! full-text search ([`Fts5Store`], [`TantivyStore`], or [`Bm25Store`]),
//! and [`HybridSearchEngine`] for Reciprocal Rank Fusion of vector + keyword results.

pub mod error;

#[cfg(all(not(feature = "fulltext"), not(feature = "fts5")))]
pub mod bm25;
pub mod flat;
#[cfg(all(not(feature = "fulltext"), feature = "fts5"))]
pub mod fts5;
#[cfg(feature = "hnsw")]
pub mod hnsw;
pub mod hybrid;
#[cfg(not(feature = "sqlite"))]
pub mod json_metadata;
#[cfg(feature = "sqlite")]
pub mod metadata;
#[cfg(feature = "fulltext")]
pub mod tantivy_store;
pub mod traits;

#[cfg(all(not(feature = "fulltext"), not(feature = "fts5")))]
pub use bm25::Bm25Store;
pub use flat::{ExportEntry, FlatVectorIndex};
#[cfg(all(not(feature = "fulltext"), feature = "fts5"))]
pub use fts5::Fts5Store;
#[cfg(feature = "hnsw")]
pub use hnsw::HnswIndex;
pub use hybrid::HybridSearchEngine;
#[cfg(not(feature = "sqlite"))]
pub use json_metadata::MetadataStore;
#[cfg(feature = "sqlite")]
pub use metadata::{MetadataStore, TransactionGuard};
#[cfg(feature = "fulltext")]
pub use tantivy_store::TantivyStore;
pub use error::StorageError;
pub use traits::{FullTextStore, VectorIndex, VectorStore};

/// Default vector store type alias.
///
/// When the `hnsw` feature is enabled, this uses the approximate
/// nearest-neighbour HNSW index. Otherwise it falls back to the
/// brute-force flat index.
#[cfg(feature = "hnsw")]
pub type SimpleVectorStore = HnswIndex;
#[cfg(not(feature = "hnsw"))]
pub type SimpleVectorStore = FlatVectorIndex;

#[cfg(feature = "fulltext")]
pub type DefaultFullTextStore = TantivyStore;
#[cfg(all(not(feature = "fulltext"), feature = "fts5"))]
pub type DefaultFullTextStore = Fts5Store;
#[cfg(all(not(feature = "fulltext"), not(feature = "fts5")))]
pub type DefaultFullTextStore = Bm25Store;
