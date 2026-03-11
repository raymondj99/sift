//! Text chunking strategies for splitting parsed documents into searchable segments.
//!
//! Provides [`FixedChunker`] (character-based), [`SemanticChunker`] (paragraph-aware),
//! [`RecursiveChunker`] (recursive character text splitter), and optionally
//! [`CodeChunker`] (AST-aware, via tree-sitter) behind the `ast` feature flag.

pub mod error;

#[cfg(feature = "ast")]
pub mod code;
pub mod fixed;
pub mod recursive;
pub mod semantic;
pub mod traits;

#[cfg(feature = "ast")]
pub use code::CodeChunker;
pub use error::ChunkerError;
pub use fixed::FixedChunker;
pub use recursive::RecursiveChunker;
pub use semantic::SemanticChunker;
pub use traits::Chunker;

use sift_core::ContentType;

/// The strategy to use when selecting a chunker.
pub enum ChunkStrategy {
    /// Automatic: code gets the AST chunker, text gets semantic, etc.
    Auto,
    /// Force the recursive character text splitter.
    Recursive,
}

/// Select the appropriate chunker based on content type.
pub fn chunker_for_content(
    content_type: ContentType,
    chunk_size: usize,
    chunk_overlap: usize,
) -> Box<dyn Chunker> {
    chunker_for_content_with_strategy(content_type, chunk_size, chunk_overlap, ChunkStrategy::Auto)
}

/// Select a chunker with an explicit strategy override.
pub fn chunker_for_content_with_strategy(
    content_type: ContentType,
    chunk_size: usize,
    chunk_overlap: usize,
    strategy: ChunkStrategy,
) -> Box<dyn Chunker> {
    match strategy {
        ChunkStrategy::Recursive => Box::new(RecursiveChunker::new(chunk_size, chunk_overlap)),
        ChunkStrategy::Auto => match content_type {
            #[cfg(feature = "ast")]
            ContentType::Code => Box::new(CodeChunker::new(chunk_size, chunk_overlap)),
            #[cfg(not(feature = "ast"))]
            ContentType::Code => Box::new(SemanticChunker::new(chunk_size, chunk_overlap)),
            ContentType::Text => Box::new(SemanticChunker::new(chunk_size, chunk_overlap)),
            _ => Box::new(FixedChunker::new(chunk_size, chunk_overlap)),
        },
    }
}
