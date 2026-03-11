use crate::error::{SiftError, SiftResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// A cooperative cancellation token for pipeline stages.
///
/// Clone-cheap (wraps an `Arc<AtomicBool>`). Call [`cancel()`](Self::cancel)
/// from a signal handler; pipeline stages poll [`is_cancelled()`](Self::is_cancelled).
#[derive(Clone)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    pub fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

/// Represents a discovered item from a source, before parsing.
#[derive(Debug, Clone)]
pub struct SourceItem {
    pub uri: String,
    pub path: PathBuf,
    pub content_hash: [u8; 32],
    pub size: u64,
    pub modified_at: Option<i64>,
    pub mime_type: Option<String>,
    pub extension: Option<String>,
}

/// A parsed document with extracted text and metadata.
#[derive(Debug, Clone)]
pub struct ParsedDocument {
    pub text: String,
    pub title: Option<String>,
    pub language: Option<String>,
    pub content_type: ContentType,
    pub metadata: HashMap<String, String>,
}

/// Content type classification for routing to the right chunker/embedder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ContentType {
    Text,
    Code,
    Image,
    Audio,
    Data,
}

impl std::fmt::Display for ContentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContentType::Text => write!(f, "text"),
            ContentType::Code => write!(f, "code"),
            ContentType::Image => write!(f, "image"),
            ContentType::Audio => write!(f, "audio"),
            ContentType::Data => write!(f, "data"),
        }
    }
}

/// A chunk of a document ready for embedding.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub text: String,
    pub source_uri: String,
    pub chunk_index: u32,
    pub content_type: ContentType,
    pub file_type: String,
    pub title: Option<String>,
    pub language: Option<String>,
    pub byte_range: Option<(u64, u64)>,
}

/// A chunk with its embedding vector, ready for storage.
#[derive(Debug, Clone)]
pub struct EmbeddedChunk {
    pub chunk: Chunk,
    pub vector: Vec<f32>,
}

/// A search result returned to the user.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub uri: String,
    pub text: String,
    pub score: f32,
    pub chunk_index: u32,
    pub content_type: ContentType,
    pub file_type: String,
    pub title: Option<String>,
    pub byte_range: Option<(u64, u64)>,
}

/// Index statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    pub total_sources: u64,
    pub total_chunks: u64,
    pub index_size_bytes: u64,
    pub file_type_counts: HashMap<String, u64>,
}

/// Scan progress event for the CLI to display.
#[derive(Debug, Clone)]
pub enum ScanEvent {
    Discovered {
        total: u64,
    },
    Parsing {
        current: u64,
        total: u64,
        path: String,
    },
    Embedding {
        current: u64,
        total: u64,
    },
    Complete {
        stats: IndexStats,
    },
    Error {
        path: String,
        message: String,
    },
}

/// Options for a scan operation.
#[derive(Debug, Clone)]
pub struct ScanOptions {
    pub paths: Vec<PathBuf>,
    pub recursive: bool,
    pub max_depth: Option<usize>,
    pub max_file_size: Option<u64>,
    pub include_globs: Vec<String>,
    pub exclude_globs: Vec<String>,
    pub file_types: Vec<String>,
    pub dry_run: bool,
    pub jobs: usize,
}

impl Default for ScanOptions {
    fn default() -> Self {
        Self {
            paths: vec![],
            recursive: true,
            max_depth: None,
            max_file_size: None,
            include_globs: vec![],
            exclude_globs: vec![],
            file_types: vec![],
            dry_run: false,
            jobs: 0,
        }
    }
}

/// Options for a search operation.
#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub query: String,
    pub max_results: usize,
    pub file_type: Option<String>,
    pub path_glob: Option<String>,
    pub threshold: f32,
    pub mode: SearchMode,
    pub context: bool,
    pub after: Option<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    Hybrid,
    VectorOnly,
    KeywordOnly,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            query: String::new(),
            max_results: 10,
            file_type: None,
            path_glob: None,
            threshold: 0.5,
            mode: SearchMode::Hybrid,
            context: false,
            after: None,
        }
    }
}

/// An embedding model that converts text to vectors.
pub trait Embedder: Send + Sync {
    /// Embed a batch of texts, returning one vector per input.
    fn embed_batch(&self, texts: &[&str]) -> SiftResult<Vec<Vec<f32>>>;

    /// Embed a single text.
    fn embed(&self, text: &str) -> SiftResult<Vec<f32>> {
        let mut results = self.embed_batch(&[text])?;
        results
            .pop()
            .ok_or_else(|| SiftError::Embedding("Empty result from embed_batch".into()))
    }

    /// Vector dimensionality.
    fn dimensions(&self) -> usize;

    /// Model name.
    fn model_name(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_options_default_context_is_false() {
        let opts = SearchOptions::default();
        assert!(!opts.context);
    }

    #[test]
    fn test_search_options_default_after_is_none() {
        let opts = SearchOptions::default();
        assert!(opts.after.is_none());
    }

    #[test]
    fn test_search_options_with_context() {
        let opts = SearchOptions {
            context: true,
            ..Default::default()
        };
        assert!(opts.context);
    }

    #[test]
    fn test_search_options_with_after() {
        let opts = SearchOptions {
            after: Some(1735689600),
            ..Default::default()
        };
        assert_eq!(opts.after, Some(1735689600));
    }

    #[test]
    fn test_scan_options_default_jobs_is_zero() {
        let opts = ScanOptions::default();
        assert_eq!(opts.jobs, 0);
    }

    // --- CancellationToken tests ---

    #[test]
    fn cancellation_token_new_is_not_cancelled() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn cancellation_token_cancel_sets_cancelled() {
        let token = CancellationToken::new();
        token.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn cancellation_token_clone_shares_state() {
        let token = CancellationToken::new();
        let clone = token.clone();
        assert!(!clone.is_cancelled());

        token.cancel();
        assert!(clone.is_cancelled());
    }

    // --- ContentType Display tests ---

    #[test]
    fn content_type_display_text() {
        assert_eq!(ContentType::Text.to_string(), "text");
    }

    #[test]
    fn content_type_display_code() {
        assert_eq!(ContentType::Code.to_string(), "code");
    }

    #[test]
    fn content_type_display_image() {
        assert_eq!(ContentType::Image.to_string(), "image");
    }

    #[test]
    fn content_type_display_audio() {
        assert_eq!(ContentType::Audio.to_string(), "audio");
    }

    #[test]
    fn content_type_display_data() {
        assert_eq!(ContentType::Data.to_string(), "data");
    }

    // --- ContentType serde roundtrip ---

    #[test]
    fn content_type_serde_roundtrip() {
        for variant in [
            ContentType::Text,
            ContentType::Code,
            ContentType::Image,
            ContentType::Audio,
            ContentType::Data,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let deserialized: ContentType = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, deserialized);
        }
    }

    // --- ScanOptions default values ---

    #[test]
    fn scan_options_default_values() {
        let opts = ScanOptions::default();
        assert!(opts.paths.is_empty());
        assert!(opts.recursive);
        assert!(opts.max_depth.is_none());
        assert!(opts.max_file_size.is_none());
        assert!(opts.include_globs.is_empty());
        assert!(opts.exclude_globs.is_empty());
        assert!(opts.file_types.is_empty());
        assert!(!opts.dry_run);
        assert_eq!(opts.jobs, 0);
    }

    // --- SearchOptions default values ---

    #[test]
    fn search_options_default_values() {
        let opts = SearchOptions::default();
        assert!(opts.query.is_empty());
        assert_eq!(opts.max_results, 10);
        assert!(opts.file_type.is_none());
        assert!(opts.path_glob.is_none());
        assert!((opts.threshold - 0.5).abs() < f32::EPSILON);
        assert_eq!(opts.mode, SearchMode::Hybrid);
        assert!(!opts.context);
        assert!(opts.after.is_none());
    }

    // --- Embedder default embed() delegates to embed_batch() ---

    struct FakeEmbedder;

    impl Embedder for FakeEmbedder {
        fn embed_batch(&self, texts: &[&str]) -> SiftResult<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|t| vec![t.len() as f32]).collect())
        }

        fn dimensions(&self) -> usize {
            1
        }

        fn model_name(&self) -> &'static str {
            "fake"
        }
    }

    #[test]
    fn embedder_default_embed_delegates_to_embed_batch() {
        let embedder = FakeEmbedder;
        let result = embedder.embed("hello").unwrap();
        assert_eq!(result, vec![5.0]);
    }
}
