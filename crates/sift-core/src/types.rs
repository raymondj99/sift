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
        f.write_str(self.as_str())
    }
}

impl ContentType {
    /// The canonical lowercase tag for this content type. The same string
    /// round-trips through [`FromStr`].
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            ContentType::Text => "text",
            ContentType::Code => "code",
            ContentType::Image => "image",
            ContentType::Audio => "audio",
            ContentType::Data => "data",
        }
    }
}

/// Returned by [`ContentType::from_str`] for tags outside the known set.
///
/// Callers that want the historical "unknown ⇒ Text" behaviour should write
/// `s.parse().unwrap_or(ContentType::Text)` so the fallback is visible at
/// the call site rather than hidden inside the parser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownContentType(pub String);

impl std::fmt::Display for UnknownContentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unknown content type: {}", self.0)
    }
}

impl std::error::Error for UnknownContentType {}

impl std::str::FromStr for ContentType {
    type Err = UnknownContentType;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "text" => ContentType::Text,
            "code" => ContentType::Code,
            "image" => ContentType::Image,
            "audio" => ContentType::Audio,
            "data" => ContentType::Data,
            _ => return Err(UnknownContentType(s.to_string())),
        })
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
///
/// Not `Clone` because the `Error` variant carries `SiftError`, which wraps
/// `std::io::Error` (non-`Clone`). Channels that need ownership transfer
/// move events; consumers that need to inspect-and-forward should match
/// before passing along.
#[derive(Debug)]
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
        /// The underlying typed error. Carrying `SiftError` instead of a
        /// stringified message lets upstream consumers filter by error class
        /// (parse vs storage vs IO) without re-parsing free-form text.
        error: crate::SiftError,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchMode {
    #[default]
    Hybrid,
    VectorOnly,
    KeywordOnly,
}

impl SearchMode {
    /// Canonical lowercase tag (`"hybrid"`, `"vector"`, `"keyword"`). Round
    /// trips through [`SearchMode::from_str`].
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            SearchMode::Hybrid => "hybrid",
            SearchMode::VectorOnly => "vector",
            SearchMode::KeywordOnly => "keyword",
        }
    }
}

impl std::fmt::Display for SearchMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for SearchMode {
    type Err = ();

    /// Parse a `SearchMode` from a lowercase tag. Unknown inputs return
    /// `Err(())` so callers can decide on a default explicitly rather than
    /// silently coercing to one mode.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "hybrid" => Ok(SearchMode::Hybrid),
            "vector" | "vector_only" => Ok(SearchMode::VectorOnly),
            "keyword" | "keyword_only" => Ok(SearchMode::KeywordOnly),
            _ => Err(()),
        }
    }
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
///
/// Asymmetric retrieval models (Nomic, EmbeddingGemma, Snowflake Arctic, etc.)
/// require *different* prefixes on the query side vs the document side. Earlier
/// versions of sift hardcoded the Nomic-specific `search_query: ` /
/// `search_document: ` strings at every call site, which would silently degrade
/// recall the moment a different model was loaded. The trait now exposes
/// per-model prefixes via [`Embedder::search_prefix`] / [`Embedder::document_prefix`]
/// and the convenience methods [`Embedder::embed_query`] /
/// [`Embedder::embed_passage`] / [`Embedder::embed_passages`] apply them
/// automatically. New code should always go through those helpers.
pub trait Embedder: Send + Sync {
    /// Embed a batch of texts, returning one vector per input.
    ///
    /// Callers that already know they have a query or a passage should prefer
    /// [`Embedder::embed_query`] / [`Embedder::embed_passages`] so per-model
    /// prefixes are applied automatically.
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

    /// Prefix prepended to *query* texts before embedding, e.g. Nomic's
    /// `"search_query: "`. Default is empty for symmetric models.
    ///
    /// Returns `&'static str` because the registered models' prefixes are
    /// compiled in via [`crate::Embedder`] implementations. Implementors
    /// that need runtime-configurable prefixes should box the strings and
    /// expose them via a different mechanism.
    fn search_prefix(&self) -> &'static str {
        ""
    }

    /// Prefix prepended to *document* / *passage* texts before embedding,
    /// e.g. Nomic's `"search_document: "`. Default is empty for symmetric
    /// models.
    fn document_prefix(&self) -> &'static str {
        ""
    }

    /// Embed a single query, applying [`Embedder::search_prefix`] automatically.
    fn embed_query(&self, text: &str) -> SiftResult<Vec<f32>> {
        let prefix = self.search_prefix();
        if prefix.is_empty() {
            self.embed(text)
        } else {
            self.embed(&format!("{prefix}{text}"))
        }
    }

    /// Embed a single passage / document chunk, applying
    /// [`Embedder::document_prefix`] automatically.
    fn embed_passage(&self, text: &str) -> SiftResult<Vec<f32>> {
        let prefix = self.document_prefix();
        if prefix.is_empty() {
            self.embed(text)
        } else {
            self.embed(&format!("{prefix}{text}"))
        }
    }

    /// Embed a batch of passages / document chunks with the document prefix.
    fn embed_passages(&self, texts: &[&str]) -> SiftResult<Vec<Vec<f32>>> {
        let prefix = self.document_prefix();
        if prefix.is_empty() {
            return self.embed_batch(texts);
        }
        let prefixed: Vec<String> = texts.iter().map(|t| format!("{prefix}{t}")).collect();
        let refs: Vec<&str> = prefixed.iter().map(String::as_str).collect();
        self.embed_batch(&refs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock embedder that records every call to `embed_batch` so tests can
    /// inspect whether the per-model prefixes were applied. Returns dummy
    /// vectors derived from the input length so equality tests are stable.
    struct MockEmbedder {
        captured: std::sync::Mutex<Vec<String>>,
        search_prefix: &'static str,
        document_prefix: &'static str,
    }

    impl MockEmbedder {
        fn new(search_prefix: &'static str, document_prefix: &'static str) -> Self {
            Self {
                captured: std::sync::Mutex::new(Vec::new()),
                search_prefix,
                document_prefix,
            }
        }

        fn captured(&self) -> Vec<String> {
            self.captured.lock().unwrap().clone()
        }
    }

    impl Embedder for MockEmbedder {
        fn embed_batch(&self, texts: &[&str]) -> SiftResult<Vec<Vec<f32>>> {
            let mut captured = self.captured.lock().unwrap();
            for t in texts {
                captured.push((*t).to_string());
            }
            Ok(texts.iter().map(|t| vec![t.len() as f32]).collect())
        }
        fn dimensions(&self) -> usize {
            1
        }
        // The trait keeps `model_name -> &str` (real impls store a `String`)
        // so a literal here triggers `unnecessary_literal_bound`. Allow it
        // for the test mock — the alternative would be a leaked `String`
        // just to satisfy the lint.
        #[allow(clippy::unnecessary_literal_bound)]
        fn model_name(&self) -> &str {
            "mock"
        }
        fn search_prefix(&self) -> &'static str {
            self.search_prefix
        }
        fn document_prefix(&self) -> &'static str {
            self.document_prefix
        }
    }

    #[test]
    fn embed_query_prepends_search_prefix() {
        let m = MockEmbedder::new("search_query: ", "search_document: ");
        let _ = m.embed_query("hello").unwrap();
        assert_eq!(m.captured(), vec!["search_query: hello"]);
    }

    #[test]
    fn embed_passage_prepends_document_prefix() {
        let m = MockEmbedder::new("search_query: ", "search_document: ");
        let _ = m.embed_passage("world").unwrap();
        assert_eq!(m.captured(), vec!["search_document: world"]);
    }

    #[test]
    fn embed_passages_prepends_to_each() {
        let m = MockEmbedder::new("Q: ", "D: ");
        let _ = m.embed_passages(&["a", "b", "c"]).unwrap();
        assert_eq!(m.captured(), vec!["D: a", "D: b", "D: c"]);
    }

    #[test]
    fn embed_query_with_empty_prefix_skips_format_allocation() {
        // Symmetric models (BGE-M3, EmbeddingGemma's neutral case) use empty
        // prefixes. The trait default fast-paths empty prefixes through
        // `embed` directly without an extra allocation, but the captured
        // string must still match the original input bit-for-bit.
        let m = MockEmbedder::new("", "");
        let _ = m.embed_query("plain").unwrap();
        let _ = m.embed_passage("text").unwrap();
        assert_eq!(m.captured(), vec!["plain", "text"]);
    }

    #[test]
    fn embed_passages_with_empty_prefix_uses_batch_directly() {
        // With an empty prefix, embed_passages must dispatch to embed_batch
        // without allocating prefixed strings.
        let m = MockEmbedder::new("", "");
        let _ = m.embed_passages(&["one", "two"]).unwrap();
        assert_eq!(m.captured(), vec!["one", "two"]);
    }

    #[test]
    fn default_embedder_prefixes_are_empty() {
        // A trait impl that doesn't override search_prefix / document_prefix
        // must report the empty string — otherwise asymmetric models would
        // silently double-prefix on legacy call sites.
        struct Bare;
        impl Embedder for Bare {
            fn embed_batch(&self, texts: &[&str]) -> SiftResult<Vec<Vec<f32>>> {
                Ok(texts.iter().map(|_| vec![0.0]).collect())
            }
            fn dimensions(&self) -> usize {
                1
            }
            #[allow(clippy::unnecessary_literal_bound)]
            fn model_name(&self) -> &str {
                "bare"
            }
        }
        let b = Bare;
        assert_eq!(b.search_prefix(), "");
        assert_eq!(b.document_prefix(), "");
    }

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
