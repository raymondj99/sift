use assert_cmd::{cargo_bin, Command};
use predicates::prelude::*;
use sift_chunker::{chunker_for_content, Chunker, FixedChunker, SemanticChunker};
use sift_core::*;
use sift_parsers::ParserRegistry;
use sift_sources::{FilesystemSource, Source};
#[cfg(feature = "fulltext")]
use sift_store::HybridSearchEngine;
use sift_store::{ExportEntry, MetadataStore, SimpleVectorStore, VectorIndex, VectorStore};
#[cfg(feature = "fulltext")]
use sift_store::{FullTextStore, TantivyStore};
use std::fs;
use std::io::{BufRead, Write};
use tempfile::TempDir;

/// Create a temporary directory with test files.
fn create_test_corpus(dir: &TempDir) {
    // Text files
    fs::write(
        dir.path().join("readme.md"),
        "# My Project\n\nThis project handles payment processing and error handling.\nIt uses Rust for performance.\n",
    ).unwrap();

    fs::write(
        dir.path().join("notes.txt"),
        "Meeting notes from 2025-01-15\n\nDiscussed quarterly revenue projections.\nAction items: review budget, update forecasts.\n",
    ).unwrap();

    // Code files
    fs::write(
        dir.path().join("main.rs"),
        r#"fn main() {
    println!("Hello, world!");
    let config = load_config();
    run_server(config);
}

fn load_config() -> Config {
    // Load configuration from TOML file
    Config::default()
}

fn run_server(config: Config) {
    println!("Starting server on port {}", config.port);
}

struct Config {
    port: u16,
}

impl Default for Config {
    fn default() -> Self {
        Self { port: 8080 }
    }
}
"#,
    )
    .unwrap();

    fs::write(
        dir.path().join("utils.py"),
        r#"def calculate_revenue(sales: list[float]) -> float:
    """Calculate total revenue from sales data."""
    return sum(sales)

def format_currency(amount: float) -> str:
    """Format a number as currency."""
    return f"${amount:,.2f}"

class RevenueReport:
    def __init__(self, year: int):
        self.year = year
        self.data = []

    def add_quarter(self, q: int, revenue: float):
        self.data.append({"quarter": q, "revenue": revenue})
"#,
    )
    .unwrap();

    // Data files
    fs::write(
        dir.path().join("data.json"),
        r#"{"name": "Alice", "department": "Engineering", "projects": ["search", "indexing"]}"#,
    )
    .unwrap();

    fs::write(
        dir.path().join("data.csv"),
        "name,score,department\nAlice,95,Engineering\nBob,87,Marketing\nCharlie,92,Engineering\n",
    )
    .unwrap();

    // Web files
    fs::write(
        dir.path().join("page.html"),
        "<html><head><title>Search Results</title></head><body><h1>Results</h1><p>Found 42 matches for your query.</p></body></html>",
    ).unwrap();

    // Config files
    fs::write(
        dir.path().join("config.toml"),
        r#"[server]
host = "0.0.0.0"
port = 3000

[database]
url = "postgres://localhost/mydb"
pool_size = 10
"#,
    )
    .unwrap();

    // Nested directory
    fs::create_dir(dir.path().join("src")).unwrap();
    fs::write(
        dir.path().join("src/lib.rs"),
        r#"pub mod search;
pub mod index;

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
"#,
    )
    .unwrap();
}

#[test]
fn test_full_pipeline_discover_parse_chunk() {
    let dir = TempDir::new().unwrap();
    create_test_corpus(&dir);

    // Phase 1: Discover
    let source = FilesystemSource::new();
    let options = ScanOptions {
        paths: vec![dir.path().to_path_buf()],
        ..Default::default()
    };

    let items = source.discover(&options).unwrap();

    assert!(
        items.len() >= 8,
        "Should discover at least 8 files, got {}",
        items.len()
    );

    // Phase 2: Parse all files
    let registry = ParserRegistry::new();
    let mut documents = Vec::new();

    for item in &items {
        let content = fs::read(&item.path).unwrap();
        match registry.parse(
            &content,
            item.mime_type.as_deref(),
            item.extension.as_deref(),
        ) {
            Ok(doc) => documents.push((item.clone(), doc)),
            Err(e) => {
                // Some files might not be parseable
                eprintln!("Parse error for {}: {}", item.uri, e);
            }
        }
    }

    assert!(!documents.is_empty(), "Should parse at least some files");

    // Phase 3: Chunk all documents
    let mut all_chunks: Vec<Chunk> = Vec::new();

    for (item, doc) in &documents {
        let chunker = chunker_for_content(doc.content_type, 200, 20);
        let raw_chunks = chunker.chunk(&doc.text);

        let file_type = item.extension.as_deref().unwrap_or("unknown");
        for (i, (text, offset)) in raw_chunks.iter().enumerate() {
            all_chunks.push(Chunk {
                text: text.clone(),
                source_uri: item.uri.clone(),
                chunk_index: i as u32,
                content_type: doc.content_type,
                file_type: file_type.to_string(),
                title: doc.title.clone(),
                language: doc.language.clone(),
                byte_range: Some((*offset as u64, (*offset + text.len()) as u64)),
            });
        }
    }

    assert!(
        all_chunks.len() > documents.len(),
        "Should have more chunks than documents (chunking works)"
    );

    // Verify content types
    let has_code = all_chunks
        .iter()
        .any(|c| c.content_type == ContentType::Code);
    let has_text = all_chunks
        .iter()
        .any(|c| c.content_type == ContentType::Text);
    #[cfg(feature = "data")]
    let has_data = all_chunks
        .iter()
        .any(|c| c.content_type == ContentType::Data);

    assert!(has_code, "Should have code chunks");
    assert!(has_text, "Should have text chunks");
    #[cfg(feature = "data")]
    assert!(has_data, "Should have data chunks");
}

#[test]
fn test_metadata_store_change_detection() {
    let dir = TempDir::new().unwrap();
    create_test_corpus(&dir);

    let meta = MetadataStore::open_in_memory().unwrap();

    // First scan - everything is new
    let source = FilesystemSource::new();
    let options = ScanOptions {
        paths: vec![dir.path().to_path_buf()],
        ..Default::default()
    };

    let items = source.discover(&options).unwrap();
    let new_count = items.len() as u64;
    for item in &items {
        let status = meta.check_source(&item.uri, &item.content_hash).unwrap();
        assert!(status.is_none(), "First scan should find all files as new");
        meta.upsert_source(
            &item.uri,
            &item.content_hash,
            item.size,
            "txt",
            item.modified_at,
            1,
        )
        .unwrap();
    }

    assert!(new_count > 0);

    // Second scan - everything should be unchanged
    let items = source.discover(&options).unwrap();
    let unchanged_count = items.len() as u64;
    for item in &items {
        let status = meta.check_source(&item.uri, &item.content_hash).unwrap();
        assert_eq!(
            status,
            Some(true),
            "Second scan should find all files unchanged"
        );
    }

    assert_eq!(new_count, unchanged_count);

    // Modify a file
    fs::write(
        dir.path().join("readme.md"),
        "# Updated Content\n\nNew text.\n",
    )
    .unwrap();

    // Third scan - one file should be changed
    let mut changed = 0u64;
    let mut same = 0u64;
    let items = source.discover(&options).unwrap();
    for item in &items {
        let status = meta.check_source(&item.uri, &item.content_hash).unwrap();
        match status {
            Some(true) => same += 1,
            Some(false) => changed += 1,
            None => panic!("Should not find new files"),
        }
    }

    assert_eq!(changed, 1, "One file should have changed");
    assert_eq!(same, new_count - 1, "Rest should be unchanged");
}

#[test]
fn test_vector_store_and_search() {
    let store = SimpleVectorStore::new();

    // Create test embedded chunks with distinct vectors
    let chunks = vec![
        EmbeddedChunk {
            chunk: Chunk {
                text: "payment processing error handling".to_string(),
                source_uri: "file:///payment.rs".to_string(),
                chunk_index: 0,
                content_type: ContentType::Code,
                file_type: "rs".to_string(),
                title: None,
                language: Some("rust".to_string()),
                byte_range: None,
            },
            vector: vec![0.9, 0.1, 0.0, 0.0],
        },
        EmbeddedChunk {
            chunk: Chunk {
                text: "quarterly revenue analysis report".to_string(),
                source_uri: "file:///revenue.md".to_string(),
                chunk_index: 0,
                content_type: ContentType::Text,
                file_type: "md".to_string(),
                title: Some("Revenue Report".to_string()),
                language: None,
                byte_range: None,
            },
            vector: vec![0.0, 0.9, 0.1, 0.0],
        },
        EmbeddedChunk {
            chunk: Chunk {
                text: "database connection pooling configuration".to_string(),
                source_uri: "file:///config.toml".to_string(),
                chunk_index: 0,
                content_type: ContentType::Data,
                file_type: "toml".to_string(),
                title: None,
                language: None,
                byte_range: None,
            },
            vector: vec![0.0, 0.0, 0.9, 0.1],
        },
    ];

    store.insert(&chunks).unwrap();
    assert_eq!(store.count().unwrap(), 3);

    // Search for something similar to payment errors
    let results = store.search(&[0.8, 0.2, 0.0, 0.0], 3).unwrap();
    assert_eq!(results[0].uri, "file:///payment.rs");

    // Search for something similar to revenue
    let results = store.search(&[0.0, 0.8, 0.2, 0.0], 3).unwrap();
    assert_eq!(results[0].uri, "file:///revenue.md");

    // Delete and verify
    let deleted = store.delete_by_uri("file:///payment.rs").unwrap();
    assert_eq!(deleted, 1);
    assert_eq!(store.count().unwrap(), 2);
}

#[cfg(feature = "fulltext")]
#[test]
fn test_tantivy_fulltext_search() {
    let store = TantivyStore::open_in_memory().unwrap();

    let chunks = vec![
        EmbeddedChunk {
            chunk: Chunk {
                text: "The payment processing system handles credit card transactions securely."
                    .to_string(),
                source_uri: "file:///payment.rs".to_string(),
                chunk_index: 0,
                content_type: ContentType::Code,
                file_type: "rs".to_string(),
                title: Some("Payment Module".to_string()),
                language: Some("rust".to_string()),
                byte_range: None,
            },
            vector: vec![],
        },
        EmbeddedChunk {
            chunk: Chunk {
                text: "Quarterly revenue projections show strong growth in Q3 2025.".to_string(),
                source_uri: "file:///report.md".to_string(),
                chunk_index: 0,
                content_type: ContentType::Text,
                file_type: "md".to_string(),
                title: Some("Q3 Report".to_string()),
                language: None,
                byte_range: None,
            },
            vector: vec![],
        },
        EmbeddedChunk {
            chunk: Chunk {
                text: "Configure the database connection pool with max 10 connections.".to_string(),
                source_uri: "file:///config.md".to_string(),
                chunk_index: 0,
                content_type: ContentType::Text,
                file_type: "md".to_string(),
                title: None,
                language: None,
                byte_range: None,
            },
            vector: vec![],
        },
    ];

    store.insert(&chunks).unwrap();

    // Search for payment-related content
    let results = store.search("payment credit card", 10).unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].uri, "file:///payment.rs");

    // Search for revenue
    let results = store.search("quarterly revenue", 10).unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].uri, "file:///report.md");

    // Search for database config
    let results = store.search("database connection pool", 10).unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].uri, "file:///config.md");
}

#[cfg(feature = "fulltext")]
#[test]
fn test_hybrid_search_rrf_fusion() {
    let vector_store = SimpleVectorStore::new();
    let fulltext_store = TantivyStore::open_in_memory().unwrap();

    let engine = HybridSearchEngine::new(vector_store, fulltext_store, 0.5);

    let chunks = vec![
        EmbeddedChunk {
            chunk: Chunk {
                text: "Error handling in payment processing requires careful retry logic."
                    .to_string(),
                source_uri: "file:///errors.rs".to_string(),
                chunk_index: 0,
                content_type: ContentType::Code,
                file_type: "rs".to_string(),
                title: None,
                language: Some("rust".to_string()),
                byte_range: None,
            },
            vector: vec![0.9, 0.1, 0.0],
        },
        EmbeddedChunk {
            chunk: Chunk {
                text: "The search engine uses cosine similarity for vector matching.".to_string(),
                source_uri: "file:///search.rs".to_string(),
                chunk_index: 0,
                content_type: ContentType::Code,
                file_type: "rs".to_string(),
                title: None,
                language: Some("rust".to_string()),
                byte_range: None,
            },
            vector: vec![0.0, 0.9, 0.1],
        },
    ];

    engine.insert(&chunks).unwrap();
    assert_eq!(engine.count().unwrap(), 2);

    // Hybrid search (both vector and keyword contribute)
    let results = engine
        .search(
            &[0.85, 0.15, 0.0], // similar to errors.rs
            "error handling payment",
            5,
            SearchMode::Hybrid,
        )
        .unwrap();

    assert!(!results.is_empty());
    assert_eq!(results[0].uri, "file:///errors.rs");

    // Keyword-only mode
    let results = engine
        .search(
            &[0.0, 0.0, 0.0],
            "cosine similarity",
            5,
            SearchMode::KeywordOnly,
        )
        .unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].uri, "file:///search.rs");

    // Vector-only mode
    let results = engine
        .search(&[0.0, 0.85, 0.15], "", 5, SearchMode::VectorOnly)
        .unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].uri, "file:///search.rs");
}

#[test]
fn test_config_defaults_and_roundtrip() {
    let config = Config::default();

    // Verify all defaults match PLAN.md spec
    assert_eq!(config.default.model, "nomic-embed-text-v2");
    assert_eq!(config.default.chunk_size, 512);
    assert_eq!(config.default.chunk_overlap, 64);
    assert_eq!(config.default.max_file_size, 100 * 1024 * 1024);
    assert_eq!(config.search.max_results, 10);
    assert!((config.search.hybrid_alpha - 0.7).abs() < f32::EPSILON);
    assert_eq!(config.server.host, "127.0.0.1");
    assert_eq!(config.server.port, 7820);

    // Roundtrip through TOML
    let toml_str = toml::to_string_pretty(&config).unwrap();
    let loaded: Config = toml::from_str(&toml_str).unwrap();
    assert_eq!(loaded.default.chunk_size, config.default.chunk_size);
    assert_eq!(loaded.server.port, config.server.port);
}

#[test]
fn test_content_type_classification() {
    let registry = ParserRegistry::new();

    // Rust code → Code
    let doc = registry
        .parse(b"fn main() {}", Some("text/x-rust"), Some("rs"))
        .unwrap();
    assert_eq!(doc.content_type, ContentType::Code);

    // Markdown → Text
    let doc = registry
        .parse(b"# Hello\nWorld", Some("text/markdown"), Some("md"))
        .unwrap();
    assert_eq!(doc.content_type, ContentType::Text);

    // JSON → Data (requires data feature)
    #[cfg(feature = "data")]
    {
        let doc = registry
            .parse(
                br#"{"key":"value"}"#,
                Some("application/json"),
                Some("json"),
            )
            .unwrap();
        assert_eq!(doc.content_type, ContentType::Data);

        // CSV → Data
        let doc = registry
            .parse(b"a,b,c\n1,2,3", Some("text/csv"), Some("csv"))
            .unwrap();
        assert_eq!(doc.content_type, ContentType::Data);
    }

    // HTML → Text
    let doc = registry
        .parse(
            b"<html><body>Hi</body></html>",
            Some("text/html"),
            Some("html"),
        )
        .unwrap();
    assert_eq!(doc.content_type, ContentType::Text);
}

#[test]
fn test_chunker_respects_boundaries() {
    // Semantic chunker should prefer paragraph boundaries
    let chunker = SemanticChunker::new(100, 0);
    let text = "First paragraph about payment processing.\n\nSecond paragraph about error handling.\n\nThird paragraph about search indexing.";
    let chunks = chunker.chunk(text);

    assert!(chunks.len() >= 2, "Should split into multiple chunks");

    // Each chunk should be non-empty and trimmed
    for (chunk, _) in &chunks {
        assert!(!chunk.is_empty());
        assert_eq!(chunk.trim(), chunk.as_str());
    }

    // Fixed chunker should split long text
    let fixed = FixedChunker::new(50, 10);
    let long_text = "word ".repeat(100);
    let chunks = fixed.chunk(&long_text);
    assert!(chunks.len() > 1, "Should split long text");
}

#[cfg(feature = "embeddings")]
#[test]
fn test_embedding_cache() {
    let cache = sift_embed::EmbeddingCache::in_memory().unwrap();

    // Miss
    assert!(cache.get("hello world").is_none());
    assert_eq!(cache.stats(), (0, 1));

    // Put + hit
    cache.put("hello world", &[1.0, 2.0, 3.0]);
    let result = cache.get("hello world").unwrap();
    assert_eq!(result, vec![1.0, 2.0, 3.0]);
    assert_eq!(cache.stats(), (1, 1));

    // Different text is a miss
    assert!(cache.get("goodbye world").is_none());
    assert_eq!(cache.stats(), (1, 2));

    assert_eq!(cache.len(), 1);
}

#[test]
fn test_vector_store_persistence() {
    let dir = TempDir::new().unwrap();
    // save/load use the parent dir for multi-file storage formats
    let path = dir.path().join("vectors.bin");

    // Create and populate
    {
        let store = SimpleVectorStore::new();
        let chunks = vec![EmbeddedChunk {
            chunk: Chunk {
                text: "persistent content".to_string(),
                source_uri: "file:///test.txt".to_string(),
                chunk_index: 0,
                content_type: ContentType::Text,
                file_type: "txt".to_string(),
                title: Some("Test".to_string()),
                language: None,
                byte_range: Some((0, 18)),
            },
            vector: vec![1.0, 0.0, 0.0],
        }];
        store.insert(&chunks).unwrap();
        VectorIndex::save(&store, &path).unwrap();
    }

    // Load and verify
    {
        let store = SimpleVectorStore::load(&path).unwrap();
        assert_eq!(store.count().unwrap(), 1);

        let results = store.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].text, "persistent content");
        assert_eq!(results[0].title.as_deref(), Some("Test"));
        assert_eq!(results[0].byte_range, Some((0, 18)));
    }
}

#[test]
fn test_source_exclusion_patterns() {
    let dir = TempDir::new().unwrap();

    // Create files including ones that should be excluded
    fs::write(dir.path().join("keep.rs"), "fn main() {}").unwrap();
    fs::write(dir.path().join("keep.txt"), "hello").unwrap();
    fs::create_dir(dir.path().join("node_modules")).unwrap();
    fs::write(
        dir.path().join("node_modules/lib.js"),
        "module.exports = {}",
    )
    .unwrap();

    let source = FilesystemSource::new();

    // node_modules should be excluded by gitignore behavior
    let options = ScanOptions {
        paths: vec![dir.path().to_path_buf()],
        ..Default::default()
    };

    let items = source.discover(&options).unwrap();

    // Should find keep.rs and keep.txt, but not node_modules/lib.js
    // (hidden dirs are excluded by default in the ignore crate)
    let uris: Vec<String> = items.iter().map(|i| i.uri.clone()).collect();
    assert!(uris.iter().any(|u| u.contains("keep.rs")));
    assert!(uris.iter().any(|u| u.contains("keep.txt")));
}

#[test]
fn test_dry_run_doesnt_modify_state() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("test.txt"), "hello world").unwrap();

    let source = FilesystemSource::new();
    let options = ScanOptions {
        paths: vec![dir.path().to_path_buf()],
        dry_run: true,
        ..Default::default()
    };

    let items = source.discover(&options).unwrap();

    // Discovery still happens in dry run
    assert_eq!(items.len(), 1);
}

// ============================================================================
// Feature 1: Parallel Scan Pipeline (rayon)
// ============================================================================

#[test]
fn test_parallel_scan_produces_chunks() {
    // Verify that scanning with multiple jobs still discovers and chunks files
    let dir = TempDir::new().unwrap();
    create_test_corpus(&dir);

    let source = FilesystemSource::new();
    let options = ScanOptions {
        paths: vec![dir.path().to_path_buf()],
        jobs: 4,
        ..Default::default()
    };

    let items = source.discover(&options).unwrap();

    assert!(items.len() >= 8, "Should discover at least 8 files");

    // Parse and chunk with rayon via parallel iteration
    let registry = ParserRegistry::new();
    let results: Vec<_> = items
        .iter()
        .filter_map(|item| {
            let content = fs::read(&item.path).ok()?;
            let doc = registry
                .parse(
                    &content,
                    item.mime_type.as_deref(),
                    item.extension.as_deref(),
                )
                .ok()?;
            let chunker = chunker_for_content(doc.content_type, 512, 64);
            let raw_chunks = chunker.chunk(&doc.text);
            Some((item.uri.clone(), raw_chunks))
        })
        .collect();

    assert!(!results.is_empty(), "Should parse some files");
    let total_chunks: usize = results.iter().map(|(_, c)| c.len()).sum();
    assert!(total_chunks > 0, "Should produce chunks");
}

#[test]
fn test_parallel_and_sequential_produce_same_file_count() {
    // Both jobs=1 and jobs=4 should discover the same files
    let dir = TempDir::new().unwrap();
    create_test_corpus(&dir);

    let source = FilesystemSource::new();

    let discover = |jobs: usize| -> usize {
        let options = ScanOptions {
            paths: vec![dir.path().to_path_buf()],
            jobs,
            ..Default::default()
        };
        source.discover(&options).unwrap().len()
    };

    let count_1 = discover(1);
    let count_4 = discover(4);

    assert_eq!(
        count_1, count_4,
        "Sequential and parallel discovery should find same number of files"
    );
}

// ============================================================================
// Feature 2: Wire Up --model Flag
// ============================================================================

#[cfg(feature = "fulltext")]
#[test]
fn test_model_override_nonexistent_returns_gracefully() {
    // When a model is specified but doesn't exist, the pipeline should
    // still function (falling back to keyword-only)
    let store = SimpleVectorStore::new();
    let tantivy = TantivyStore::open_in_memory().unwrap();
    let engine = HybridSearchEngine::new(store, tantivy, 0.7);

    // Insert with zero vectors (simulating no embedder)
    let chunks = vec![EmbeddedChunk {
        chunk: Chunk {
            text: "search for this text please".to_string(),
            source_uri: "file:///test.txt".to_string(),
            chunk_index: 0,
            content_type: ContentType::Text,
            file_type: "txt".to_string(),
            title: None,
            language: None,
            byte_range: None,
        },
        vector: vec![0.0f32; 768],
    }];

    engine.insert(&chunks).unwrap();

    // Keyword-only search should still work without embedder
    let results = engine
        .search(
            &vec![0.0f32; 768],
            "search text",
            5,
            SearchMode::KeywordOnly,
        )
        .unwrap();
    assert!(!results.is_empty());
}

// ============================================================================
// Feature 3: Wire Up --context Flag
// ============================================================================

#[test]
fn test_search_options_context_field() {
    let opts = SearchOptions {
        query: "test".to_string(),
        context: true,
        ..Default::default()
    };
    assert!(opts.context);

    let opts_default = SearchOptions::default();
    assert!(!opts_default.context);
}

// ============================================================================
// Feature 5: Add --after Flag to Search
// ============================================================================

#[test]
fn test_after_filter_with_metadata_store() {
    let meta = MetadataStore::open_in_memory().unwrap();
    let hash = [0u8; 32];

    // Insert sources with different modification times
    meta.upsert_source("file:///old.rs", &hash, 100, "rs", Some(1000), 3)
        .unwrap();
    meta.upsert_source("file:///recent.rs", &hash, 200, "rs", Some(5000), 5)
        .unwrap();
    meta.upsert_source("file:///newest.rs", &hash, 150, "rs", Some(9000), 2)
        .unwrap();

    // Filter: only files modified after timestamp 4000
    let allowed = meta.uris_modified_after(4000).unwrap();
    assert_eq!(allowed.len(), 2);
    assert!(allowed.contains("file:///recent.rs"));
    assert!(allowed.contains("file:///newest.rs"));
    assert!(!allowed.contains("file:///old.rs"));
}

#[test]
fn test_after_filter_applied_to_search_results() {
    // Simulate the search flow: get results, then filter by after_ts
    let meta = MetadataStore::open_in_memory().unwrap();
    let hash = [0u8; 32];

    meta.upsert_source("file:///old.txt", &hash, 10, "txt", Some(100), 1)
        .unwrap();
    meta.upsert_source("file:///new.txt", &hash, 10, "txt", Some(5000), 1)
        .unwrap();

    let store = SimpleVectorStore::new();
    let chunks = vec![
        EmbeddedChunk {
            chunk: Chunk {
                text: "old content".to_string(),
                source_uri: "file:///old.txt".to_string(),
                chunk_index: 0,
                content_type: ContentType::Text,
                file_type: "txt".to_string(),
                title: None,
                language: None,
                byte_range: None,
            },
            vector: vec![1.0, 0.0, 0.0],
        },
        EmbeddedChunk {
            chunk: Chunk {
                text: "new content".to_string(),
                source_uri: "file:///new.txt".to_string(),
                chunk_index: 0,
                content_type: ContentType::Text,
                file_type: "txt".to_string(),
                title: None,
                language: None,
                byte_range: None,
            },
            vector: vec![0.0, 1.0, 0.0],
        },
    ];

    store.insert(&chunks).unwrap();

    // Get all results
    let mut results = store.search(&[0.5, 0.5, 0.0], 10).unwrap();
    assert_eq!(results.len(), 2);

    // Apply after filter (after_ts = 1000)
    let allowed = meta.uris_modified_after(1000).unwrap();
    results.retain(|r| allowed.contains(&r.uri));

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].uri, "file:///new.txt");
}

// ============================================================================
// Feature 6: Add sift export Command
// ============================================================================

#[test]
fn test_export_all_basic() {
    let store = SimpleVectorStore::new();

    let chunks = vec![
        EmbeddedChunk {
            chunk: Chunk {
                text: "first chunk".to_string(),
                source_uri: "file:///a.txt".to_string(),
                chunk_index: 0,
                content_type: ContentType::Text,
                file_type: "txt".to_string(),
                title: Some("File A".to_string()),
                language: None,
                byte_range: Some((0, 11)),
            },
            vector: vec![1.0, 0.0],
        },
        EmbeddedChunk {
            chunk: Chunk {
                text: "second chunk".to_string(),
                source_uri: "file:///b.rs".to_string(),
                chunk_index: 0,
                content_type: ContentType::Code,
                file_type: "rs".to_string(),
                title: None,
                language: Some("rust".to_string()),
                byte_range: Some((0, 12)),
            },
            vector: vec![0.0, 1.0],
        },
    ];

    store.insert(&chunks).unwrap();

    let mut entries = store.export_all().unwrap();
    entries.sort_by(|a, b| a.uri.cmp(&b.uri));
    assert_eq!(entries.len(), 2);

    // Verify first entry (sorted by URI)
    assert_eq!(entries[0].uri, "file:///a.txt");
    assert_eq!(entries[0].text, "first chunk");
    assert_eq!(entries[0].chunk_index, 0);
    assert_eq!(entries[0].content_type, ContentType::Text);
    assert_eq!(entries[0].file_type, "txt");
    assert_eq!(entries[0].title.as_deref(), Some("File A"));
    assert_eq!(entries[0].byte_range, Some((0, 11)));
    assert_eq!(entries[0].vector, vec![1.0, 0.0]);

    // Verify second entry
    assert_eq!(entries[1].uri, "file:///b.rs");
    assert_eq!(entries[1].content_type, ContentType::Code);
    assert!(entries[1].title.is_none());
}

#[test]
fn test_export_jsonl_format() {
    // Simulate what the export command does: serialize entries as JSONL
    let store = SimpleVectorStore::new();

    let chunks = vec![
        EmbeddedChunk {
            chunk: Chunk {
                text: "test content".to_string(),
                source_uri: "file:///test.txt".to_string(),
                chunk_index: 0,
                content_type: ContentType::Text,
                file_type: "txt".to_string(),
                title: Some("Test Title".to_string()),
                language: None,
                byte_range: Some((10, 22)),
            },
            vector: vec![0.5, 0.5],
        },
        EmbeddedChunk {
            chunk: Chunk {
                text: "code content".to_string(),
                source_uri: "file:///main.rs".to_string(),
                chunk_index: 1,
                content_type: ContentType::Code,
                file_type: "rs".to_string(),
                title: None,
                language: Some("rust".to_string()),
                byte_range: None,
            },
            vector: vec![0.3, 0.7],
        },
    ];

    store.insert(&chunks).unwrap();
    let mut entries = store.export_all().unwrap();
    entries.sort_by(|a, b| a.uri.cmp(&b.uri));

    // Serialize as JSONL (without vectors)
    let mut output = Vec::new();
    for entry in &entries {
        let mut obj = serde_json::json!({
            "uri": entry.uri,
            "text": entry.text,
            "chunk_index": entry.chunk_index,
            "content_type": entry.content_type,
            "file_type": entry.file_type,
        });
        if let Some(ref title) = entry.title {
            obj["title"] = serde_json::Value::String(title.clone());
        }
        if let Some((start, end)) = entry.byte_range {
            obj["byte_range"] = serde_json::json!([start, end]);
        }
        let line = serde_json::to_string(&obj).unwrap();
        writeln!(output, "{}", line).unwrap();
    }

    // Parse back and verify JSONL
    let reader = std::io::BufReader::new(output.as_slice());
    let mut lines: Vec<serde_json::Value> = reader
        .lines()
        .map(|l| serde_json::from_str(&l.unwrap()).unwrap())
        .collect();
    lines.sort_by(|a, b| a["uri"].as_str().cmp(&b["uri"].as_str()));

    assert_eq!(lines.len(), 2);
    // Sorted by URI: file:///main.rs < file:///test.txt
    assert_eq!(lines[0]["uri"], "file:///main.rs");
    assert!(lines[0].get("title").is_none()); // None title not included
    assert!(lines[0].get("byte_range").is_none()); // None byte_range not included

    assert_eq!(lines[1]["uri"], "file:///test.txt");
    assert_eq!(lines[1]["title"], "Test Title");
    assert_eq!(lines[1]["byte_range"][0], 10);
    assert_eq!(lines[1]["byte_range"][1], 22);
    assert!(lines[1].get("vector").is_none()); // no vectors by default
}

#[test]
fn test_export_with_vectors() {
    let store = SimpleVectorStore::new();

    store
        .insert(&[EmbeddedChunk {
            chunk: Chunk {
                text: "vectorized".to_string(),
                source_uri: "file:///v.txt".to_string(),
                chunk_index: 0,
                content_type: ContentType::Text,
                file_type: "txt".to_string(),
                title: None,
                language: None,
                byte_range: None,
            },
            vector: vec![0.1, 0.2, 0.3],
        }])
        .unwrap();

    let entries = store.export_all().unwrap();

    // Serialize with vectors
    let entry = &entries[0];
    let obj = serde_json::json!({
        "uri": entry.uri,
        "text": entry.text,
        "chunk_index": entry.chunk_index,
        "content_type": entry.content_type,
        "file_type": entry.file_type,
        "vector": entry.vector,
    });

    let parsed: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&obj).unwrap()).unwrap();
    let vec_arr = parsed["vector"].as_array().unwrap();
    assert_eq!(vec_arr.len(), 3);
    assert!((vec_arr[0].as_f64().unwrap() - 0.1).abs() < 1e-6);
    assert!((vec_arr[1].as_f64().unwrap() - 0.2).abs() < 1e-6);
    assert!((vec_arr[2].as_f64().unwrap() - 0.3).abs() < 1e-6);
}

#[test]
fn test_export_file_type_filter() {
    let store = SimpleVectorStore::new();

    store
        .insert(&[
            EmbeddedChunk {
                chunk: Chunk {
                    text: "rust code".to_string(),
                    source_uri: "file:///main.rs".to_string(),
                    chunk_index: 0,
                    content_type: ContentType::Code,
                    file_type: "rs".to_string(),
                    title: None,
                    language: None,
                    byte_range: None,
                },
                vector: vec![1.0],
            },
            EmbeddedChunk {
                chunk: Chunk {
                    text: "python code".to_string(),
                    source_uri: "file:///main.py".to_string(),
                    chunk_index: 0,
                    content_type: ContentType::Code,
                    file_type: "py".to_string(),
                    title: None,
                    language: None,
                    byte_range: None,
                },
                vector: vec![1.0],
            },
            EmbeddedChunk {
                chunk: Chunk {
                    text: "readme text".to_string(),
                    source_uri: "file:///readme.md".to_string(),
                    chunk_index: 0,
                    content_type: ContentType::Text,
                    file_type: "md".to_string(),
                    title: None,
                    language: None,
                    byte_range: None,
                },
                vector: vec![1.0],
            },
        ])
        .unwrap();

    let entries = store.export_all().unwrap();
    assert_eq!(entries.len(), 3);

    // Simulate file type filter (what export command does)
    let rs_only: Vec<&ExportEntry> = entries.iter().filter(|e| e.file_type == "rs").collect();
    assert_eq!(rs_only.len(), 1);
    assert_eq!(rs_only[0].uri, "file:///main.rs");

    let py_only: Vec<&ExportEntry> = entries.iter().filter(|e| e.file_type == "py").collect();
    assert_eq!(py_only.len(), 1);
    assert_eq!(py_only[0].uri, "file:///main.py");
}

#[test]
fn test_export_to_file() {
    let dir = TempDir::new().unwrap();
    let output_path = dir.path().join("export.jsonl");

    let store = SimpleVectorStore::new();
    store
        .insert(&[
            EmbeddedChunk {
                chunk: Chunk {
                    text: "chunk one".to_string(),
                    source_uri: "file:///a.txt".to_string(),
                    chunk_index: 0,
                    content_type: ContentType::Text,
                    file_type: "txt".to_string(),
                    title: None,
                    language: None,
                    byte_range: None,
                },
                vector: vec![1.0, 0.0],
            },
            EmbeddedChunk {
                chunk: Chunk {
                    text: "chunk two".to_string(),
                    source_uri: "file:///b.txt".to_string(),
                    chunk_index: 0,
                    content_type: ContentType::Text,
                    file_type: "txt".to_string(),
                    title: None,
                    language: None,
                    byte_range: None,
                },
                vector: vec![0.0, 1.0],
            },
        ])
        .unwrap();

    // Write JSONL to file
    let entries = store.export_all().unwrap();
    let mut file = fs::File::create(&output_path).unwrap();
    for entry in &entries {
        let obj = serde_json::json!({
            "uri": entry.uri,
            "text": entry.text,
            "chunk_index": entry.chunk_index,
            "content_type": entry.content_type,
            "file_type": entry.file_type,
        });
        writeln!(file, "{}", serde_json::to_string(&obj).unwrap()).unwrap();
    }

    // Verify file content (sort by URI since HashMap order is non-deterministic)
    let content = fs::read_to_string(&output_path).unwrap();
    let mut parsed: Vec<serde_json::Value> = content
        .trim()
        .split('\n')
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    parsed.sort_by(|a, b| a["uri"].as_str().cmp(&b["uri"].as_str()));
    assert_eq!(parsed.len(), 2);

    assert_eq!(parsed[0]["uri"], "file:///a.txt");
    assert_eq!(parsed[0]["text"], "chunk one");

    assert_eq!(parsed[1]["uri"], "file:///b.txt");
    assert_eq!(parsed[1]["text"], "chunk two");
}

// ============================================================================
// CLI End-to-End Tests (assert_cmd)
// ============================================================================

/// Helper to get a `Command` for the `sift` binary with a unique index.
/// Sets ORT_LOG_LEVEL to prevent ONNX Runtime from writing to stdout,
/// which would corrupt structured (JSON/CSV) output parsing in tests.
fn sift_cmd(index_name: &str) -> Command {
    let mut cmd = Command::new(cargo_bin!("sift"));
    cmd.arg("--index")
        .arg(index_name)
        .env("ORT_LOG_LEVEL", "fatal");
    cmd
}

/// Extract a JSON value from stdout that may contain leading non-JSON lines
/// (e.g. ONNX Runtime log output). Finds the first `[` or `{` and parses from there.
fn parse_json_stdout(stdout: &str) -> serde_json::Value {
    let trimmed = stdout.trim();
    // Try parsing as-is first
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
        return v;
    }
    // Find first JSON start character
    if let Some(pos) = trimmed.find(['[', '{']) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&trimmed[pos..]) {
            return v;
        }
    }
    panic!(
        "Could not find valid JSON in stdout:\n{}",
        &trimmed[..trimmed.len().min(500)]
    );
}

/// Create a test corpus and return the TempDir (must stay alive for the test).
fn setup_cli_corpus() -> TempDir {
    let dir = TempDir::new().unwrap();
    create_test_corpus(&dir);
    dir
}

#[test]
fn test_cli_scan_and_status() {
    let dir = setup_cli_corpus();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    // Scan
    sift_cmd(&idx)
        .args(["scan", dir.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("Indexed"))
        .stdout(predicate::str::contains("files"));

    // Status
    sift_cmd(&idx)
        .arg("status")
        .assert()
        .success()
        .stdout(predicate::str::contains("Index Status"))
        .stdout(predicate::str::contains("Sources:"))
        .stdout(predicate::str::contains("Chunks:"));
}

#[test]
fn test_cli_list_after_scan() {
    let dir = setup_cli_corpus();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    sift_cmd(&idx)
        .args(["scan", dir.path().to_str().unwrap()])
        .assert()
        .success();

    sift_cmd(&idx)
        .arg("list")
        .assert()
        .success()
        .stdout(predicate::str::contains("sources indexed"))
        .stdout(predicate::str::contains("chunks)"));
}

#[test]
fn test_cli_search_keyword_only() {
    let dir = setup_cli_corpus();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    sift_cmd(&idx)
        .args(["scan", dir.path().to_str().unwrap()])
        .assert()
        .success();

    sift_cmd(&idx)
        .args(["search", "--keyword-only", "payment processing"])
        .assert()
        .success()
        .stdout(predicate::str::contains("payment"));
}

#[test]
fn test_cli_search_json_output() {
    let dir = setup_cli_corpus();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    sift_cmd(&idx)
        .args(["scan", dir.path().to_str().unwrap()])
        .assert()
        .success();

    let output = sift_cmd(&idx)
        .args(["search", "--keyword-only", "--json", "revenue"])
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let parsed = parse_json_stdout(&stdout);
    assert!(parsed.is_array());
}

#[test]
fn test_cli_search_type_filter() {
    let dir = setup_cli_corpus();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    sift_cmd(&idx)
        .args(["scan", dir.path().to_str().unwrap()])
        .assert()
        .success();

    let output = sift_cmd(&idx)
        .args([
            "search",
            "--keyword-only",
            "--json",
            "--type",
            "rs",
            "config",
        ])
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let parsed: Vec<serde_json::Value> = parse_json_stdout(&stdout)
        .as_array()
        .expect("expected JSON array")
        .clone();
    for result in &parsed {
        assert_eq!(result["file_type"], "rs");
    }
}

#[test]
fn test_cli_search_max_results() {
    let dir = setup_cli_corpus();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    sift_cmd(&idx)
        .args(["scan", dir.path().to_str().unwrap()])
        .assert()
        .success();

    let output = sift_cmd(&idx)
        .args([
            "search",
            "--keyword-only",
            "--json",
            "--max-results",
            "1",
            "the",
        ])
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let parsed: Vec<serde_json::Value> = parse_json_stdout(&stdout)
        .as_array()
        .expect("expected JSON array")
        .clone();
    assert!(parsed.len() <= 1);
}

#[test]
fn test_cli_search_no_results() {
    let dir = setup_cli_corpus();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    sift_cmd(&idx)
        .args(["scan", dir.path().to_str().unwrap()])
        .assert()
        .success();

    sift_cmd(&idx)
        .args(["search", "--keyword-only", "zzzznonexistentqueryzzz"])
        .assert()
        .success()
        .stdout(predicate::str::contains("No results found"));
}

#[test]
fn test_cli_search_path_glob() {
    let dir = setup_cli_corpus();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    sift_cmd(&idx)
        .args(["scan", dir.path().to_str().unwrap()])
        .assert()
        .success();

    let output = sift_cmd(&idx)
        .args([
            "search",
            "--keyword-only",
            "--json",
            "--path",
            "*/src/*",
            "version",
        ])
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let parsed: Vec<serde_json::Value> = parse_json_stdout(&stdout)
        .as_array()
        .expect("expected JSON array")
        .clone();
    for result in &parsed {
        let uri = result["uri"].as_str().unwrap();
        assert!(uri.contains("/src/"), "URI should match path glob: {}", uri);
    }
}

#[test]
fn test_cli_search_after_filter() {
    let dir = setup_cli_corpus();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    sift_cmd(&idx)
        .args(["scan", dir.path().to_str().unwrap()])
        .assert()
        .success();

    // Files were just created, so --after 1d should include them
    sift_cmd(&idx)
        .args(["search", "--keyword-only", "--after", "1d", "config"])
        .assert()
        .success();

    // --after with a future date should return no results
    let output = sift_cmd(&idx)
        .args([
            "search",
            "--keyword-only",
            "--json",
            "--after",
            "2099-01-01",
            "config",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let parsed: Vec<serde_json::Value> = parse_json_stdout(&stdout)
        .as_array()
        .expect("expected JSON array")
        .clone();
    assert!(
        parsed.is_empty(),
        "Future date should filter out all results"
    );
}

#[test]
fn test_cli_scan_dry_run() {
    let dir = setup_cli_corpus();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    sift_cmd(&idx)
        .args(["scan", "--dry-run", dir.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("Dry run"))
        .stdout(predicate::str::contains("Would index"));

    // After dry run, status should show 0 sources
    sift_cmd(&idx)
        .arg("status")
        .assert()
        .success()
        .stdout(predicate::str::contains("Sources:").and(predicate::str::contains("0")));
}

#[test]
fn test_cli_scan_type_filter() {
    let dir = setup_cli_corpus();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    sift_cmd(&idx)
        .args(["scan", "--type", "rs", dir.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("rs:"));

    // List should only show .rs files
    let output = sift_cmd(&idx)
        .args(["--format", "json", "list"])
        .output()
        .unwrap();

    let stdout = String::from_utf8(output.stdout).unwrap();
    let parsed: Vec<serde_json::Value> = parse_json_stdout(&stdout)
        .as_array()
        .expect("expected JSON array")
        .clone();
    for item in &parsed {
        assert_eq!(item["file_type"], "rs");
    }
}

#[test]
fn test_cli_scan_include_exclude() {
    let dir = setup_cli_corpus();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    // Include only .md files
    sift_cmd(&idx)
        .args(["scan", "--include", "*.md", dir.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("md:"));

    let idx2 = format!("cli-test-{}", uuid::Uuid::now_v7());

    // Exclude .rs files
    sift_cmd(&idx2)
        .args(["scan", "--exclude", "*.rs", dir.path().to_str().unwrap()])
        .assert()
        .success();

    let output = sift_cmd(&idx2)
        .args(["--format", "json", "list"])
        .output()
        .unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    let parsed: Vec<serde_json::Value> = parse_json_stdout(&stdout)
        .as_array()
        .expect("expected JSON array")
        .clone();
    for item in &parsed {
        assert_ne!(item["file_type"], "rs", "Excluded .rs should not appear");
    }
}

#[test]
fn test_cli_scan_max_depth() {
    let dir = setup_cli_corpus();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    // max-depth 1 should not descend into src/
    sift_cmd(&idx)
        .args(["scan", "--max-depth", "1", dir.path().to_str().unwrap()])
        .assert()
        .success();

    let output = sift_cmd(&idx)
        .args(["--format", "json", "list"])
        .output()
        .unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    let parsed: Vec<serde_json::Value> = parse_json_stdout(&stdout)
        .as_array()
        .expect("expected JSON array")
        .clone();
    for item in &parsed {
        let uri = item["uri"].as_str().unwrap();
        assert!(
            !uri.contains("/src/"),
            "max-depth 1 should not include nested files: {}",
            uri
        );
    }
}

#[test]
fn test_cli_incremental_scan() {
    let dir = setup_cli_corpus();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    // First scan
    sift_cmd(&idx)
        .args(["scan", dir.path().to_str().unwrap()])
        .assert()
        .success();

    // Second scan should skip unchanged files
    sift_cmd(&idx)
        .args(["scan", dir.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("skipped"));
}

#[test]
fn test_cli_remove_source() {
    let dir = setup_cli_corpus();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    sift_cmd(&idx)
        .args(["scan", dir.path().to_str().unwrap()])
        .assert()
        .success();

    let readme_path = dir.path().join("readme.md");
    sift_cmd(&idx)
        .args(["remove", readme_path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("Removed"));

    // Verify it's gone from the list
    let output = sift_cmd(&idx)
        .args(["--format", "json", "list"])
        .output()
        .unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(!stdout.contains("readme.md"));
}

#[test]
fn test_cli_export_jsonl() {
    let dir = setup_cli_corpus();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    sift_cmd(&idx)
        .args(["scan", dir.path().to_str().unwrap()])
        .assert()
        .success();

    let output = sift_cmd(&idx).arg("export").output().unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    // Each line should be valid JSON
    for line in stdout.trim().lines() {
        let parsed: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("Invalid JSONL line: {}\nError: {}", line, e));
        assert!(parsed.get("uri").is_some());
        assert!(parsed.get("text").is_some());
    }
}

#[test]
fn test_cli_export_to_file() {
    let dir = setup_cli_corpus();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());
    let export_dir = TempDir::new().unwrap();
    let export_path = export_dir.path().join("out.jsonl");

    sift_cmd(&idx)
        .args(["scan", dir.path().to_str().unwrap()])
        .assert()
        .success();

    sift_cmd(&idx)
        .args(["export", "-o", export_path.to_str().unwrap()])
        .assert()
        .success();

    let content = fs::read_to_string(&export_path).unwrap();
    assert!(!content.is_empty());
    let line_count = content.trim().lines().count();
    assert!(line_count > 0, "Export file should have at least one line");
}

#[test]
fn test_cli_export_type_filter() {
    let dir = setup_cli_corpus();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    sift_cmd(&idx)
        .args(["scan", dir.path().to_str().unwrap()])
        .assert()
        .success();

    let output = sift_cmd(&idx)
        .args(["export", "--type", "rs"])
        .output()
        .unwrap();

    let stdout = String::from_utf8(output.stdout).unwrap();
    for line in stdout.trim().lines() {
        let parsed: serde_json::Value = serde_json::from_str(line).unwrap();
        assert_eq!(parsed["file_type"], "rs");
    }
}

#[test]
fn test_cli_export_with_vectors() {
    let dir = setup_cli_corpus();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    sift_cmd(&idx)
        .args(["scan", dir.path().to_str().unwrap()])
        .assert()
        .success();

    let output = sift_cmd(&idx)
        .args(["export", "--vectors"])
        .output()
        .unwrap();

    let stdout = String::from_utf8(output.stdout).unwrap();
    for line in stdout.trim().lines() {
        let parsed: serde_json::Value = serde_json::from_str(line).unwrap();
        // With --vectors, vector field should be present
        assert!(
            parsed.get("vector").is_some(),
            "Should include vector field"
        );
    }
}

#[test]
fn test_cli_config_get_set() {
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    // Get a default value
    sift_cmd(&idx)
        .args(["config", "search.max_results"])
        .assert()
        .success()
        .stdout(predicate::str::contains("search.max_results = "));

    // Print full config
    sift_cmd(&idx)
        .arg("config")
        .assert()
        .success()
        .stdout(predicate::str::contains("[default]"))
        .stdout(predicate::str::contains("[search]"))
        .stdout(predicate::str::contains("[server]"));
}

#[test]
fn test_cli_config_unknown_key() {
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    sift_cmd(&idx)
        .args(["config", "nonexistent.key"])
        .assert()
        .success()
        .stdout(predicate::str::contains("key not found"));
}

#[test]
fn test_cli_config_set_invalid_value() {
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    sift_cmd(&idx)
        .args(["config", "search.max_results", "not_a_number"])
        .assert()
        .failure();
}

#[test]
fn test_cli_config_set_unknown_key() {
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    sift_cmd(&idx)
        .args(["config", "fake.key", "value"])
        .assert()
        .failure();
}

#[test]
fn test_cli_status_empty_index() {
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    sift_cmd(&idx)
        .arg("status")
        .assert()
        .success()
        .stdout(predicate::str::contains("No index found"));
}

#[test]
fn test_cli_status_json() {
    let dir = setup_cli_corpus();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    sift_cmd(&idx)
        .args(["scan", dir.path().to_str().unwrap()])
        .assert()
        .success();

    let output = sift_cmd(&idx)
        .args(["--format", "json", "status"])
        .output()
        .unwrap();

    let stdout = String::from_utf8(output.stdout).unwrap();
    let parsed = parse_json_stdout(&stdout);
    assert!(parsed["total_sources"].as_u64().unwrap() > 0);
    assert!(parsed["total_chunks"].as_u64().unwrap() > 0);
}

#[test]
fn test_cli_list_json() {
    let dir = setup_cli_corpus();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    sift_cmd(&idx)
        .args(["scan", dir.path().to_str().unwrap()])
        .assert()
        .success();

    let output = sift_cmd(&idx)
        .args(["--format", "json", "list"])
        .output()
        .unwrap();

    let stdout = String::from_utf8(output.stdout).unwrap();
    let parsed: Vec<serde_json::Value> = parse_json_stdout(&stdout)
        .as_array()
        .expect("expected JSON array")
        .clone();
    assert!(!parsed.is_empty());
    for item in &parsed {
        assert!(item.get("uri").is_some());
        assert!(item.get("file_type").is_some());
        assert!(item.get("chunks").is_some());
    }
}

#[test]
fn test_cli_help_and_version() {
    Command::new(cargo_bin!("sift"))
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("sift"));

    Command::new(cargo_bin!("sift"))
        .arg("--version")
        .assert()
        .success();
}

#[test]
fn test_cli_scan_nonexistent_path() {
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    // Scanning a nonexistent path discovers 0 files (not an error)
    sift_cmd(&idx)
        .args(["scan", "/nonexistent/path/that/does/not/exist"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Indexed 0 files"));
}

#[test]
fn test_cli_remove_nonexistent() {
    let dir = setup_cli_corpus();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    sift_cmd(&idx)
        .args(["scan", dir.path().to_str().unwrap()])
        .assert()
        .success();

    sift_cmd(&idx)
        .args(["remove", "/nonexistent/file.txt"])
        .assert()
        .success()
        .stdout(
            predicate::str::contains("0 not found").or(predicate::str::contains("1 not found")),
        );
}

#[test]
fn test_cli_search_context_flag() {
    let dir = setup_cli_corpus();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    sift_cmd(&idx)
        .args(["scan", dir.path().to_str().unwrap()])
        .assert()
        .success();

    // --context should show line numbers with > markers
    sift_cmd(&idx)
        .args(["search", "--keyword-only", "--context", "server"])
        .assert()
        .success();
}

#[test]
fn test_cli_search_threshold_filter() {
    let dir = setup_cli_corpus();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    sift_cmd(&idx)
        .args(["scan", dir.path().to_str().unwrap()])
        .assert()
        .success();

    // Very high threshold should return fewer/no results
    let output = sift_cmd(&idx)
        .args([
            "search",
            "--keyword-only",
            "--json",
            "--threshold",
            "999.0",
            "server",
        ])
        .output()
        .unwrap();

    let stdout = String::from_utf8(output.stdout).unwrap();
    let parsed: Vec<serde_json::Value> = parse_json_stdout(&stdout)
        .as_array()
        .expect("expected JSON array")
        .clone();
    assert!(
        parsed.is_empty(),
        "Very high threshold should filter out all results"
    );
}

#[test]
fn test_cli_quiet_suppresses_logs() {
    let dir = setup_cli_corpus();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    let output = sift_cmd(&idx)
        .args(["--quiet", "scan", dir.path().to_str().unwrap()])
        .output()
        .unwrap();

    assert!(output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    // Quiet mode should suppress INFO-level log lines
    assert!(
        !stderr.contains("INFO"),
        "Quiet mode should suppress INFO logs"
    );
}

#[test]
fn test_cli_sift_index_env() {
    let dir = setup_cli_corpus();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    // Use SIFT_INDEX env var instead of --index flag
    Command::new(cargo_bin!("sift"))
        .env("SIFT_INDEX", &idx)
        .env("ORT_LOG_LEVEL", "fatal")
        .args(["scan", dir.path().to_str().unwrap()])
        .assert()
        .success();

    Command::new(cargo_bin!("sift"))
        .env("SIFT_INDEX", &idx)
        .env("ORT_LOG_LEVEL", "fatal")
        .arg("status")
        .assert()
        .success()
        .stdout(predicate::str::contains("Sources:"));
}

#[test]
fn test_cli_sift_format_env() {
    let dir = setup_cli_corpus();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    sift_cmd(&idx)
        .args(["scan", dir.path().to_str().unwrap()])
        .assert()
        .success();

    // SIFT_FORMAT=json should produce JSON output
    let output = Command::new(cargo_bin!("sift"))
        .env("SIFT_FORMAT", "json")
        .env("ORT_LOG_LEVEL", "fatal")
        .args(["--index", &idx, "status"])
        .output()
        .unwrap();

    let stdout = String::from_utf8(output.stdout).unwrap();
    let parsed = parse_json_stdout(&stdout);
    assert!(parsed.get("total_sources").is_some());
}

#[test]
fn test_cli_scan_empty_directory() {
    let dir = TempDir::new().unwrap();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    sift_cmd(&idx)
        .args(["scan", dir.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("Indexed 0 files"));
}

#[test]
fn test_cli_scan_single_file() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("hello.txt");
    fs::write(&file, "Hello, world! This is a test file.").unwrap();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    sift_cmd(&idx)
        .args(["scan", dir.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("Indexed 1 files"));
}

#[test]
fn test_cli_search_csv_output() {
    let dir = setup_cli_corpus();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    sift_cmd(&idx)
        .args(["scan", dir.path().to_str().unwrap()])
        .assert()
        .success();

    let output = sift_cmd(&idx)
        .args(["--format", "csv", "search", "--keyword-only", "server"])
        .output()
        .unwrap();

    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("uri,score,chunk_index,file_type,text"));
}

#[test]
fn test_cli_list_csv_output() {
    let dir = setup_cli_corpus();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    sift_cmd(&idx)
        .args(["scan", dir.path().to_str().unwrap()])
        .assert()
        .success();

    let output = sift_cmd(&idx)
        .args(["--format", "csv", "list"])
        .output()
        .unwrap();

    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("uri,file_type,chunks"));
}

#[test]
fn test_cli_status_csv_output() {
    let dir = setup_cli_corpus();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    sift_cmd(&idx)
        .args(["scan", dir.path().to_str().unwrap()])
        .assert()
        .success();

    let output = sift_cmd(&idx)
        .args(["--format", "csv", "status"])
        .output()
        .unwrap();

    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("total_sources,total_chunks,index_size_bytes"));
}

#[test]
fn test_cli_scan_multiple_paths() {
    let dir1 = TempDir::new().unwrap();
    let dir2 = TempDir::new().unwrap();
    fs::write(dir1.path().join("a.txt"), "File A content").unwrap();
    fs::write(dir2.path().join("b.txt"), "File B content").unwrap();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    sift_cmd(&idx)
        .args([
            "scan",
            dir1.path().to_str().unwrap(),
            dir2.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Indexed 2 files"));
}

#[test]
fn test_cli_remove_multiple() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("a.txt"), "aaa").unwrap();
    fs::write(dir.path().join("b.txt"), "bbb").unwrap();
    fs::write(dir.path().join("c.txt"), "ccc").unwrap();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    sift_cmd(&idx)
        .args(["scan", dir.path().to_str().unwrap()])
        .assert()
        .success();

    let a_path = dir.path().join("a.txt");
    let b_path = dir.path().join("b.txt");
    sift_cmd(&idx)
        .args(["remove", a_path.to_str().unwrap(), b_path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("Removed 2 source"));
}

// ============================================================================
// Error-path integration tests
// ============================================================================

#[test]
fn test_corrupt_vector_index_reports_error() {
    let dir = TempDir::new().unwrap();
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());

    // Scan some files first to create an index
    fs::write(dir.path().join("a.txt"), "hello").unwrap();
    sift_cmd(&idx)
        .args(["scan", dir.path().to_str().unwrap()])
        .assert()
        .success();

    // Corrupt the vector index
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap();
    let sift_dir = std::path::PathBuf::from(home)
        .join(".sift")
        .join("indexes")
        .join(&idx);
    let vectors_path = sift_dir.join("vectors.bin");
    if vectors_path.exists() {
        fs::write(&vectors_path, b"corrupted data").unwrap();
    }

    // Search should fail gracefully
    sift_cmd(&idx).args(["search", "test"]).assert().failure();
}

#[test]
fn test_search_on_nonexistent_index() {
    let idx = format!("cli-test-nonexistent-{}", uuid::Uuid::now_v7());
    // An empty/nonexistent index returns "No results found" with exit 0
    sift_cmd(&idx)
        .args(["search", "hello"])
        .assert()
        .success()
        .stdout(predicate::str::contains("No results"));
}

#[test]
fn test_scan_with_zero_valid_files() {
    let dir = TempDir::new().unwrap();
    // Create only hidden files (ignored by default)
    fs::write(dir.path().join(".hidden"), "secret").unwrap();

    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());
    sift_cmd(&idx)
        .args(["scan", dir.path().to_str().unwrap()])
        .assert()
        .success();
}

#[test]
fn test_scan_nonexistent_directory() {
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());
    // Scanning a nonexistent directory discovers 0 files, succeeds with empty output
    sift_cmd(&idx)
        .args(["scan", "/nonexistent/path/that/does/not/exist"])
        .assert()
        .success()
        .stdout(predicate::str::contains("0 files"));
}

#[test]
fn test_config_set_invalid_value() {
    let idx = format!("cli-test-{}", uuid::Uuid::now_v7());
    // Setting a non-numeric value for a numeric config key should fail
    sift_cmd(&idx)
        .args(["config", "set", "search.max_results", "not_a_number"])
        .assert()
        .failure();
}
