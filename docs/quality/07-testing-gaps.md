# 07: Testing Gaps

**Category**: Robustness
**Severity**: Medium
**Effort**: Medium
**Crates**: all

## Current State

The codebase has solid unit test coverage for happy paths: 43 CLI integration tests, per-module unit tests for types, config, filesystem, parsers, chunkers, stores, and the hybrid search engine. CI runs clippy, fmt, and tests with default and all features.

The gaps are in adversarial inputs, error recovery, and performance regression prevention.

## Problem

### 1. No property-based / fuzz testing

For a tool that processes arbitrary file content and user input, there are no property-based tests. Specific gaps:

| Component | Untested Properties |
|-----------|-------------------|
| `RecursiveChunker` | For any input text and chunk_size > 0, all original content must appear in at least one chunk; no chunk exceeds chunk_size; offsets are valid byte positions |
| `SemanticChunker` | Same as above; additionally: split_quality never panics on any byte position |
| `cosine_similarity` | Symmetric: cos(a,b) = cos(b,a); identity: cos(a,a) = 1.0 for non-zero a; bounded: -1.0 ≤ cos(a,b) ≤ 1.0 |
| `BM25Store` | Serialization roundtrip: deserialize(serialize(x)) == x for any valid state |
| `FlatVectorIndex` | Binary roundtrip: load(save(x)) == x; handles non-ASCII URIs, empty vectors, very long texts |
| `parse_after_date` | Any valid YYYY-MM-DD produces a timestamp ≥ 0 for dates ≥ 1970; relative durations produce timestamps < now |

**Fix**: Add `proptest` (or `quickcheck`) as a dev-dependency:

```rust
// sift-chunker/src/recursive.rs
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn all_content_preserved(text in "\\PC{1,2000}", chunk_size in 10..500usize) {
            let chunker = RecursiveChunker::new(chunk_size, 0);
            let chunks = chunker.chunk(&text);
            // Every word in the original should appear in some chunk
            for word in text.split_whitespace() {
                prop_assert!(chunks.iter().any(|(c, _)| c.contains(word)));
            }
        }

        #[test]
        fn no_chunk_exceeds_size(text in "\\PC{1,2000}", chunk_size in 10..500usize) {
            let chunker = RecursiveChunker::new(chunk_size, 0);
            let chunks = chunker.chunk(&text);
            for (chunk, _) in &chunks {
                prop_assert!(chunk.len() <= chunk_size + 50, // allow small overrun from merging
                    "chunk len {} exceeds {}", chunk.len(), chunk_size);
            }
        }
    }
}
```

### 2. No error-path testing in integration tests

The 2164-line integration test file tests happy paths (scan, search, list, status, remove, config, export). Missing:

| Scenario | What Could Break |
|----------|-----------------|
| Corrupt `vectors.bin` (truncated, bad magic, wrong dimension) | `FlatVectorIndex::load_bin` should return a clear error, not panic |
| Corrupt `metadata.db` (SQLite corruption) | Should recover gracefully |
| Read-only index directory | Should report permission error, not panic |
| Scan with zero valid files | Should produce clean output, not empty/weird results |
| Search on empty index | Already tested, but not for all output formats |
| Config set with invalid values | `sift config search.hybrid_alpha not_a_number` |
| Concurrent scans | Race condition detection (even if just warning) |
| Files deleted between discover and parse | Should skip gracefully with warning |

**Fix**: Add error-path integration tests:

```rust
#[test]
fn test_corrupt_vector_index_reports_error() {
    let dir = setup_index_dir();
    std::fs::write(dir.join("vectors.bin"), b"corrupted data").unwrap();
    Command::cargo_bin("sift").unwrap()
        .args(["--index", &index_name, "search", "test"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("Invalid binary vector file"));
}

#[test]
fn test_scan_readonly_directory() {
    let dir = setup_index_dir();
    // Make index dir read-only
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o444)).unwrap();
    Command::cargo_bin("sift").unwrap()
        .args(["--index", &index_name, "scan", "."])
        .assert()
        .failure();
}
```

### 3. No benchmarks for parse, chunk, or full pipeline

The only benchmark is `sift-store/benches/vector_search.rs` (cosine similarity). Missing:

| Component | What to Benchmark |
|-----------|------------------|
| `ParserRegistry::parse` | Throughput for text, code, JSON, HTML (bytes/sec) |
| `RecursiveChunker::chunk` | Throughput for various chunk sizes and text lengths |
| `CodeChunker::chunk_with_language` | Tree-sitter parsing + chunking time per language |
| Full pipeline | End-to-end scan time for the benchmark corpus (see `07-benchsuite.md`) |
| `FlatVectorIndex::search` | Already benchmarked, but add 768-dim realistic vectors |
| `BM25Store::search` | Keyword search latency vs index size |

**Fix**: Add criterion benchmarks to each crate:

```rust
// sift-parsers/benches/parse.rs
fn bench_parse_text(c: &mut Criterion) {
    let registry = ParserRegistry::new();
    let content = std::fs::read("benches/fixtures/large.md").unwrap();
    c.bench_function("parse_markdown_100kb", |b| {
        b.iter(|| registry.parse(&content, Some("text/markdown"), Some("md")))
    });
}
```

### 4. Missing edge case tests for existing modules

| Module | Missing Test |
|--------|-------------|
| `config.rs` | `Config::load_from` with malformed TOML (partial parse) |
| `config.rs` | `Config::save` when parent directory doesn't exist |
| `filesystem.rs` | Symlink handling (follow vs skip) |
| `filesystem.rs` | Files with no extension and no MIME |
| `filesystem.rs` | Very long file paths (>PATH_MAX) |
| `flat.rs` | Binary load with mismatched dimension count |
| `flat.rs` | Binary load with truncated entry data |
| `hybrid.rs` | RRF fusion with duplicate chunk_index values |
| `bm25.rs` | Search with special characters in query |
| `bm25.rs` | Tokenization of CJK / non-Latin text |
| `output.rs` | CSV output with text containing commas and newlines |

### 5. No CI coverage reporting

CI runs tests but doesn't measure or track coverage. Adding `cargo-tarpaulin` or `cargo-llvm-cov` would identify untested code paths:

```yaml
# .github/workflows/ci.yml
- name: Coverage
  run: cargo tarpaulin --workspace --out xml
- uses: codecov/codecov-action@v4
```

## Priority

1. **Property tests for chunkers** — chunkers are the most likely source of subtle bugs (off-by-one in offsets, content loss, infinite loops on edge-case input)
2. **Error-path integration tests** — prevents regressions in user-visible error messages
3. **Benchmarks** — needed before implementing any of the performance optimizations in docs/performance/ and docs/quality/03-04 to measure actual impact
4. **Edge case unit tests** — incremental; add as bugs are found
5. **Coverage reporting** — nice to have for tracking over time
