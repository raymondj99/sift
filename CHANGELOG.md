# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Per-language tree-sitter AST chunking (`ast-rust`, `ast-python`, etc.)
- SQLite FTS5 full-text search as default keyword engine
- Binary vector index format (`vectors.bin`) with JSON migration
- Feature-gated install profiles: minimal (6.6 MB), default (14 MB), standard, full
- Vision embedding support (Nomic Embed Vision v1.5)
- Audio metadata extraction (MP3, FLAC, WAV, OGG, AAC, M4A)
- HNSW approximate nearest-neighbor index
- HTTP API server with search and status endpoints
- Filesystem watcher for automatic re-indexing
- Shell completion generation (`--features completions`)
- JSONL export with optional vector output
- Date filtering (`--after 7d`, `--after 2025-01-01`)
- Context display (`--context`) for showing surrounding source lines
- Named indexes (`--index`)
- JSON/CSV output formats (`--format json`)

## [0.1.2] - 2026-04-04

### Added
- **sift-memory crate**: temporal knowledge graph for AI agent memory persistence.
  Entities, observations (with bi-temporal validity), and directed relations stored
  in SQLite. Hybrid recall via RRF fusion of vector + BM25 keyword search. Decay
  scoring at query time (recency x confidence). Deterministic consolidation
  (exact-text dedup, Jaccard-based contradiction detection). 28 tests.
- **MCP tool `sift_index_text`**: store arbitrary text directly in the search index
  with custom URIs (e.g., `memory://facts/...`). Supports auto-embedding when the
  model is available, with keyword-only fallback.
- **MCP tool `sift_delete`**: remove indexed content by exact URI match.
- **MCP tool `sift_list_sources`**: browse indexed files with optional path
  filtering and pagination.
- **MCP tool `sift_remember`**: store entities, observations, and relations in
  persistent memory via a single call.
- **MCP tool `sift_recall`**: semantic search over stored memories with temporal
  filtering, entity type filtering, confidence thresholds, and decay scoring.
- **MCP tool `sift_forget`**: soft-delete memory observations (sets `valid_until`,
  preserves audit trail).
- **MCP tool `sift_memory_status`**: memory store statistics (entity/observation/
  relation counts, type breakdown, age range).
- Enhanced MCP `get_info()` instructions with agent-optimized tool discovery
  guidance for all 10 tools.

### Changed
- MCP server expanded from 3 read-only tools to 10 tools (3 read + 3 write + 4
  memory), transforming sift from a search engine into a memory-capable system.

## [0.1.1] - 2026-04-04

### Changed
- **MCP server**: wrap search engine in an LRU cache (50 entries, 60 s TTL) so
  repeated agent queries avoid full re-search.
- **MCP server**: deduplicate file reads per request — multiple results from the
  same source file now share a single disk read.
- **Embedding**: `tokenize_batch` returns flat `Vec<i64>` tensors directly,
  eliminating 3×batch_size intermediate `Vec<Vec<i64>>` allocations per
  inference call.
- **Embedding cache**: `put_batch` reuses a single serialisation buffer across
  the batch instead of allocating `Vec<u8>` per embedding.
- **Chunking**: `force_split` and `apply_overlap` use `char_indices()` iterators
  instead of `chars().collect::<Vec<char>>()`, removing an O(text_len)
  allocation per chunk.
- **Pipeline**: `zero_vector_chunks` now queries the embedder for its actual
  dimensionality instead of hard-coding 768.

### Fixed
- Zero-vector fallback used hard-coded 768-dimension vectors regardless of
  the active model's output size. Now uses `embedder.dimensions()`.

### Added
- MCP input validation: reject unknown `mode`, `detail`, and `scope` values
  with descriptive error messages; validate query length and block path
  traversal attempts.
- 8 new tests covering MCP input validation edge cases.

## [0.1.0] - 2025-01-01

### Added
- Initial release
- Parallel scan/parse/chunk/embed pipeline
- 30+ file format parsers
- BM25 keyword search
- Cosine similarity vector search
- Hybrid search with Reciprocal Rank Fusion
- BLAKE3 incremental content hashing
- ONNX Runtime embedding (Nomic Embed Text v2)
- SQLite metadata storage
- CLI with scan, search, status, list, remove, config, export commands
