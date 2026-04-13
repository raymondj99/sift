# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.5] - 2026-04-13

### Added
- **Cortex automated memory system** — dual-path architecture with hot-path
  encoding (Claude Code hooks, <100ms, zero LLM cost) and cold-path
  consolidation (5-phase pipeline: episode processing, deduplication,
  episodic-to-semantic promotion, skill extraction, decay & pruning).
- **Three memory tiers**: episodic (raw session events), semantic (consolidated
  facts promoted by access frequency or age), procedural (learned workflow
  patterns, never auto-deleted).
- **Conflict detection**: `sift_remember` flags contradictions when new facts
  conflict with existing observations on the same entity.
- **Retrieval-dependent strengthening**: frequently recalled memories gain
  relevance via access counting and logarithmic boosting.
- **Agent reasoning layer**: `sift memory generate-rules` produces
  `.claude/rules/` files from consolidated memory (decisions, preferences,
  corrections, workflows). Auto-regenerated after each consolidation cycle.
- **Daemon file watcher**: when `watch.enabled = true`, the daemon monitors
  all previously-indexed directories and re-indexes modified files on change.
  Watch paths derived automatically from indexed sources; nested directories
  collapsed. New `[watch]` config section with `enabled` and `debounce_ms`.
- **Benchmark regression CI**: `sift bench all` (7 Cortex invariant tests) and
  CodingMem dry-run (memory pipeline integrity) run after tests pass.
- **Auto-generated Homebrew formula** in release CI with real SHA256 checksums.
- MCP end-to-end test: 33 checks across full pipeline.
- `sift init` template now includes `[watch]` and `[memory]` sections.
- Workspace metadata (`description`, `repository`, `homepage`) for crates.io.
- 18 new tests (839 → 857).

### Changed
- **Configurable semantic deduplication**: `semantic_dedup_threshold` config
  value is now wired into the consolidation engine (was defined but hardcoded
  at 0.92). `consolidate_with_threshold()` API for programmatic control.
- **Embedding-based semantic dedup**: observations with moderate text overlap
  (0.3–threshold) are now also checked via cosine similarity. Catches
  semantically equivalent observations with different wording that pure
  Jaccard text comparison missed.
- `WatchDaemon::run_with_shutdown()` accepts `Arc<AtomicBool>` for graceful
  shutdown with pending change flush.
- README: added daemon mode section, updated config reference with `[watch]`
  and `[memory]`.

### Fixed
- Homebrew formula: corrected test command (`vx` → `sift`), Linux target
  (`musl` → `gnu`), added `v` prefix to release URLs.
- Tantivy query parser input sanitization for unmatched parentheses.

## [0.1.4] - 2026-04-05

### Added
- **Daemon mode** (`sift daemon start|stop|status`): persistent background process
  that keeps the embedding model and search index hot in memory, eliminating the
  1–5 s cold-start on every CLI invocation. Serves the full HTTP API over a Unix
  domain socket (`~/.sift/daemon.sock`) with graceful shutdown on SIGTERM/SIGINT
  and PID-file lifecycle management.
- **Transparent daemon routing**: read operations (`search`, `status`, `list`)
  auto-route through the daemon when it is running, falling back to direct
  execution when it is not. Write operations (`scan`, `remove`) auto-stop the
  daemon, execute, then auto-restart it. The daemon client uses only
  `std::os::unix::net` — zero new dependencies, works in any build.
- **`sift init` command**: scaffolds a `.sift.toml` project config with
  commented-out defaults. Auto-appends `.sift/` to `.gitignore` when inside a
  git repository. Supports `--force` to overwrite an existing config.
- **Shell completions enabled by default**: `sift completions bash|zsh|fish`
  now works out of the box without `--features completions`.
- **`GET /api/list` endpoint**: list indexed sources via the HTTP API.
- **`serve_unix()`**: new function in sift-server for serving Axum routers over
  Unix domain sockets with graceful shutdown support.
- **`byte_range` in search API**: `SearchResultItem` now includes the source
  byte range, enabling context display from API consumers.
- Per-language tree-sitter AST chunking (`ast-rust`, `ast-python`, etc.)
- SQLite FTS5 full-text search as default keyword engine
- Binary vector index format (`vectors.bin`) with JSON migration
- Feature-gated install profiles: minimal (6.6 MB), default (14 MB), standard, full
- Vision embedding support (Nomic Embed Vision v1.5)
- Audio metadata extraction (MP3, FLAC, WAV, OGG, AAC, M4A)
- HNSW approximate nearest-neighbor index
- HTTP API server with search and status endpoints
- Filesystem watcher for automatic re-indexing
- JSONL export with optional vector output
- Date filtering (`--after 7d`, `--after 2025-01-01`)
- Context display (`--context`) for showing surrounding source lines
- Named indexes (`--index`)
- JSON/CSV output formats (`--format json`)

### Changed
- `completions` feature added to default feature set in sift-cli.
- `tokio` dependency in sift-cli now includes `net` and `signal` features
  (required for daemon Unix socket and signal handling).

## [0.1.3] - 2026-04-04

### Added
- **MCP tool `sift_list_entities`**: browse all memory entities with optional type
  filtering and pagination. Returns observation counts per entity in a single
  query (LEFT JOIN + COUNT, no N+1).
- **MCP tool `sift_get_entity`**: retrieve all observations and relations for a
  named entity. Batch-resolves relation target names.
- **Entity-name fallback in recall**: when keyword search returns no results,
  scans entity names for case-insensitive matches against the query.
  Effective in keyword-only mode (no embeddings).
- **FTS5 prefix matching**: terms >= 4 characters now generate both exact and
  prefix queries (`"program" OR program*`), so "program" matches "programming".
  FTS5 operators (AND, OR, NOT, NEAR) are excluded from prefix expansion.

### Changed
- **`mcp` feature now includes `embeddings`**: the MCP server always loads the
  embedding model for semantic recall. Memory observations are re-embedded on
  first startup (when vector index count diverges from SQLite) and skipped on
  subsequent starts.
- **Feature flags propagated to sift-memory**: sift-mcp now forwards `fts5`,
  `hnsw`, `sqlite`, and `embeddings` features to sift-memory, fixing a
  mismatch where `#[cfg]` guards didn't match the actual compiled store types.
- **Vector store loaded from disk on startup**: `MemoryStore::open()` now calls
  `load_or_create()` / `load_or_migrate()` instead of `new()`, preserving the
  HNSW index across restarts.
- **Embedder wired to MemoryStore**: when the `embeddings` feature is active,
  the shared `Arc<dyn Embedder>` is passed to the memory store for observation
  embedding and hybrid recall.

### Fixed
- **FTS filename migration**: older builds created the FTS5 database at
  `memory_bm25.json` (due to the feature-flag mismatch). On first open with
  corrected flags, the file is renamed to `memory_fts.db` with safe fallback
  if the rename fails.
- **Stale zero-vectors replaced on embedder attach**: `with_embedder()` detects
  when the vector index is out of sync and rebuilds all observation embeddings.
  Logs elapsed time for observability.
- **Deadlock in recall fallback**: `recall()` now drops the SQLite lock before
  calling `recall_by_entity_names()` which re-acquires it.

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
