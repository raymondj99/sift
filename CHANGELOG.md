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
