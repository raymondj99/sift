# sift

[![CI](https://github.com/raymondj99/sift/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/raymondj99/sift/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/raymondj99/sift/graph/badge.svg)](https://codecov.io/gh/raymondj99/sift)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)

**Point at anything. Search everything.**

`sift` is a fast CLI tool that indexes files and makes them searchable from a single binary. Written in Rust with a modular feature-gated architecture — build only what you need, from a 6.6 MB floor to a full-featured 17 MB binary.

```
$ sift scan ~/Documents
Indexed 14,293 files (42,819 chunks) — 0 skipped, 0 errors
  rs: 6,112  md: 2,658  pdf: 892  docx: 341  csv: 203

$ sift search --keyword-only "quarterly revenue"
  1. ~/Documents/Finance/Q3-Report.pdf        4.21
  2. ~/Documents/Finance/Board-Deck-Oct.pptx  3.87
  3. ~/Documents/Email/cfo-thread-0922.eml    2.94

$ sift search --type rs "error handling retry"
  1. src/services/payment/errors.rs        4.08
  2. src/services/payment/handler.rs       3.21
```

## Features

- **Zero config** — `sift scan .` works immediately. No YAML, no API keys, no setup.
- **Single binary** — Download or `cargo install`, run it.
- **Incremental indexing** — BLAKE3 content hashing. Only re-indexes what changed.
- **Local-first** — Everything runs on your machine. No data leaves your network.
- **Parallel pipeline** — Rayon-powered parallel parsing and chunking. Control with `--jobs`.
- **Hybrid search** — BM25 keyword search (FTS5) + vector similarity (cosine), fused with Reciprocal Rank Fusion.
- **Modular builds** — Feature-gate everything: AST, embeddings, parsers, server. Pay only for what you use.
- **30+ file formats** — Text, code, PDF, Office, HTML, CSV, JSON, email, images, audio, archives.
- **Context display** — `--context` shows surrounding source lines, `grep -C` style.
- **Date filtering** — `--after 7d` or `--after 2025-01-01` to filter by modification date.
- **JSON output** — `--json` for piping to `jq`, scripts, or other tools.
- **Export** — `sift export` dumps your index as JSONL for external tools and pipelines.

## Install

Pre-compiled binaries for macOS, Linux, and Windows are available on the
[releases page](https://github.com/raymondj99/sift/releases).

### Homebrew (macOS / Linux)

```bash
brew install raymondj99/tap/sift
```

### Cargo (any platform)

```bash
cargo install --path crates/sift-cli
```

### Build from source

```bash
git clone https://github.com/raymondj99/sift.git
cd sift
cargo build --release
# Binary at ./target/release/sift
```

### Build profiles

```bash
# Default — keyword search, common parsers, progress bars (~14 MB)
cargo install --path crates/sift-cli

# Minimal — keyword search only (~6.6 MB)
cargo install --path crates/sift-cli --no-default-features --features fts5,sqlite

# Standard — + AST chunking, embeddings, all parsers
cargo install --path crates/sift-cli --features standard

# Full — + HTTP server, HNSW, vision, audio, completions (~17 MB)
cargo install --path crates/sift-cli --features full
```

### Prerequisites

- **From source:** Rust 1.75+ (install via [rustup](https://rustup.rs/))
- **Embeddings feature:** ONNX Runtime shared library (`libonnxruntime.so` / `libonnxruntime.dylib`)

## Quick start

```bash
# Index current directory
sift scan .

# Search with BM25 keyword ranking
sift search --keyword-only "database connection pool"

# Search only Rust files
sift search --type rs "error handling"

# Show surrounding source context
sift search --keyword-only --context "config parsing"

# Only files modified in the last week
sift search --keyword-only --after 7d "TODO"

# JSON output for scripting
sift search --keyword-only --json "migration" | jq '.[0].uri'

# Show index stats
sift status

# List indexed files
sift list

# Export index as JSONL
sift export

# Remove a source
sift remove ./old-data/
```

### With embeddings (semantic search)

If built with the `embeddings` feature and ONNX Runtime is available:

```bash
# Download embedding model
sift models download nomic-embed-text-v2

# Scan with embeddings
sift scan .

# Hybrid search (vector + keyword, default)
sift search "error handling in payment service"

# Pure vector search
sift search --vector-only "retry logic patterns"
```

## Commands

| Command | Description |
|---------|-------------|
| `sift scan <PATH>...` | Scan and index data sources |
| `sift search <QUERY>` | Search across indexed data |
| `sift status` | Show index statistics |
| `sift list` | List indexed sources |
| `sift remove <PATH>` | Remove a source from the index |
| `sift config [KEY] [VALUE]` | View or set configuration |
| `sift export` | Export index data as JSONL |
| `sift models [list\|download]` | Manage embedding models |
| `sift watch [PATH]` | Watch for changes and re-index (requires `serve` feature) |
| `sift serve` | Start HTTP API server (requires `serve` feature) |

### Scan options

| Flag | Description |
|------|-------------|
| `--jobs <N>` | Parallel workers (0 = auto) |
| `--model <NAME>` | Override embedding model |
| `--max-depth <N>` | Maximum directory depth |
| `--max-file-size <BYTES>` | Skip files larger than this |
| `--include <GLOB>` | Only include files matching glob |
| `--exclude <GLOB>` | Exclude files matching glob |
| `--type <EXT>` | Only index specific file types |
| `--dry-run` | Preview without indexing |

### Search options

| Flag | Description |
|------|-------------|
| `-c, --context` | Show surrounding source lines |
| `--after <DATE>` | Filter by modification date (`2025-01-01`, `7d`, `2w`, `3m`) |
| `-n, --max-results <N>` | Maximum results (default: 10) |
| `-t, --type <EXT>` | Filter by file type |
| `--path <GLOB>` | Filter by path pattern |
| `--threshold <F>` | Minimum similarity (0.0-1.0) |
| `--vector-only` | Pure vector search |
| `--keyword-only` | Pure BM25 keyword search |
| `--json` | Output as JSON |

### Export options

| Flag | Description |
|------|-------------|
| `--vectors` | Include embedding vectors in output |
| `-o, --output <FILE>` | Write to file instead of stdout |
| `-t, --type <EXT>` | Filter by file type |

## Search modes

- **Hybrid** (default) — Combines vector similarity and BM25 keyword search using Reciprocal Rank Fusion.
- **Keyword-only** (`--keyword-only`) — BM25 full-text search via SQLite FTS5. No embedding model needed.
- **Vector-only** (`--vector-only`) — Pure cosine similarity. Requires a downloaded embedding model.

Without a model, `sift search` falls back to keyword-only BM25, which works well for exact term matching.

## Supported formats

| Category | Formats |
|----------|---------|
| **Text** | `.txt`, `.md`, `.rst`, `.org` |
| **Code** | `.rs`, `.py`, `.js`, `.ts`, `.go`, `.c`, `.cpp`, `.java`, `.rb`, `.sh`, `.swift`, `.kt`, `.zig`, `.lua`, and more |
| **Data** | `.json`, `.jsonl`, `.csv`, `.toml`, `.yaml` |
| **Web** | `.html`, `.htm`, `.xml` |
| **Documents** | `.pdf`, `.docx`, `.pptx`, `.xlsx` (requires feature flags) |
| **Email** | `.eml`, `.mbox` (requires `email` feature) |
| **Archives** | `.zip`, `.tar`, `.gz` (requires `archive` feature) |
| **Images** | `.png`, `.jpg`, `.gif`, `.webp` (metadata; vision embedding with `vision` feature) |
| **Audio** | `.mp3`, `.wav`, `.flac`, `.ogg`, `.aac`, `.m4a` (metadata, requires `audio` feature) |

## Feature flags

sift uses Cargo feature flags to control binary size. Only compile what you need.

### Install profiles

| Profile | Command | Size | What you get |
|---------|---------|------|-------------|
| **Minimal** | `--no-default-features --features fts5,sqlite` | ~6.6 MB | Text/code search only |
| **Default** | *(none)* | ~14 MB | + CSV, Office, archives, progress bars |
| **Standard** | `--features standard` | ~14 MB | + AST chunking, PDF, email, embeddings |
| **Full** | `--features full` | ~17 MB | + HTTP server, HNSW, vision, audio, completions |

### Individual features

| Feature | Description |
|---------|-------------|
| `fts5` | SQLite FTS5 keyword search (BM25 ranking) |
| `sqlite` | SQLite metadata storage |
| `data` | CSV parser |
| `office` | DOCX/PPTX parser |
| `epub` | EPUB parser |
| `archive` | ZIP/TAR/GZ extraction |
| `pdf` | PDF text extraction |
| `email` | EML/MBOX parsing |
| `spreadsheets` | XLSX parser |
| `audio` | Audio metadata extraction (MP3, FLAC, etc.) |
| `embeddings` | ONNX Runtime vector embeddings |
| `vision` | Cross-modal image embedding (Nomic Embed Vision) |
| `ast` | Tree-sitter AST-aware code chunking (all languages) |
| `ast-rust`, `ast-python`, ... | Per-language AST chunking |
| `fancy` | Progress bars and colored output |
| `serve` | HTTP API server + filesystem watcher |
| `hnsw` | HNSW approximate nearest-neighbor index |
| `fulltext` | Tantivy full-text search (alternative to FTS5) |
| `completions` | Shell completion generation |

## Architecture

```
Source -> Discovery -> Parsing -> Chunking -> Embedding -> Storage -> Search
          (walkdir)   (per-type)  (semantic)   (ONNX)     (SQLite)
```

Eight crates in a Cargo workspace:

| Crate | Purpose |
|-------|---------|
| `sift-core` | Config, error types, pipeline data types |
| `sift-sources` | Source connectors (filesystem) |
| `sift-parsers` | File format parsers with MIME-based dispatch |
| `sift-chunker` | Fixed-size and AST-aware semantic chunking |
| `sift-embed` | ONNX Runtime embedding with model management |
| `sift-store` | SQLite metadata, FTS5 keyword search, vector store, hybrid search |
| `sift-server` | HTTP API (Axum) and filesystem watcher |
| `sift-cli` | CLI entry point |

## Configuration

Config lives at `~/.sift/config.toml` (auto-created with defaults):

```toml
[default]
model = "nomic-embed-text-v2"
chunk_size = 512
chunk_overlap = 64
max_file_size = 104857600  # 100 MB
jobs = 0                   # 0 = auto-detect CPU count

[search]
max_results = 10
hybrid_alpha = 0.7  # 0.0 = pure BM25, 1.0 = pure vector
```

```bash
sift config default.chunk_size 256
sift config search.hybrid_alpha 0.5
```

## Environment variables

| Variable | Description |
|----------|-------------|
| `SIFT_INDEX` | Named index to use (default: `default`) |
| `SIFT_FORMAT` | Output format: `human`, `json`, `csv` |
| `RUST_LOG` | Log level: `error`, `warn`, `info`, `debug`, `trace` |

## Building from source

```bash
git clone https://github.com/raymondj99/sift.git
cd sift

# Development build
cargo build

# Release build (optimized, ~14 MB with defaults)
cargo build --release

# Run tests
cargo test

# Run tests with all features
cargo test --all-features

# Lint
cargo clippy
cargo clippy --all-features

# Format
cargo fmt --check
```

### Release profile

The release build uses aggressive optimizations:

```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
strip = true
panic = "abort"
```

## License

MIT OR Apache-2.0. See [LICENSE](LICENSE) for details.
