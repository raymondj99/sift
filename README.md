# sift

[![CI](https://github.com/raymondj99/sift/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/raymondj99/sift/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/raymondj99/sift/graph/badge.svg)](https://codecov.io/gh/raymondj99/sift)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)

**Point at anything. Search everything.**

`sift` is a fast CLI tool that indexes files and makes them searchable from a single binary. Written in Rust with a modular feature-gated architecture — build only what you need, from a 2.2 MB floor to a full-featured 18 MB binary.

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

### Quick install (one-liner)

```bash
curl -fsSL https://raw.githubusercontent.com/raymondj99/sift/main/install.sh | sh
```

Detects your OS/arch, downloads the latest release, verifies the SHA-256
checksum, and installs to `/usr/local/bin` (or `~/.local/bin` when sudo
isn't available). All optional:

```bash
SIFT_VERSION=v0.1.6         # pin to a specific version
SIFT_INSTALL_DIR=/custom    # custom install directory
SIFT_NO_COMPLETIONS=1       # skip shell completion install
SIFT_NO_MODIFY_PATH=1       # silence the PATH-guidance note
```

### Homebrew (macOS / Linux)

```bash
brew install raymondj99/tap/sift-search
```

### Manual download (audit-friendly)

For users who prefer to inspect the tarball before running anything,
download from the [releases page](https://github.com/raymondj99/sift/releases).
Replace `VERSION` below with the release tag (e.g. `v0.1.6`):

```bash
# macOS (Apple Silicon)
curl -LO https://github.com/raymondj99/sift/releases/download/VERSION/sift-VERSION-aarch64-apple-darwin.tar.gz
tar xzf sift-VERSION-aarch64-apple-darwin.tar.gz
sudo mv sift-VERSION-aarch64-apple-darwin/sift /usr/local/bin/

# macOS (Intel)
curl -LO https://github.com/raymondj99/sift/releases/download/VERSION/sift-VERSION-x86_64-apple-darwin.tar.gz
tar xzf sift-VERSION-x86_64-apple-darwin.tar.gz
sudo mv sift-VERSION-x86_64-apple-darwin/sift /usr/local/bin/

# Linux (x86_64)
curl -LO https://github.com/raymondj99/sift/releases/download/VERSION/sift-VERSION-x86_64-unknown-linux-gnu.tar.gz
tar xzf sift-VERSION-x86_64-unknown-linux-gnu.tar.gz
sudo mv sift-VERSION-x86_64-unknown-linux-gnu/sift /usr/local/bin/
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
| `sift init` | Initialize a `.sift.toml` project config in the current directory |
| `sift mcp` | Start MCP server for AI agent integration (requires `mcp` feature) |
| `sift memory <CMD>` | Manage the Cortex automated memory system (`status`, `consolidate`, `generate-rules`, `init-hooks`) |
| `sift memory-tool <CMD>` | Anthropic `memory_20250818` adapter backed by sift memory (`exec`, `path`, `migrate`) |
| `sift daemon` | Manage the sift background daemon (requires `serve` feature) |
| `sift bench` | Run Cortex memory system benchmarks |
| `sift watch [PATH]` | Watch for changes and re-index (requires `serve` feature) |
| `sift serve` | Start HTTP API server (requires `serve` feature) |
| `sift completions <SHELL>` | Generate shell completions (requires `completions` feature) |

### Scan options

| Flag | Description |
|------|-------------|
| `-j, --jobs <N>` | Parallel workers (0 = auto) |
| `--model <NAME>` | Override embedding model |
| `--max-depth <N>` | Maximum directory depth |
| `--max-file-size <BYTES>` | Skip files larger than this |
| `--include <GLOB>` | Only include files matching glob |
| `--exclude <GLOB>` | Exclude files matching glob |
| `-t, --type <EXT>` | Only index specific file types |
| `--dry-run` | Preview without indexing |
| `--prune` | Remove index entries for deleted files |

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
| `-o, --open` | Open top result in default application |
| `--json` | Output as JSON |

### Export options

| Flag | Description |
|------|-------------|
| `--vectors` | Include embedding vectors in output |
| `-o, --output <FILE>` | Write to file instead of stdout |
| `-t, --type <EXT>` | Filter by file type |

## AI agent integration

Sift includes an MCP (Model Context Protocol) server that exposes search, indexing, and memory tools to AI agents. Start it with `sift mcp` — it communicates via JSON-RPC 2.0 over stdio.

### Available MCP tools

| Tool | Type | Description |
|------|------|-------------|
| `sift_search` | Read | Hybrid semantic + keyword search across indexed files |
| `sift_status` | Read | Index statistics (file counts, types, storage size) |
| `sift_search_skills` | Read | Discover agent skills (SKILL.md files) |
| `sift_list_sources` | Read | Browse indexed files with path filtering |
| `sift_index_text` | Write | Store arbitrary text with custom `memory://` URIs |
| `sift_delete` | Write | Remove content from the index by URI |
| `sift_remember` | Write | Persist entities, observations, and relations to memory |
| `sift_recall` | Read | Semantic search over stored memories with conflict detection |
| `sift_list_entities` | Read | Browse all entities in memory with optional type filter |
| `sift_get_entity` | Read | Get all facts and relationships for a named entity |
| `sift_forget` | Write | Soft-delete a specific observation by ID |
| `sift_forget_entity` | Write | Hard-delete an entity and all its observations |
| `sift_prune` | Write | Remove ghost entities with zero observations |
| `sift_consolidate` | Write | Run the 5-phase memory consolidation pipeline |
| `sift_memory_status` | Read | Memory store statistics (entities, observations, tiers) |

### Setup

Before configuring any agent, index the directories you want searchable:

```bash
# Index your project
sift scan ~/my-project

# (Optional) Download embedding model for semantic search
sift models download nomic-embed-text-v2

# Verify
sift status
```

### Claude Code

Add sift to your project's `.mcp.json` (or `~/.claude.json` for global):

```json
{
  "mcpServers": {
    "sift": {
      "command": "sift",
      "args": ["mcp"]
    }
  }
}
```

Or add it interactively:

```bash
claude mcp add sift -- sift mcp
```

Once configured, Claude Code can use all sift tools directly. Try asking:
- "Use sift to search for error handling patterns"
- "Remember that I prefer integration tests over mocks"
- "What do you recall about my testing preferences?"

### Cursor

Add to your Cursor MCP configuration at `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "sift": {
      "command": "sift",
      "args": ["mcp"]
    }
  }
}
```

Restart Cursor after saving. Sift tools will appear in the agent's available tool list.

### VS Code (Copilot)

Add to your VS Code settings (`settings.json`) or workspace `.vscode/mcp.json`:

```json
{
  "mcp": {
    "servers": {
      "sift": {
        "command": "sift",
        "args": ["mcp"]
      }
    }
  }
}
```

### OpenAI Codex CLI

Add sift to your `~/.codex/config.toml`:

```toml
[mcp_servers.sift]
command = "sift"
args = ["mcp"]
```

To install the full Codex integration automatically, including hooks:

```bash
sift integrate codex
```

This writes the MCP server entry in `~/.codex/config.toml`, enables
`features.codex_hooks = true`, and installs repo-local hooks at
`<project-root>/.codex/hooks.json`.

Codex hook parity is currently best-effort:
- `Stop` runs `sift memory ingest --event stop` followed by
  `sift memory consolidate --quiet`, which regenerates `AGENTS.md`
- `PostToolUse` captures `Bash` commands only
- Codex does not yet emit hooks for `apply_patch`, `Write`, MCP tools, or
  other non-Bash edits, so it cannot fully match Claude Code's edit capture
- Codex has no `PostCompact` equivalent, so `Stop` is the summary signal

### Memory persistence (Cortex)

Sift includes **Cortex**, an automated memory system that gives AI agents persistent memory across sessions. It uses a temporal knowledge graph with three tiers:

- **Episodic** — raw session events captured via Claude Code hooks (<100ms, zero LLM cost)
- **Semantic** — consolidated facts promoted from episodic tier by access frequency or age
- **Procedural** — learned workflow patterns (never auto-deleted)

```
Agent → sift_remember("Raymond prefers Rust") → SQLite + search index
Agent → sift_recall("language preferences")   → ranked results with scores
Agent → sift_forget(observation_id)            → soft-delete, audit trail preserved
```

Key features:
- **LLM-powered extraction** — session summaries are decomposed into individually categorized observations (decisions, corrections, workflows, preferences) on named entities via an LLM. Supports Anthropic, OpenAI-compatible, and Ollama providers.
- **Conflict detection** — `sift_remember` flags when new facts contradict existing ones
- **5-phase consolidation** — dedup, promotion, skill extraction, decay, and pruning
- **Retrieval-dependent strengthening** — frequently recalled memories gain relevance
- **Rules generation** — `sift memory generate-rules` always updates `AGENTS.md`, and also writes `.claude/rules/` when `.claude/` exists. `AGENTS.md` is regenerated by consolidation and rule generation, not by MCP calls alone.

To enable LLM extraction (recommended), add to `~/.sift/config.toml`:

```toml
[memory]
llm_extraction = true
llm_extraction_provider = "anthropic"           # or "openai", "ollama"
llm_extraction_model = "claude-haiku-4-5-20251001"
```

Set `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY` / `OLLAMA_HOST`) in your environment. Cost: ~$0.003 per session with Haiku.

Memory is stored at `~/.sift/indexes/{index}/memory/` and persists across sessions.

### First-class AI memory surfaces

Sift now has one memory backend with two user-facing surfaces:

- **Universal MCP memory** — every client supported by `sift integrate` reaches the same sift memory through MCP tools like `sift_remember`, `sift_recall`, and `sift_get_entity`
- **Anthropic native memory-tool** — `sift memory-tool` exposes Anthropic's `memory_20250818` file API as a projection over sift memory, not a separate directory store

Supported memory matrix in this wave:

| Surface | Clients |
|---------|---------|
| **MCP memory support** | Claude Code, Cursor, Windsurf, Codex, Gemini CLI, opencode, Zed, VS Code, LM Studio |
| **Native file-shaped memory support** | Anthropic memory-tool only |
| **Generated rules** | `AGENTS.md` always; `.claude/rules/` only when `.claude/` exists |

### Anthropic memory-tool adapter

`sift memory-tool` implements a virtual writable namespace at
`/memories/entities/<slug>--<entity_id>.md` for existing entities. New pages are
created via `/memories/entities/<slug>.md`, then re-viewed at their canonical
ID-qualified path. Pages render directly from sift memory and
preserve stable inline IDs for observations and relations:

```md
---
entity: Raymond
entity_id: ent_...
entity_type: person
page_revision: rev_...
---

# Raymond

## Observations
- [obs_...] prefers Rust over Python

## Relations
- [rel_...] maintains -> sift
```

Edits are validated against a freshly rendered page and mapped back into sift's
knowledge graph. In v1, `str_replace` is the supported mutating edit path for
existing pages; `insert` is intentionally rejected because line-number inserts
cannot be made safe with the current Anthropic API shape. Unsupported freeform
edits, heading changes, or anchorless rewrites are rejected instead of silently
drifting the underlying memory.

`str_replace` is line-targeted CAS: a concurrent edit to the anchored line
you targeted rejects with `page changed, re-view and retry`; concurrent edits
elsewhere on the page are accepted as long as the targeted bytes still match.

Edit contract:

| Edit | Backend action |
|------|----------------|
| Remove observation bullet | Invalidate the anchored observation |
| Rewrite observation bullet | Supersede the anchored observation (lineage preserved via `logical_id`) |
| Remove relation bullet | Invalidate the anchored relation |
| Rewrite relation bullet | Invalidate old relation, create a new outgoing relation |
| Change `entity_type` | Update the existing entity in place |
| Delete entity page | `forget_entity` on the backing entity |
| Add observation or relation bullet via `str_replace` | Rejected in v1 |
| Change `entity`, `entity_id`, title, or section headings | Rejected |
| Freeform prose outside the schema | Rejected |
| `rename` command | Rejected in v1 |

Known limitations: observation content containing literal newlines does not
round-trip cleanly; mixed remove+add edits are rejected unless they map to a
single rewrite; `create` accepts a bare-slug path but subsequent `view`,
`str_replace`, and `delete` use the `<slug>--<entity_id>` form.

If you used the earlier file-backed adapter, import old notes once with:

```bash
sift memory-tool migrate
```

Legacy notes are imported onto the synthetic entity `legacy-memory-tool notes`
and become discoverable via `sift_recall`.

### Daemon mode

The daemon keeps the search engine and embedding model loaded in memory, eliminating cold-start latency. It can also automatically re-index files when they change on disk.

```bash
sift daemon start    # start background daemon
sift daemon status   # check if running
sift daemon stop     # stop daemon
```

To enable automatic re-indexing, add to `~/.sift/config.toml`:

```toml
[watch]
enabled = true
debounce_ms = 1000
```

The watcher automatically monitors all directories you've previously indexed. When files change, it debounces events and re-indexes only the modified files using the same pipeline as `sift scan`.

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
| **Minimal** | `--no-default-features --features fts5,sqlite` | ~2 MB | Text/code search only |
| **Default** | *(none)* | ~4 MB | + CSV, Office, archives, progress bars |
| **Standard** | `--features standard` | ~13 MB | + AST chunking, PDF, email, embeddings |
| **Full** | `--features full` | ~18 MB | + HTTP server, HNSW, vision, audio, completions |

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
| `serve` | HTTP API server + filesystem watcher + daemon |
| `mcp` | MCP server for AI agent integration |
| `hnsw` | HNSW approximate nearest-neighbor index |
| `fulltext` | Tantivy full-text search (alternative to FTS5) |
| `completions` | Shell completion generation |

## Architecture

```
Source -> Discovery -> Parsing -> Chunking -> Embedding -> Storage -> Search
          (walkdir)   (per-type)  (semantic)   (ONNX)     (SQLite)
```

Ten crates in a Cargo workspace:

| Crate | Purpose |
|-------|---------|
| `sift-core` | Config, error types, retry, pipeline data types |
| `sift-sources` | Source connectors (filesystem) |
| `sift-parsers` | File format parsers with MIME-based dispatch |
| `sift-chunker` | Fixed-size, semantic, recursive, and AST-aware chunking |
| `sift-embed` | ONNX Runtime embedding with model management and cache |
| `sift-store` | SQLite metadata, FTS5/Tantivy keyword search, flat/HNSW vector store, LRU cache, hybrid RRF search |
| `sift-memory` | Temporal knowledge graph for AI agent memory persistence |
| `sift-server` | HTTP API (Axum) with rate limiting, and filesystem watcher |
| `sift-mcp` | Model Context Protocol server (JSON-RPC 2.0 over stdio) |
| `sift-cli` | CLI entry point and pipeline orchestration |

## Configuration

Config lives at `~/.sift/config.toml` (auto-created with defaults).
Per-project overrides can be placed in `.sift.toml` at the project root.

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

[ignore]
patterns = ["*.lock", "node_modules"]

[server]
host = "127.0.0.1"
port = 7820

[watch]
enabled = true       # auto-re-index on file changes (daemon mode)
debounce_ms = 1000   # batch changes within this window (ms)

[memory]
enabled = true                   # Cortex automated memory
consolidation_interval = 1800   # seconds between runs
decay_rate = 0.01               # Ebbinghaus forgetting curve
llm_extraction = true            # LLM-powered observation extraction
llm_extraction_provider = "anthropic"  # or "openai", "ollama"
llm_extraction_model = "claude-haiku-4-5-20251001"
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
| `SIFT_MODEL` | Override embedding model |
| `SIFT_JOBS` | Parallel worker count (0 = auto) |
| `ANTHROPIC_API_KEY` | API key for LLM extraction (Anthropic provider) |
| `OPENAI_API_KEY` | API key for LLM extraction (OpenAI provider) |
| `OLLAMA_HOST` | Ollama endpoint (default: `http://localhost:11434`) |
| `RUST_LOG` | Log level: `error`, `warn`, `info`, `debug`, `trace` |
| `NO_COLOR` | Disable colored output (any value) |

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
