# Contributing to sift

Thanks for your interest in contributing to sift.

## Getting started

```bash
git clone https://github.com/raymondj99/sift.git
cd sift
cargo build
cargo test
```

Requires Rust 1.75+.

### Feature profiles

The default build includes core search functionality. For the full feature set:

```bash
# Default: FTS5 + SQLite + HNSW + document parsers + shell completions
cargo build

# With daemon, HTTP server, and MCP:
cargo build --features serve,mcp,embeddings

# Everything (release builds use this):
cargo build --features full
```

### Project config

Run `sift init` in the repo root to create a `.sift.toml` with project-specific
overrides (chunk size, ignore patterns, etc.). This file is checked into version
control so all contributors share the same search config.

## Architecture

```
crates/
  sift-cli/        CLI binary, daemon mode, pipeline orchestration
  sift-core/       Shared types, config, error handling
  sift-store/      SQLite metadata, FTS5/BM25 keyword search, HNSW vectors
  sift-embed/      ONNX Runtime embedding models
  sift-server/     Axum HTTP API (TCP + Unix socket)
  sift-mcp/        MCP protocol server for AI agent integration
  sift-memory/     Entity-based knowledge graph
  sift-parsers/    30+ file format parsers
  sift-chunker/    Document chunking with AST-aware code splitting
  sift-sources/    Filesystem source discovery
```

## Development workflow

1. Fork the repo and create a feature branch
2. Make your changes
3. Run `cargo test` and `cargo clippy`
4. Submit a pull request

Pre-commit hooks run `cargo fmt`, `cargo clippy`, and `cargo deny` automatically.

## Testing

```bash
# Run all tests
cargo test

# Test a specific crate
cargo test -p sift-cli

# Test with optional features (daemon, server)
cargo test -p sift-cli --features serve

# Test the daemon module specifically
cargo test -p sift-cli --features serve daemon
```

## Code style

- Run `cargo fmt` before committing
- No warnings from `cargo clippy`
- Write tests for new functionality
- Keep functions short and focused
- Prefer explicit error handling over `.unwrap()` in library code

## Adding a new CLI command

1. Create `crates/sift-cli/src/commands/<name>.rs` with a `pub fn run(...)` entry point
2. Add the module to `commands/mod.rs` (feature-gate if needed)
3. Add a variant to the `Commands` enum in `main.rs`
4. Add the dispatch arm in `run_command()`
5. If the command is a read operation, add daemon routing via `daemon_client::get()`
6. If it's a write operation, add daemon auto-stop/restart around the operation
7. Write tests

## Adding a new file format parser

1. Create a new file in `crates/sift-parsers/src/` (e.g., `pdf.rs`)
2. Implement the `Parser` trait
3. Register it in `crates/sift-parsers/src/registry.rs`
4. Add MIME types and extensions
5. Write tests

## Adding a new API endpoint

1. Add the handler and route in `crates/sift-server/src/routes.rs`
2. Register the route in `create_router()`
3. Add corresponding daemon client support in `crates/sift-cli/src/daemon_client.rs`
4. Write tests using the `TestHarness` pattern in `routes.rs`

## Adding a new source connector

1. Create a new file in `crates/sift-sources/src/` (e.g., `s3.rs`)
2. Implement the `Source` trait
3. Export it in `crates/sift-sources/src/lib.rs`
4. Wire it up in the CLI scan command

## Reporting bugs

Open an issue with:
- What you expected
- What happened
- Steps to reproduce
- `sift --version` output
