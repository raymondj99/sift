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

## Development workflow

1. Fork the repo and create a feature branch
2. Make your changes
3. Run `cargo test` and `cargo clippy`
4. Submit a pull request

## Code style

- Run `cargo fmt` before committing
- No warnings from `cargo clippy`
- Write tests for new functionality
- Keep functions short and focused
- Prefer explicit error handling over `.unwrap()` in library code

## Adding a new file format parser

1. Create a new file in `crates/sift-parsers/src/` (e.g., `pdf.rs`)
2. Implement the `Parser` trait
3. Register it in `crates/sift-parsers/src/registry.rs`
4. Add MIME types and extensions
5. Write tests

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
