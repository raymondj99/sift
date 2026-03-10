# sift Code Quality Improvement Index

Findings from a full codebase review against the standards of [ripgrep](https://github.com/BurntSushi/ripgrep), [fd](https://github.com/sharkdp/fd), and other production-grade Rust CLI tools.

## Improvement Specs

| # | File | Primary Crates | Category | Severity | Effort |
|---|------|---------------|----------|----------|--------|
| 1 | [01-error-handling.md](01-error-handling.md) | all | Robustness | High | Medium |
| 2 | [02-type-duplication.md](02-type-duplication.md) | sift-store, sift-core | Organization | Medium | Low |
| 3 | [03-hot-path-allocations.md](03-hot-path-allocations.md) | sift-sources, sift-cli | Performance | High | Low |
| 4 | [04-pipeline-bottlenecks.md](04-pipeline-bottlenecks.md) | sift-cli, sift-store | Performance | High | Medium |
| 5 | [05-atomicity-and-safety.md](05-atomicity-and-safety.md) | sift-store, sift-core | Robustness | High | Medium |
| 6 | [06-api-simplification.md](06-api-simplification.md) | all | Simplicity | Medium | Low |
| 7 | [07-testing-gaps.md](07-testing-gaps.md) | all | Robustness | Medium | Medium |

## Summary of Findings

The codebase is well-structured for a workspace of this scope. The crate boundaries are mostly sensible (core types, sources, parsers, chunkers, embeddings, storage, CLI). Feature gating is thorough. The pipeline design with bounded channels and rayon parallelism is good. Tests are present and cover the happy path adequately.

The gaps versus production-grade Rust CLI tools fall into three categories:

1. **Robustness**: Error handling is the biggest gap. ~40 instances of `.map_err(|e| SiftError::Variant(format!("...: {}", e)))` lose type information and create maintenance burden. Silent error swallowing (`let _ =`, `.filter_map(|r| r.ok())`) hides failures. No atomic writes, no file locking, no signal handling.

2. **Performance**: The pipeline has structural bottlenecks — the embed stage is single-threaded, `ParserRegistry` is allocated per-file inside rayon workers, full files are read into memory for MIME detection (only needs first 8KB), `canonicalize()` syscall on every file, BM25 saves to disk on every insert, and brute-force vector search does a full sort instead of partial sort.

3. **Simplicity**: 5+ types duplicate the same 7 fields (uri, text, chunk_index, content_type, file_type, title, byte_range). Legacy wrapper functions that just delegate. Hand-rolled JSON serialization and date parsing where libraries exist. The `Embedder` trait is defined in `sift-core` but conceptually belongs in `sift-embed`.
