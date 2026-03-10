# sift Performance Optimization Index

Optimizations derived from analysis of [ripgrep](https://github.com/BurntSushi/ripgrep), [fd](https://github.com/sharkdp/fd), and other high-performance Rust CLI tools.

## Optimization Specs

| # | File | Target Crate | Impact | Effort | Status |
|---|------|-------------|--------|--------|--------|
| 1 | [01-single-pass-file-discovery.md](01-single-pass-file-discovery.md) | sift-sources | High | Low | Done |
| 2 | [02-parallel-directory-walking.md](02-parallel-directory-walking.md) | sift-sources | High | Medium | Pending |
| 3 | [03-simd-cosine-similarity.md](03-simd-cosine-similarity.md) | sift-store | High | Low | Pending |
| 4 | [04-batch-metadata-queries.md](04-batch-metadata-queries.md) | sift-store, sift-cli | Medium | Low | Done |
| 5 | [05-sqlite-write-batching.md](05-sqlite-write-batching.md) | sift-store | Medium | Low | Done |
| 6 | [06-global-allocator.md](06-global-allocator.md) | sift-cli | Medium | Low | Done |
| 7 | [07-benchsuite.md](07-benchsuite.md) | workspace | Critical | Medium | Done |

## Architecture Context

```
Source -> Discovery -> Parsing -> Chunking -> Embedding -> Storage -> Search
  (sync)   (rayon)    (rayon)    (serial)    (serial)     (sync)
```

### Current Bottlenecks (ordered by severity)

1. **File discovery reads each file twice** (MIME detect + BLAKE3 hash) - `sift-sources/src/filesystem.rs`
2. **Directory walking is single-threaded** (uses `ignore` sequential walker) - `sift-sources/src/filesystem.rs`
3. **Cosine similarity is scalar** (no SIMD) - `sift-store/src/flat.rs:422-443`
4. **Metadata check is N individual SQLite queries** - `sift-cli/src/pipeline.rs:237-247`
5. **Store stage has no transaction batching** - `sift-cli/src/pipeline.rs:448-489`

### CI Commands (from `.github/workflows/ci.yml`)

```bash
# Lint
cargo fmt --all -- --check
cargo clippy --workspace --all-targets
cargo clippy --workspace --all-targets --all-features

# Test
cargo test --workspace
cargo test --workspace --all-features
cargo test --workspace --all-features -- --include-ignored
```
