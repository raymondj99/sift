# TODO — Prioritized Improvements

Tasks ordered by severity × leverage. References link to detailed specs in `docs/`.

## High Severity / Low Effort (Implement First)

- [x] **ParserRegistry allocated per-file** — `pipeline.rs:336` allocates 10+ boxed parsers per file in rayon loop. Create once, share via `&`. Saves ~100K heap allocs per 10K-file scan.
  → `docs/quality/03-hot-path-allocations.md` §1

- [x] **Drop `canonicalize()` syscall** — `filesystem.rs:230` calls `canonicalize()` on every discovered file. Use path directly. Saves 1 syscall/file.
  → `docs/quality/03-hot-path-allocations.md` §2

- [x] **BM25 saves entire index on every insert** — `bm25.rs:310` serializes + writes the full index after each file. Defer to `flush()` at end of pipeline.
  → `docs/quality/04-pipeline-bottlenecks.md` §2

- [x] **Full sort for top-k search** — `flat.rs:372` does O(n log n) sort for top_k=10. Use `select_nth_unstable_by` for O(n + k log k).
  → `docs/quality/04-pipeline-bottlenecks.md` §3

- [x] **`format!` allocations in RRF fusion** — `hybrid.rs:85` allocates a `String` per result. Use `(&str, u32)` tuple key.
  → `docs/quality/03-hot-path-allocations.md` §4

- [x] **MIME type string allocations** — `filesystem.rs:88` returns `String` for static MIME strings. Return `&'static str`.
  → `docs/quality/03-hot-path-allocations.md` §5

- [x] **`read_context_lines` reads entire file** — `output.rs:109` stores all lines to display 5. Use sliding window.
  → `docs/quality/03-hot-path-allocations.md` §6

- [x] **Delete legacy wrapper functions** — `output.rs:41-43,160-162` are dead wrappers. Update call sites, remove wrappers.
  → `docs/quality/06-api-simplification.md` §1

## High Severity / Medium Effort (Implement Next)

- [x] **Atomic file writes** — All `std::fs::write` calls replaced with `sift_core::atomic_write` (write-then-rename via `tempfile`).
  → `docs/quality/05-atomicity-and-safety.md` §1

- [x] **Parallelize embed stage** — `pipeline.rs` Stage 3 uses `par_bridge()` for concurrent embedding.
  → `docs/quality/04-pipeline-bottlenecks.md` §1

- [x] **Transaction guard pattern** — `TransactionGuard` RAII type auto-rolls back on drop. Used in pipeline Stage 4.
  → `docs/quality/05-atomicity-and-safety.md` §5

- [x] **Signal handling for graceful shutdown** — `ctrlc` handler sets `AtomicBool` flag; all pipeline stages check it and drain cleanly.
  → `docs/quality/05-atomicity-and-safety.md` §3

## Medium Severity

- [x] **BM25 serde serialization** — Hand-rolled JSON replaced with `#[derive(Serialize, Deserialize)]`.
  → `docs/quality/06-api-simplification.md` §2

- [ ] **Parallel directory walking** — Use `ignore::WalkBuilder::build_parallel()`.
  → `docs/performance/02-parallel-directory-walking.md`

- [x] **File locking for concurrent access** — Advisory exclusive lock via `fs4` in `run_scan_pipeline`.
  → `docs/quality/05-atomicity-and-safety.md` §2

- [ ] **Property-based tests for chunkers** — Most likely source of subtle bugs.
  → `docs/quality/07-testing-gaps.md` §1

- [ ] **Error-path integration tests** — Corrupt index, read-only dirs, concurrent scans.
  → `docs/quality/07-testing-gaps.md` §2

## Low Severity / Nice-to-Have

- [ ] **Simplify `Source` trait** — Callback → `Vec` return.
  → `docs/quality/06-api-simplification.md` §4

- [ ] **SIMD cosine similarity** — Use `std::simd` or `pulp` for vectorized dot product.
  → `docs/performance/03-simd-cosine-similarity.md`

- [ ] **`dirs_home` falls back to `.`** — Should error, not silently misconfigure.
  → `docs/quality/05-atomicity-and-safety.md` §4

- [ ] **Collapse feature flags** — 22 flags → coarser groups.
  → `docs/quality/06-api-simplification.md` §6

- [ ] **CI coverage reporting** — `cargo-tarpaulin` or `cargo-llvm-cov`.
  → `docs/quality/07-testing-gaps.md` §5

## Already Implemented (from `docs/performance/`)

- [x] Single-pass file discovery (hash + MIME in one read) — `docs/performance/01`
- [x] Batch metadata queries (`load_all_hashes`) — `docs/performance/04`
- [x] SQLite write batching (begin/commit transaction) — `docs/performance/05`
- [x] jemalloc global allocator — `docs/performance/06`
- [x] Benchmark suite (corpus generation + hyperfine runner) — `docs/performance/07`
