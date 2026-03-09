# sift Benchmark Suite

Benchmarks for measuring sift performance, inspired by
[ripgrep's benchsuite](https://github.com/BurntSushi/ripgrep/tree/master/benchsuite).

## Structure

```
benchsuite/
  run.sh              End-to-end CLI benchmarks (hyperfine)
  create_corpus.sh    Generate synthetic test corpus (~3500 files)
  results/            Benchmark output (gitignored)

crates/sift-store/
  benches/
    vector_search.rs  Criterion microbenchmarks (cosine, search, insert)
```

## Microbenchmarks (Criterion)

```bash
cargo bench -p sift-store
```

Compare before/after a change:

```bash
cargo bench -p sift-store -- --save-baseline before
# ... make changes ...
cargo bench -p sift-store -- --baseline before
```

## End-to-End Benchmarks (Hyperfine)

```bash
cargo build --release
bash benchsuite/run.sh
```

Override defaults with environment variables:

```bash
SIFT_BENCH_CORPUS=/path/to/corpus SIFT_BIN=./target/release/sift bash benchsuite/run.sh
```

## What's Measured

| Benchmark | What it tests |
|---|---|
| Fresh scan | Full pipeline: discover, parse, chunk, embed, store |
| Incremental scan | Skip-unchanged logic via content hash comparison |
| Keyword search | FTS5 full-text search latency |
| Dry run | File discovery and filtering only |
| Vector search | Cosine similarity at various index sizes (criterion) |
| Vector insert | Batch insert throughput (criterion) |
