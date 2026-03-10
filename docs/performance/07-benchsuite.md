# 07: Benchmark Suite

## Problem

sift has no benchmark infrastructure. Without benchmarks, we can't measure the impact of performance optimizations or catch regressions.

ripgrep has a comprehensive [benchsuite](https://github.com/BurntSushi/ripgrep/tree/master/benchsuite) that tests against real-world data. We need something similar.

## Solution

Create a benchmark suite using `criterion` for microbenchmarks and `hyperfine` for end-to-end CLI benchmarks.

### Implementation

#### A) Criterion Microbenchmarks

**File: `Cargo.toml` (workspace)**

```toml
[workspace.dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
```

**File: `crates/sift-store/Cargo.toml`**

```toml
[dev-dependencies]
criterion = { workspace = true }

[[bench]]
name = "vector_search"
harness = false

[[bench]]
name = "fts5_search"
harness = false
```

**File: `crates/sift-store/benches/vector_search.rs`**

```rust
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use sift_core::{Chunk, ContentType, EmbeddedChunk};
use sift_store::FlatVectorIndex;
use sift_store::traits::VectorStore;

fn make_random_vector(dim: usize, seed: u64) -> Vec<f32> {
    // Simple deterministic pseudo-random
    (0..dim)
        .map(|i| ((seed as f32 * 0.618 + i as f32 * 0.317).sin()))
        .collect()
}

fn make_embedded_chunk(uri: &str, dim: usize, seed: u64) -> EmbeddedChunk {
    EmbeddedChunk {
        chunk: Chunk {
            text: format!("chunk text for {}", uri),
            source_uri: uri.to_string(),
            chunk_index: 0,
            content_type: ContentType::Text,
            file_type: "txt".to_string(),
            title: None,
            language: None,
            byte_range: None,
        },
        vector: make_random_vector(dim, seed),
    }
}

fn bench_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity");

    for dim in [128, 256, 512, 768, 1024] {
        let a: Vec<f32> = make_random_vector(dim, 42);
        let b: Vec<f32> = make_random_vector(dim, 99);

        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |bench, _| {
            bench.iter(|| {
                // Call the cosine function (need to make it pub(crate) or test helper)
                let mut dot = 0.0f32;
                let mut na = 0.0f32;
                let mut nb = 0.0f32;
                for i in 0..a.len() {
                    dot += a[i] * b[i];
                    na += a[i] * a[i];
                    nb += b[i] * b[i];
                }
                let denom = (na * nb).sqrt();
                if denom == 0.0 { 0.0f32 } else { dot / denom }
            });
        });
    }
    group.finish();
}

fn bench_vector_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_search");
    let dim = 768;

    for n in [100, 1_000, 10_000, 50_000] {
        let store = FlatVectorIndex::new();
        let chunks: Vec<EmbeddedChunk> = (0..n)
            .map(|i| make_embedded_chunk(&format!("file:///{}.txt", i), dim, i as u64))
            .collect();
        store.insert(&chunks).unwrap();

        let query = make_random_vector(dim, 12345);

        group.bench_with_input(BenchmarkId::new("entries", n), &n, |bench, _| {
            bench.iter(|| store.search(&query, 10).unwrap());
        });
    }
    group.finish();
}

fn bench_vector_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_insert");
    let dim = 768;

    for batch_size in [1, 10, 100, 1000] {
        let chunks: Vec<EmbeddedChunk> = (0..batch_size)
            .map(|i| make_embedded_chunk(&format!("file:///{}.txt", i), dim, i as u64))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &batch_size,
            |bench, _| {
                let store = FlatVectorIndex::new();
                bench.iter(|| store.insert(&chunks).unwrap());
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_cosine_similarity,
    bench_vector_search,
    bench_vector_insert,
);
criterion_main!(benches);
```

**File: `crates/sift-store/benches/fts5_search.rs`**

```rust
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

#[cfg(feature = "fts5")]
fn bench_fts5_search(c: &mut Criterion) {
    use sift_store::Fts5Store;
    use sift_store::traits::FullTextStore;

    let mut group = c.benchmark_group("fts5_search");

    for n in [100, 1_000, 10_000] {
        let store = Fts5Store::open_in_memory().unwrap();

        // Populate with realistic text
        for i in 0..n {
            let text = format!(
                "Document {} about {} and {} with {} considerations",
                i,
                ["performance", "architecture", "testing", "deployment"][i % 4],
                ["Rust", "Python", "JavaScript", "Go"][i % 4],
                ["security", "scalability", "reliability", "maintainability"][i % 4],
            );
            store.insert_text(&format!("file:///{}.txt", i), &text, i as u32).unwrap();
        }

        group.bench_with_input(BenchmarkId::new("docs", n), &n, |bench, _| {
            bench.iter(|| store.search("performance Rust", 10).unwrap());
        });
    }
    group.finish();
}

#[cfg(feature = "fts5")]
criterion_group!(benches, bench_fts5_search);
#[cfg(feature = "fts5")]
criterion_main!(benches);

#[cfg(not(feature = "fts5"))]
fn main() {}
```

#### B) End-to-End CLI Benchmarks

**File: `benchsuite/run.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

# sift Benchmark Suite
# Requires: hyperfine, a test corpus

CORPUS="${SIFT_BENCH_CORPUS:-/tmp/sift-bench-corpus}"
RESULTS_DIR="benchsuite/results"
VX="${SIFT_BIN:-./target/release/sift}"

mkdir -p "$RESULTS_DIR"

echo "=== sift Benchmark Suite ==="
echo "Corpus: $CORPUS"
echo "Binary: $VX"
echo ""

# Ensure corpus exists
if [ ! -d "$CORPUS" ]; then
    echo "Creating test corpus at $CORPUS..."
    bash benchsuite/create_corpus.sh "$CORPUS"
fi

# Clean any existing index
rm -rf ~/.sift/indexes/bench-* 2>/dev/null || true

echo "--- Benchmark 1: Fresh scan (cold) ---"
hyperfine --warmup 0 --runs 3 \
    --export-json "$RESULTS_DIR/fresh_scan.json" \
    "SIFT_INDEX=bench-fresh $VX scan $CORPUS"

echo ""
echo "--- Benchmark 2: Incremental scan (all cached) ---"
SIFT_INDEX=bench-incr $VX scan "$CORPUS" > /dev/null 2>&1
hyperfine --warmup 1 --runs 5 \
    --export-json "$RESULTS_DIR/incremental_scan.json" \
    "SIFT_INDEX=bench-incr $VX scan $CORPUS"

echo ""
echo "--- Benchmark 3: Keyword search ---"
SIFT_INDEX=bench-search $VX scan "$CORPUS" > /dev/null 2>&1
hyperfine --warmup 3 --runs 10 \
    --export-json "$RESULTS_DIR/keyword_search.json" \
    "SIFT_INDEX=bench-search $VX search --keyword-only 'function error handling'" \
    "SIFT_INDEX=bench-search $VX search --keyword-only 'import os sys'" \
    "SIFT_INDEX=bench-search $VX search --keyword-only 'database connection pool'"

echo ""
echo "--- Benchmark 4: Dry run (discovery only) ---"
hyperfine --warmup 3 --runs 5 \
    --export-json "$RESULTS_DIR/dry_run.json" \
    "SIFT_INDEX=bench-dry $VX scan --dry-run $CORPUS"

echo ""
echo "=== Results saved to $RESULTS_DIR ==="
```

**File: `benchsuite/create_corpus.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

CORPUS="$1"
mkdir -p "$CORPUS"

echo "Generating test corpus at $CORPUS..."

# Generate code files
for i in $(seq 1 2000); do
    dir="$CORPUS/src/module_$(printf '%03d' $((i % 50)))"
    mkdir -p "$dir"
    cat > "$dir/file_$i.rs" << 'RUST'
use std::collections::HashMap;

/// A sample struct for benchmarking.
pub struct Handler {
    config: HashMap<String, String>,
    count: usize,
}

impl Handler {
    pub fn new() -> Self {
        Self {
            config: HashMap::new(),
            count: 0,
        }
    }

    pub fn process(&mut self, input: &str) -> Result<String, Box<dyn std::error::Error>> {
        self.count += 1;
        let result = format!("processed: {} (count: {})", input, self.count);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process() {
        let mut h = Handler::new();
        assert!(h.process("test").is_ok());
    }
}
RUST
done

# Generate text files
for i in $(seq 1 1000); do
    dir="$CORPUS/docs/section_$(printf '%02d' $((i % 20)))"
    mkdir -p "$dir"
    cat > "$dir/doc_$i.md" << 'MD'
# Performance Optimization Guide

This document covers various performance optimization techniques
for building high-performance CLI tools in Rust.

## Key Principles

1. Minimize allocations in hot paths
2. Use SIMD for data-parallel operations
3. Batch I/O operations
4. Leverage parallel processing with rayon

## Database Optimization

When using SQLite, always enable WAL mode and batch writes
within transactions for maximum throughput.

## Error Handling

Use proper error types with thiserror for zero-cost abstractions.
Avoid dynamic dispatch in performance-critical paths.
MD
done

# Generate JSON/CSV data files
for i in $(seq 1 500); do
    echo '{"id":'$i',"name":"item_'$i'","value":'$((RANDOM % 1000))'}' > "$CORPUS/data/item_$i.json"
done 2>/dev/null
mkdir -p "$CORPUS/data"
for i in $(seq 1 500); do
    echo '{"id":'$i',"name":"item_'$i'","value":'$((RANDOM % 1000))'}' > "$CORPUS/data/item_$i.json"
done

echo "Created corpus with ~3500 files"
find "$CORPUS" -type f | wc -l
```

#### C) CI Integration

Add a benchmark job to `.github/workflows/ci.yml` (run on schedule, not every PR):

```yaml
  benchmark:
    name: Benchmarks
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - name: Install system dependencies
        run: sudo apt-get update && sudo apt-get install -y pkg-config libssl-dev libbz2-dev
      - name: Run criterion benchmarks
        run: cargo bench --workspace --features fts5,sqlite
      - name: Archive benchmark results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: target/criterion/
```

## File Structure

```
benchsuite/
  run.sh              # End-to-end CLI benchmarks
  create_corpus.sh    # Generate test corpus
  results/            # Benchmark output (gitignored)
crates/sift-store/
  benches/
    vector_search.rs  # Criterion: cosine, vector search, insert
    fts5_search.rs    # Criterion: FTS5 keyword search
```

## Running Benchmarks

```bash
# Microbenchmarks (criterion)
cargo bench -p sift-store --features fts5,sqlite

# End-to-end benchmarks (hyperfine)
cargo build --release
bash benchsuite/run.sh

# Compare before/after a change
cargo bench -p sift-store -- --save-baseline before
# ... make changes ...
cargo bench -p sift-store -- --baseline before
```

## Evaluation Metric

The benchsuite itself is the evaluation infrastructure. Each optimization PR should include benchmark results showing:

1. **Before/after wall-clock times** for affected operations
2. **Criterion regression check** (`--baseline`) for microbenchmarks
3. **Memory usage comparison** (peak RSS via `/usr/bin/time -v`)

Target metrics to track:
- `sift scan --dry-run` latency (discovery + filtering)
- `sift scan` total latency (full pipeline)
- `sift search --keyword-only` latency
- `sift search` latency (hybrid)
- Vector search throughput (queries/sec at various index sizes)
- Peak RSS during scan
