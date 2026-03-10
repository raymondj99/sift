# 06: Global Allocator (jemalloc)

## Problem

The default system allocator on most platforms is not optimized for the allocation patterns of multi-threaded Rust programs. sift's pipeline has several threads doing frequent small allocations (strings, vectors, Vec<f32>) and the default allocator can become a contention point.

ripgrep and many high-performance Rust CLI tools use `jemalloc` as the global allocator for measurably better performance on Linux and macOS.

## Solution

Use `tikv-jemallocator` as the global allocator on supported platforms (Linux, macOS). Windows and other platforms fall back to the system allocator.

### Implementation

**File: `crates/sift-cli/Cargo.toml`**

```toml
[target.'cfg(not(target_env = "msvc"))'.dependencies]
tikv-jemallocator = "0.6"
```

**File: `crates/sift-cli/src/main.rs`**

Add at the top of the file (before any other code):

```rust
#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
```

### Why jemalloc over mimalloc?

- **jemalloc**: Better for long-running multi-threaded workloads. More memory-efficient under sustained allocation/deallocation. Used by ripgrep, fd, Firefox.
- **mimalloc**: Better for short bursts. Can have higher RSS. Recent reports of performance regression on some workloads (microsoft/mimalloc#1119).
- **System allocator**: Fine for simple programs, but scales poorly under thread contention.

For sift's pipeline pattern (multiple threads doing parse/chunk work with frequent string allocations), jemalloc is the best fit.

### Platform Notes

- **macOS**: jemalloc works well, replaces the built-in malloc which is already decent
- **Linux**: Significant improvement over glibc malloc for multi-threaded workloads
- **Windows (MSVC)**: jemalloc doesn't compile easily; skip it (system allocator is OK)
- **musl**: Works but may need static linking configuration

## Tests

This is a transparent change - all existing tests should pass unchanged. Add a basic smoke test:

```rust
#[test]
fn test_allocator_works() {
    // Verify jemalloc is functional by doing typical allocations
    let mut vecs: Vec<Vec<f32>> = Vec::new();
    for _ in 0..1000 {
        vecs.push(vec![0.0f32; 768]); // Typical embedding vector
    }
    assert_eq!(vecs.len(), 1000);
    assert_eq!(vecs[0].len(), 768);
}

#[test]
fn test_allocator_multithreaded() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let counter = Arc::new(AtomicUsize::new(0));
    let threads: Vec<_> = (0..8).map(|_| {
        let counter = counter.clone();
        std::thread::spawn(move || {
            for _ in 0..1000 {
                let v: Vec<String> = (0..100).map(|i| format!("string_{}", i)).collect();
                counter.fetch_add(v.len(), Ordering::Relaxed);
            }
        })
    }).collect();

    for t in threads {
        t.join().unwrap();
    }

    assert_eq!(counter.load(Ordering::Relaxed), 8 * 1000 * 100);
}
```

## Evaluation Metric

**Benchmark: Full scan pipeline with 10,000 files**

```bash
# Build two binaries
cargo build --release -p sift-cli                    # with jemalloc
cargo build --release -p sift-cli --features no-jemalloc  # without (if feature-gated)

# Compare
hyperfine --warmup 3 \
  './target/release/sift-jemalloc scan /path/to/corpus' \
  './target/release/sift-system scan /path/to/corpus'

# Also measure peak RSS
/usr/bin/time -v ./target/release/sift scan /path/to/corpus 2>&1 | grep "Maximum resident"
```

Expected improvement: **5-15% faster** scan throughput, **10-20% lower peak RSS** under multi-threaded workloads. The improvement is most visible when parsing many small files (lots of string allocations across threads).
