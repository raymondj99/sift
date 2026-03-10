# 02: Parallel Directory Walking

## Problem

In `sift-sources/src/filesystem.rs:135`, the `ignore` crate is used with its **sequential** walker (`builder.build()`), not its parallel walker (`builder.build_parallel()`). This means file discovery is single-threaded, which is the first stage of the pipeline and blocks everything downstream.

ripgrep's key insight: the `ignore` crate provides a parallel directory walker that uses work-stealing across threads, so wide directory trees get traversed much faster on multi-core systems.

## Solution

Replace `builder.build()` (sequential `Walk` iterator) with `builder.build_parallel()` (parallel `WalkParallel`), which uses the `ParallelVisitor` pattern to walk directories across multiple threads.

### Implementation

**File: `crates/sift-sources/src/filesystem.rs`**

Update the `Source` trait to support a thread-safe callback, and use `build_parallel()`:

```rust
use crate::traits::Source;
use std::path::Path;
use std::sync::Mutex;
use tracing::debug;
use sift_core::{ScanOptions, SourceItem, SiftResult};

pub struct FilesystemSource;

impl FilesystemSource {
    pub fn new() -> Self {
        Self
    }

    fn read_and_analyze(path: &Path) -> SiftResult<([u8; 32], Option<String>)> {
        let data = std::fs::read(path)?;
        let mime = infer::get(&data).map(|kind| kind.mime_type().to_string());
        let hash = *blake3::hash(&data).as_bytes();
        Ok((hash, mime))
    }

    fn mime_from_extension(path: &Path) -> Option<String> {
        let ext = path.extension()?.to_str()?;
        // ... existing match block ...
    }
}

impl Source for FilesystemSource {
    fn discover(
        &self,
        options: &ScanOptions,
        callback: &mut dyn FnMut(SourceItem) -> SiftResult<()>,
    ) -> SiftResult<u64> {
        let count = std::sync::atomic::AtomicU64::new(0);
        let items: Mutex<Vec<SourceItem>> = Mutex::new(Vec::new());

        for scan_path in &options.paths {
            let mut builder = ignore::WalkBuilder::new(scan_path);
            builder
                .hidden(true)
                .git_ignore(true)
                .git_global(true)
                .git_exclude(true)
                .threads(options.jobs.max(1)); // Use job count for parallelism

            if let Some(max_depth) = options.max_depth {
                builder.max_depth(Some(max_depth));
            }

            // Add custom ignore patterns
            let mut overrides = ignore::overrides::OverrideBuilder::new(scan_path);
            for pattern in &options.exclude_globs {
                let _ = overrides.add(&format!("!{}", pattern));
            }
            for pattern in &options.include_globs {
                let _ = overrides.add(pattern);
            }
            if let Ok(ov) = overrides.build() {
                builder.overrides(ov);
            }

            // Use parallel walker
            let max_file_size = options.max_file_size;
            let file_types = options.file_types.clone();
            let items_ref = &items;
            let count_ref = &count;

            builder.build_parallel().run(|| {
                // Each thread gets its own closure
                Box::new(move |entry| {
                    let entry = match entry {
                        Ok(e) => e,
                        Err(e) => {
                            debug!("Walk error: {}", e);
                            return ignore::WalkState::Continue;
                        }
                    };

                    let path = entry.path();
                    if !path.is_file() {
                        return ignore::WalkState::Continue;
                    }

                    // Check file size
                    let metadata = match path.metadata() {
                        Ok(m) => m,
                        Err(_) => return ignore::WalkState::Continue,
                    };
                    let size = metadata.len();
                    if let Some(max_size) = max_file_size {
                        if size > max_size {
                            debug!("Skipping {} (too large: {} bytes)", path.display(), size);
                            return ignore::WalkState::Continue;
                        }
                    }

                    // Single-pass: read once for MIME + hash
                    let (content_hash, content_mime) = match FilesystemSource::read_and_analyze(path) {
                        Ok(r) => r,
                        Err(e) => {
                            debug!("Read error for {}: {}", path.display(), e);
                            return ignore::WalkState::Continue;
                        }
                    };

                    let extension = path
                        .extension()
                        .and_then(|e| e.to_str())
                        .map(|s| s.to_lowercase());

                    let mime_type = content_mime
                        .or_else(|| FilesystemSource::mime_from_extension(path));

                    // Filter by file type
                    if !file_types.is_empty() {
                        let matches = file_types.iter().any(|ft| {
                            extension.as_deref() == Some(ft.as_str())
                                || mime_type
                                    .as_deref()
                                    .is_some_and(|m| m.contains(ft.as_str()))
                        });
                        if !matches {
                            return ignore::WalkState::Continue;
                        }
                    }

                    if mime_type.is_none() && extension.is_none() {
                        return ignore::WalkState::Continue;
                    }

                    let modified_at = metadata
                        .modified()
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_secs() as i64);

                    let item = SourceItem {
                        uri: format!(
                            "file://{}",
                            path.canonicalize().unwrap_or(path.to_path_buf()).display()
                        ),
                        path: path.to_path_buf(),
                        content_hash,
                        size,
                        modified_at,
                        mime_type,
                        extension,
                    };

                    if let Ok(mut items) = items_ref.lock() {
                        items.push(item);
                    }
                    count_ref.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                    ignore::WalkState::Continue
                })
            });
        }

        // Drain collected items through the callback
        let collected = items.into_inner().map_err(|e| {
            sift_core::SiftError::Source(format!("Lock poisoned: {}", e))
        })?;
        for item in collected {
            callback(item)?;
        }

        Ok(count.load(std::sync::atomic::Ordering::Relaxed))
    }
}
```

### Key Design Decisions (from ripgrep)

1. **`build_parallel()` uses work-stealing** - threads that finish early steal work from slow threads
2. **Depth-first traversal** - ripgrep switched from BFS to DFS (PR #1554) to reduce peak memory on wide trees
3. **Thread count matches job count** - the `threads()` builder method controls parallel walker thread count
4. **Each thread gets its own visitor** - avoids contention; we collect into a Mutex<Vec> then drain

### Note on ScanOptions

Add a `jobs` field to `ScanOptions` if not already present (currently it's in the CLI options). The parallel walker needs to know how many threads to use.

## Tests

```rust
#[test]
fn test_parallel_discover_finds_all_files() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("a.txt"), "aaa").unwrap();
    fs::write(dir.path().join("b.rs"), "fn main() {}").unwrap();
    fs::create_dir(dir.path().join("sub")).unwrap();
    fs::write(dir.path().join("sub/c.md"), "# Title").unwrap();

    let source = FilesystemSource::new();
    let mut options = ScanOptions {
        paths: vec![dir.path().to_path_buf()],
        ..Default::default()
    };
    options.jobs = 4; // Force parallel

    let mut items = vec![];
    let count = source
        .discover(&options, &mut |item| {
            items.push(item);
            Ok(())
        })
        .unwrap();

    assert_eq!(count, 3);
    assert_eq!(items.len(), 3);
}

#[test]
fn test_parallel_discover_respects_max_file_size() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("small.txt"), "hi").unwrap();
    fs::write(dir.path().join("big.txt"), "x".repeat(1000)).unwrap();

    let source = FilesystemSource::new();
    let options = ScanOptions {
        paths: vec![dir.path().to_path_buf()],
        max_file_size: Some(100),
        ..Default::default()
    };

    let mut items = vec![];
    source
        .discover(&options, &mut |item| {
            items.push(item);
            Ok(())
        })
        .unwrap();

    assert_eq!(items.len(), 1);
    assert!(items[0].uri.contains("small.txt"));
}

#[test]
fn test_parallel_discover_deep_nested() {
    let dir = TempDir::new().unwrap();
    let mut path = dir.path().to_path_buf();
    for i in 0..10 {
        path = path.join(format!("level{}", i));
        fs::create_dir(&path).unwrap();
    }
    fs::write(path.join("deep.txt"), "deep content").unwrap();

    let source = FilesystemSource::new();
    let options = ScanOptions {
        paths: vec![dir.path().to_path_buf()],
        ..Default::default()
    };

    let mut items = vec![];
    source.discover(&options, &mut |item| {
        items.push(item);
        Ok(())
    }).unwrap();

    assert_eq!(items.len(), 1);
    assert!(items[0].uri.contains("deep.txt"));
}
```

## Evaluation Metric

**Benchmark: Scan a large directory tree with 50,000+ files**

```bash
# Generate test corpus with nested structure
mkdir -p /tmp/sift-bench-corpus
# ... populate with files ...

# Before (sequential)
hyperfine --warmup 3 'sift scan --dry-run /tmp/sift-bench-corpus'

# After (parallel)
hyperfine --warmup 3 'sift scan --dry-run /tmp/sift-bench-corpus'

# Compare with different thread counts
hyperfine --warmup 3 \
  'sift scan --dry-run --jobs 1 /tmp/sift-bench-corpus' \
  'sift scan --dry-run --jobs 4 /tmp/sift-bench-corpus' \
  'sift scan --dry-run --jobs 8 /tmp/sift-bench-corpus'
```

Expected improvement: **2-4x faster** discovery on multi-core systems with wide directory trees.
