# 01: Single-Pass File Discovery

## Problem

In `sift-sources/src/filesystem.rs`, every discovered file is read **twice**:

1. `detect_mime()` at line 20: `std::fs::read(path)` to detect MIME via `infer` crate
2. `hash_file()` at line 14: `std::fs::read(path)` to compute BLAKE3 hash

For a 10,000-file scan with average 50KB files, this means ~1GB of redundant I/O.

## Solution

Combine MIME detection and BLAKE3 hashing into a single file read. The `infer` crate only needs the first 8KB of a file to detect its type. BLAKE3 needs the full content. Read the file once and feed the same buffer to both.

### Implementation

**File: `crates/sift-sources/src/filesystem.rs`**

Replace the two separate methods with a single `read_and_analyze` method:

```rust
/// Read a file once, returning (content_hash, mime_type).
/// The `infer` crate only inspects the first few KB, so we pass the
/// already-read buffer to both `infer::get` and `blake3::hash`.
fn read_and_analyze(path: &Path) -> SiftResult<([u8; 32], Option<String>)> {
    let data = std::fs::read(path)?;

    // MIME detection from content (first bytes)
    let mime = infer::get(&data).map(|kind| kind.mime_type().to_string());

    // BLAKE3 hash of full content
    let hash = *blake3::hash(&data).as_bytes();

    Ok((hash, mime))
}
```

Then update the `discover` method to call `read_and_analyze` once:

```rust
// In the discover loop, replace lines 162-192 with:
let (content_hash, content_mime) = match Self::read_and_analyze(path) {
    Ok(result) => result,
    Err(e) => {
        debug!("Read error for {}: {}", path.display(), e);
        continue;
    }
};

// Fall back to extension-based MIME if content detection failed
let mime_type = content_mime.or_else(|| Self::mime_from_extension(path));
```

Extract the extension-based MIME lookup into a separate `mime_from_extension` method (the existing match block starting at line 28).

### For Large Files: Streaming Hash

For files larger than a configurable threshold (e.g., 10MB), use BLAKE3's streaming API to avoid loading the entire file into memory, while still reading the first 8KB for MIME detection:

```rust
fn read_and_analyze_large(path: &Path, threshold: u64) -> SiftResult<([u8; 32], Option<String>)> {
    let file = std::fs::File::open(path)?;
    let file_size = file.metadata()?.len();

    if file_size <= threshold {
        // Small file: read all at once
        let data = std::fs::read(path)?;
        let mime = infer::get(&data).map(|k| k.mime_type().to_string());
        let hash = *blake3::hash(&data).as_bytes();
        return Ok((hash, mime));
    }

    // Large file: stream through BLAKE3, detect MIME from first 8KB
    use std::io::Read;
    let mut reader = std::io::BufReader::new(file);
    let mut hasher = blake3::Hasher::new();
    let mut header = vec![0u8; 8192.min(file_size as usize)];
    reader.read_exact(&mut header)?;
    let mime = infer::get(&header).map(|k| k.mime_type().to_string());
    hasher.update(&header);

    let mut buf = vec![0u8; 64 * 1024]; // 64KB buffer
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 { break; }
        hasher.update(&buf[..n]);
    }

    Ok((*hasher.finalize().as_bytes(), mime))
}
```

## Tests

Add to `crates/sift-sources/src/filesystem.rs`:

```rust
#[test]
fn test_read_and_analyze_returns_hash_and_mime() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.txt");
    fs::write(&path, "hello world").unwrap();

    let (hash, mime) = FilesystemSource::read_and_analyze(&path).unwrap();
    // Hash should be deterministic
    let expected_hash = *blake3::hash(b"hello world").as_bytes();
    assert_eq!(hash, expected_hash);
    // .txt won't be detected by infer (it uses magic bytes), mime should be None
    assert!(mime.is_none());
}

#[test]
fn test_read_and_analyze_detects_png_mime() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.png");
    // Write minimal PNG header
    let png_header = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR";
    fs::write(&path, png_header).unwrap();

    let (_, mime) = FilesystemSource::read_and_analyze(&path).unwrap();
    assert_eq!(mime.as_deref(), Some("image/png"));
}

#[test]
fn test_single_pass_matches_separate_calls() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("code.rs");
    fs::write(&path, "fn main() { println!(\"hello\"); }").unwrap();

    let separate_hash = FilesystemSource::hash_file(&path).unwrap();
    let (combined_hash, _) = FilesystemSource::read_and_analyze(&path).unwrap();
    assert_eq!(separate_hash, combined_hash);
}
```

## Evaluation Metric

**Benchmark: Scan 10,000 mixed files (text, code, binary)**

```
Before: Time to complete discovery phase (file I/O dominated)
After:  Time to complete discovery phase with single-pass

Expected improvement: ~40-50% reduction in discovery I/O time
Metric: wall-clock time for `sift scan --dry-run` on a test corpus
```

Add to benchsuite:
```bash
# Create test corpus
hyperfine --warmup 3 \
  'sift scan --dry-run /path/to/test-corpus' \
  --export-json bench-discovery.json
```
