# 03: Hot-Path Allocations & Redundant I/O

**Category**: Performance
**Severity**: High
**Effort**: Low
**Crates**: sift-sources, sift-cli, sift-store, sift-parsers

## Problem

### 1. `ParserRegistry` allocated per-file inside rayon worker

```rust
// pipeline.rs:336
discover_rx.into_iter().par_bridge().for_each(|item| {
    let parser_registry = ParserRegistry::new();  // <-- allocates Vec<Box<dyn Parser>> every iteration
    // ...
});
```

`ParserRegistry::new()` allocates 10-14 boxed parser objects. For 10,000 files, that's 100,000+ heap allocations that are immediately discarded. The registry is stateless — it should be created once and shared.

**Fix**: Create the registry before the parallel loop. `ParserRegistry` is `Send + Sync` (all parsers are unit structs implementing `Send + Sync`), so it can be shared via `&`:

```rust
let parser_registry = ParserRegistry::new();
discover_rx.into_iter().par_bridge().for_each(|item| {
    // Use &parser_registry
});
```

### 2. `canonicalize()` syscall on every file

```rust
// filesystem.rs:228-231
uri: format!(
    "file://{}",
    path.canonicalize().unwrap_or(path.to_path_buf()).display()
),
```

`canonicalize()` is a syscall that resolves symlinks and normalizes the path. It's called for every discovered file. On a 50,000 file scan, that's 50,000 additional syscalls. The `ignore` crate already returns canonical-enough paths.

**Fix**: Use the path directly, or call `std::fs::canonicalize` only on the root scan paths and construct child URIs relative to those:

```rust
uri: format!("file://{}", path.display()),
```

### 3. Full file read into memory for MIME detection

```rust
// filesystem.rs:108
pub(crate) fn read_and_analyze(path: &Path) -> SiftResult<([u8; 32], Option<String>)> {
    let data = std::fs::read(path)?;  // reads entire file — up to 100MB
    let mime = infer::get(&data).map(|kind| kind.mime_type().to_string());
    let hash = *blake3::hash(&data).as_bytes();
    Ok((hash, mime))
}
```

`infer::get()` only examines the first few bytes (magic number detection). But we read the entire file into a single `Vec<u8>`. For a 50MB PDF, this allocates 50MB just to check 8 bytes of magic.

**Fix**: Read a small header for MIME, then stream-hash the full file:

```rust
pub(crate) fn read_and_analyze(path: &Path) -> SiftResult<([u8; 32], Option<String>)> {
    let mut file = std::fs::File::open(path)?;

    // Read header for MIME detection (infer needs at most ~8KB)
    let mut header = [0u8; 8192];
    let header_len = std::io::Read::read(&mut file, &mut header)?;
    let mime = infer::get(&header[..header_len]).map(|k| k.mime_type().to_string());

    // Stream-hash the full file (BLAKE3 supports streaming)
    file.seek(std::io::SeekFrom::Start(0))?;
    let mut hasher = blake3::Hasher::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = std::io::Read::read(&mut file, &mut buf)?;
        if n == 0 { break; }
        hasher.update(&buf[..n]);
    }

    Ok((*hasher.finalize().as_bytes(), mime))
}
```

Note: the file is also read again in the parse stage (`pipeline.rs:339`: `std::fs::read(&item.path)`). The initial `read_and_analyze` read could potentially be reused if the file is small enough, but that's a separate optimization (see doc 04).

### 4. `format!` allocations in RRF fusion keys

```rust
// hybrid.rs:85
let key = format!("{}:{}", result.uri, result.chunk_index);
```

This allocates a new `String` for every search result in both the vector and BM25 result sets. For `fetch_k = top_k * 3 = 30` results from each source, that's 60 allocations per search query.

**Fix**: Use a tuple key to avoid the allocation:

```rust
let mut scores: HashMap<(&str, u32), (SearchResult, f32)> = HashMap::new();
// ...
let key = (result.uri.as_str(), result.chunk_index);
```

### 5. MIME type string allocations on every file

```rust
// filesystem.rs:23-89
pub(crate) fn mime_from_extension(path: &Path) -> Option<String> {
    // ...
    Some(match ext {
        "txt" => "text/plain",
        // ...
    }.to_string())  // <-- allocates a new String every time
}
```

MIME types are static strings. Return `&'static str` instead of `String`:

```rust
pub(crate) fn mime_from_extension(path: &Path) -> Option<&'static str> {
    // ...
    Some(match ext {
        "txt" => "text/plain",
        // ...
    })
}
```

This cascades to `SourceItem.mime_type: Option<&'static str>` or at minimum avoids the `.to_string()` call. If `SourceItem` needs to own the string (for content-based MIME from `infer`), use `Cow<'static, str>`.

### 6. `read_context_lines` reads entire file

```rust
// output.rs:109-117
for (line_num, line_result) in reader.lines().enumerate() {
    let line = line_result.ok()?;
    // ... stores ALL lines in memory
    lines.push((line_num, line));
    offset = line_end;
}
```

This reads and stores every line of a file just to display 5 lines around a byte offset. For a 10,000-line file, this allocates 10,000 strings.

**Fix**: Seek to `start_byte - some_margin`, read a window, then find the target line within that window.

## Impact Estimates

| Fix | Files Affected | Allocations Saved (10K file scan) |
|-----|---------------|----------------------------------|
| ParserRegistry reuse | pipeline.rs | ~100,000 |
| Drop canonicalize | filesystem.rs | 10,000 syscalls |
| Stream MIME+hash | filesystem.rs | Peak memory: 100MB → 64KB |
| RRF tuple keys | hybrid.rs | ~60 per search |
| Static MIME strings | filesystem.rs, types.rs | ~10,000 |
