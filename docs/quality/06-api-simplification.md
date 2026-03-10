# 06: API Simplification & Dead Code

**Category**: Simplicity
**Severity**: Medium
**Effort**: Low
**Crates**: all

## Problem

### 1. Legacy wrapper functions

```rust
// output.rs:41-43
pub fn print_search_results(results: &[SearchResult], format: &OutputFormat, show_context: bool) {
    format_search_results(results, format, show_context);
}

// output.rs:160-162
pub fn print_index_stats(stats: &IndexStats, format: &OutputFormat) {
    format_stats(stats, format);
}
```

These one-line wrappers add no value. They exist as "legacy" aliases. Delete them and update call sites.

### 2. Hand-rolled JSON serialization in BM25

```rust
// bm25.rs:83-126
fn serialize(inner: &Bm25Inner) -> String {
    let mut docs_json = Vec::new();
    for (&id, meta) in &inner.docs {
        let title = meta.title.as_ref()
            .map(|t| format!(",\"title\":\"{}\"", json_escape(t)))
            .unwrap_or_default();
        // ... 40 lines of manual JSON construction ...
    }
}
```

This bypasses serde entirely and uses a hand-written `json_escape` function that doesn't handle all edge cases (null bytes, unicode escape sequences, forward slashes). The `Bm25Inner` types already have serde derives available.

**Fix**: Make `Bm25Inner` serializable:

```rust
#[derive(Serialize, Deserialize)]
struct Bm25Inner {
    index: HashMap<String, Vec<(u32, f32)>>,
    docs: HashMap<u32, DocMeta>,
    doc_count: u32,
    avg_dl: f64,
    next_id: u32,
}

fn serialize(inner: &Bm25Inner) -> String {
    serde_json::to_string(inner).expect("BM25 serialization cannot fail")
}

fn deserialize(data: &str) -> SiftResult<Bm25Inner> {
    serde_json::from_str(data).map_err(|e| SiftError::Storage(e.to_string()))
}
```

This eliminates ~80 lines of fragile hand-rolled serialization code.

### 3. Hand-rolled date parsing

```rust
// main.rs:508-512
let m = if month <= 2 { month + 9 } else { month - 3 };
let y = if month <= 2 { year - 1 } else { year };
let days_from_epoch = 365 * y + y / 4 - y / 100 + y / 400 + (m * 306 + 5) / 10 + day - 1 - 719468;
```

This is a correct algorithm (Rata Die / civil-to-days), but it's complex, untested for edge cases (leap year boundaries, century years), and the relative duration parsing approximates months as 30 days.

**Fix**: Use the `jiff` or `time` crate. `time` is lightweight (160KB):

```rust
// time = "0.3"
use time::Date;

fn parse_after_date(s: &str) -> anyhow::Result<i64> {
    // ... relative duration parsing stays the same ...

    // ISO 8601 parsing
    let date = Date::parse(s, &time::format_description::well_known::Iso8601::DEFAULT)?;
    let epoch = Date::from_calendar_date(1970, time::Month::January, 1)?;
    Ok((date - epoch).whole_seconds())
}
```

Or, if you want to keep zero dependencies, at least add test coverage for the edge cases: Feb 29 in leap years, Dec 31, Jan 1, year 2100 (not a leap year), etc.

### 4. `Source` trait uses callback instead of iterator

```rust
// traits.rs
pub trait Source: Send + Sync {
    fn discover(
        &self,
        options: &ScanOptions,
        callback: &mut dyn FnMut(SourceItem) -> SiftResult<()>,
    ) -> SiftResult<u64>;
}
```

The callback pattern is unusual in Rust. The pipeline immediately collects all items into a `Vec`:

```rust
// pipeline.rs:222-226
let mut items: Vec<SourceItem> = Vec::new();
source.discover(options, &mut |item| {
    items.push(item);
    Ok(())
})?;
```

The callback provides no benefit here. An iterator or direct `Vec` return would be simpler:

```rust
pub trait Source: Send + Sync {
    fn discover(&self, options: &ScanOptions) -> SiftResult<Vec<SourceItem>>;
}
```

If streaming is desired (to start processing before discovery completes), return a channel receiver or implement `Iterator`.

### 5. `check_source` is unused outside tests

`MetadataStore::check_source` exists in both `metadata.rs` and `json_metadata.rs`, but the pipeline uses `load_all_hashes` for batch lookups. The method is only called in unit tests.

This is dead API weight. Either remove it (the batch approach is strictly better) or mark it `#[cfg(test)]`.

### 6. Feature flag proliferation

The CLI crate has 22 feature flags, many of which are thin passthroughs:

```toml
pdf = ["sift-parsers/pdf"]
spreadsheets = ["sift-parsers/spreadsheets"]
email = ["sift-parsers/email"]
office = ["sift-parsers/office"]
epub = ["sift-parsers/epub"]
archive = ["sift-parsers/archive"]
data = ["sift-parsers/data"]
```

Each of these adds conditional compilation complexity. Consider collapsing format-specific features into coarser groups:

```toml
documents = ["sift-parsers/pdf", "sift-parsers/office", "sift-parsers/epub"]
data-formats = ["sift-parsers/spreadsheets", "sift-parsers/data"]
```

Or simply make all lightweight parsers default-on (they add minimal binary size since they're just zip/xml readers).

### 7. `color_stub` returns `&str` vs `colored` returns `ColoredString`

The `Colorize` stub trait returns `&str`:
```rust
fn red(&self) -> &str { self }
```

But the `colored` crate's `Colorize` trait returns `ColoredString`. This means call sites like:
```rust
stats.errors.to_string().red().to_string()
```
compile differently depending on the feature flag — with `fancy`, `.red()` returns a `ColoredString` which `.to_string()` formats with ANSI codes. Without `fancy`, `.red()` returns `&str` which `.to_string()` copies.

The mismatch is subtle but can cause type errors if code paths evolve. Consider using `colored`'s own `ColoredString::normal()` pattern or making the stub return `String`.

## Summary

| Fix | Lines Removed | Lines Added | Net |
|-----|--------------|-------------|-----|
| Delete legacy wrappers | 6 | 0 | -6 |
| Serde for BM25 | ~80 | ~10 | -70 |
| Remove check_source from public API | ~20 | 0 | -20 |
| Simplify Source trait | ~8 | ~3 | -5 |
| Collapse feature flags | ~15 | ~5 | -10 |
| **Total** | **~129** | **~18** | **~-111** |
