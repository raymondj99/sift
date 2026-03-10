# 01: Error Handling Overhaul

**Category**: Robustness
**Severity**: High
**Effort**: Medium
**Crates**: all

## Problem

### 1. String-based error variants lose type information

`SiftError` has 6 variants that wrap `String`:

```rust
// sift-core/src/error.rs
pub enum SiftError {
    Io(#[from] std::io::Error),
    Config(String),
    Parse { path: String, message: String },
    Embedding(String),
    Storage(String),
    Search(String),
    Model(String),
    Source(String),
    Other(#[from] anyhow::Error),
}
```

Every call site does manual string formatting:

```rust
// This pattern appears 40+ times across the codebase
conn.execute_batch("...")
    .map_err(|e| sift_core::SiftError::Storage(format!("SQLite pragma error: {}", e)))?;
```

This loses the original error type, makes programmatic error matching impossible, and creates enormous boilerplate. Compare ripgrep's approach: it wraps actual error types via `#[from]` and uses proper error chains.

### 2. Silent error swallowing

Multiple locations silently discard errors:

| Location | Code | Risk |
|----------|------|------|
| `filesystem.rs:144` | `let _ = overrides.add(...)` | Invalid glob patterns silently ignored |
| `metadata.rs:187` | `.filter_map(\|r\| r.ok())` | SQLite row errors silently dropped |
| `pipeline.rs:470` | `let _ = engine.delete_by_uri(...)` | Failed deletions during re-index go unnoticed |
| `pipeline.rs:398` | `let _ = parse_tx.send(parsed)` | Dropped chunks with no logging |

### 3. Lock error boilerplate

Every `Mutex::lock()` call across the codebase wraps the PoisonError manually:

```rust
let conn = self.conn.lock()
    .map_err(|e| sift_core::SiftError::Storage(format!("Lock error: {}", e)))?;
```

The `hnsw.rs` module introduced a `lock_err` helper but it's not used elsewhere.

### 4. Dual error type antipattern

The CLI uses `anyhow::Result` at the top level, while library crates use `SiftResult<T> = Result<T, SiftError>`. `SiftError::Other(#[from] anyhow::Error)` creates a bridge, but now errors can be wrapped in either type and conversion is lossy.

## Proposed Fix

### Step 1: Add proper error type wrapping

```rust
pub enum SiftError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Config error: {0}")]
    Config(#[from] toml::de::Error),

    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Parse error for {path}: {message}")]
    Parse { path: String, message: String },

    #[error("Embedding error: {0}")]
    Embedding(String),  // keep String — ort errors are complex

    #[error("{context}: {source}")]
    Storage {
        context: &'static str,
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}
```

This eliminates ~30 `.map_err(format!(...))` calls. The `rusqlite::Error` and `serde_json::Error` conversions happen automatically via `?`.

### Step 2: Introduce a lock helper in sift-core

```rust
// sift-core/src/error.rs
pub fn lock_err<T>(description: &'static str) -> impl FnOnce(std::sync::PoisonError<T>) -> SiftError {
    move |e| SiftError::Storage {
        context: description,
        source: Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())),
    }
}
```

Then all lock sites become:

```rust
let conn = self.conn.lock().map_err(sift_core::lock_err("metadata store"))?;
```

### Step 3: Log-and-continue instead of silent discard

Replace `let _ =` with explicit logging:

```rust
// Before
let _ = engine.delete_by_uri(&store_item.item.uri);

// After
if let Err(e) = engine.delete_by_uri(&store_item.item.uri) {
    warn!("Failed to delete old chunks for {}: {}", store_item.item.uri, e);
}
```

### Step 4: Drop `anyhow` from library crates

Keep `anyhow` only in `sift-cli`. Library crates should use `SiftError` exclusively. Remove `SiftError::Other(#[from] anyhow::Error)`.

## Verification

```bash
# After refactoring, these should produce zero new warnings:
cargo clippy --workspace --all-targets
cargo test --workspace
```

Count the number of `.map_err(|e| SiftError::` occurrences before and after — target is reducing from ~40 to <10.
