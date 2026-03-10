# 04: Batch Metadata Queries

## Problem

In `sift-cli/src/pipeline.rs:237-247`, after discovering all files, we check each file's hash against the metadata store **one at a time**:

```rust
for item in items {
    match metadata.check_source(&item.uri, &item.content_hash)? {
        Some(true) => { stats.skipped += 1; }
        _ => { to_process.push(item); }
    }
}
```

Each `check_source()` call (in `sift-store/src/metadata.rs:77-93`) acquires a Mutex, prepares a statement, and executes a SQLite query. For 10,000 files, that's 10,000 Mutex lock/unlock cycles and 10,000 individual SQL queries.

## Solution

Add a batch method to `MetadataStore` that loads all known content hashes into a `HashMap` in a single SQL query, then does in-memory lookups.

### Implementation

**File: `crates/sift-store/src/metadata.rs`**

Add a new batch method:

```rust
use std::collections::HashMap;

impl MetadataStore {
    /// Load all known (uri, content_hash) pairs into memory for fast batch lookups.
    /// Returns a map from URI to stored content hash.
    pub fn load_all_hashes(&self) -> SiftResult<HashMap<String, Vec<u8>>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| sift_core::SiftError::Storage(format!("Lock error: {}", e)))?;

        let mut stmt = conn
            .prepare("SELECT uri, content_hash FROM sources")
            .map_err(|e| sift_core::SiftError::Storage(format!("Query error: {}", e)))?;

        let map = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
            })
            .map_err(|e| sift_core::SiftError::Storage(format!("Query error: {}", e)))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(map)
    }
}
```

**File: `crates/sift-cli/src/pipeline.rs`**

Replace the per-item loop with a batch lookup:

```rust
// Phase 2: Filter unchanged files (batch)
let known_hashes = metadata.load_all_hashes()?;
let mut to_process: Vec<SourceItem> = Vec::new();

for item in items {
    match known_hashes.get(&item.uri) {
        Some(stored_hash) if stored_hash.as_slice() == item.content_hash.as_slice() => {
            stats.skipped += 1;
            debug!("Unchanged, skipping: {}", item.uri);
        }
        _ => {
            to_process.push(item);
        }
    }
}
```

### Memory Considerations

For 100K files, each entry is ~100 bytes (URI) + 32 bytes (hash) = ~132 bytes, totaling ~13MB. This is well within acceptable memory usage for a CLI tool.

For very large indexes (1M+ files), add an optional path-limited query:

```rust
/// Load hashes only for URIs that start with any of the given prefixes.
pub fn load_hashes_for_paths(&self, prefixes: &[String]) -> SiftResult<HashMap<String, Vec<u8>>> {
    let conn = self.conn.lock()
        .map_err(|e| sift_core::SiftError::Storage(format!("Lock error: {}", e)))?;

    let mut map = HashMap::new();
    for prefix in prefixes {
        let uri_prefix = format!("file://{}", prefix);
        let mut stmt = conn
            .prepare("SELECT uri, content_hash FROM sources WHERE uri LIKE ?1")
            .map_err(|e| sift_core::SiftError::Storage(format!("Query error: {}", e)))?;

        let rows = stmt
            .query_map(rusqlite::params![format!("{}%", uri_prefix)], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
            })
            .map_err(|e| sift_core::SiftError::Storage(format!("Query error: {}", e)))?;

        for row in rows.filter_map(|r| r.ok()) {
            map.insert(row.0, row.1);
        }
    }
    Ok(map)
}
```

## Tests

**File: `crates/sift-store/src/metadata.rs`**

```rust
#[test]
fn test_load_all_hashes_empty() {
    let store = MetadataStore::open_in_memory().unwrap();
    let hashes = store.load_all_hashes().unwrap();
    assert!(hashes.is_empty());
}

#[test]
fn test_load_all_hashes_returns_all() {
    let store = MetadataStore::open_in_memory().unwrap();
    let hash_a = [1u8; 32];
    let hash_b = [2u8; 32];

    store.upsert_source("file:///a.txt", &hash_a, 10, "txt", None, 1).unwrap();
    store.upsert_source("file:///b.rs", &hash_b, 20, "rs", None, 3).unwrap();

    let hashes = store.load_all_hashes().unwrap();
    assert_eq!(hashes.len(), 2);
    assert_eq!(hashes["file:///a.txt"].as_slice(), hash_a.as_slice());
    assert_eq!(hashes["file:///b.rs"].as_slice(), hash_b.as_slice());
}

#[test]
fn test_load_all_hashes_after_update() {
    let store = MetadataStore::open_in_memory().unwrap();
    let hash_v1 = [1u8; 32];
    let hash_v2 = [2u8; 32];

    store.upsert_source("file:///a.txt", &hash_v1, 10, "txt", None, 1).unwrap();
    store.upsert_source("file:///a.txt", &hash_v2, 10, "txt", None, 1).unwrap();

    let hashes = store.load_all_hashes().unwrap();
    assert_eq!(hashes.len(), 1);
    assert_eq!(hashes["file:///a.txt"].as_slice(), hash_v2.as_slice());
}

#[test]
fn test_batch_filter_matches_individual() {
    let store = MetadataStore::open_in_memory().unwrap();
    let hash_a = [1u8; 32];
    let hash_b = [2u8; 32];

    store.upsert_source("file:///a.txt", &hash_a, 10, "txt", None, 1).unwrap();
    store.upsert_source("file:///b.txt", &hash_b, 20, "txt", None, 2).unwrap();

    // Individual checks
    assert_eq!(store.check_source("file:///a.txt", &hash_a).unwrap(), Some(true));
    assert_eq!(store.check_source("file:///a.txt", &hash_b).unwrap(), Some(false));
    assert!(store.check_source("file:///c.txt", &hash_a).unwrap().is_none());

    // Batch check should give same results
    let hashes = store.load_all_hashes().unwrap();
    assert!(hashes.get("file:///a.txt").map_or(false, |h| h.as_slice() == hash_a.as_slice()));
    assert!(hashes.get("file:///a.txt").map_or(true, |h| h.as_slice() != hash_b.as_slice()));
    assert!(hashes.get("file:///c.txt").is_none());
}
```

## Evaluation Metric

**Benchmark: Filter phase with 10,000 and 50,000 indexed files**

```bash
# Measure time of the "Filter unchanged files" phase
# Run with RUST_LOG=info to see phase timings
RUST_LOG=info vx scan /path/to/already-indexed-corpus 2>&1 | grep "files to process"

# Benchmark comparison
hyperfine --warmup 3 \
  'sift-before scan /path/to/corpus' \
  'sift-after scan /path/to/corpus'
```

Expected improvement: **5-10x faster** filter phase (single SQL query + HashMap lookups vs N individual queries).
