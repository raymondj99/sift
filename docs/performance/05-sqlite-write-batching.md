# 05: SQLite Write Batching

## Problem

In `sift-cli/src/pipeline.rs:448-489`, the store stage inserts chunks and updates metadata **one file at a time** without explicit transaction boundaries:

```rust
for store_item in embed_rx {
    let _ = engine.delete_by_uri(&store_item.item.uri);
    engine.insert(&store_item.embedded)?;
    metadata.upsert_source(...)?;
}
```

Each `insert()` and `upsert_source()` call implicitly creates its own SQLite transaction (auto-commit mode). This means every file triggers multiple `fsync` operations, which is extremely slow on rotational storage and still significant on SSDs.

SQLite best practice: batch many writes into a single transaction. This can yield **10-100x** improvement for bulk inserts.

## Solution

Wrap the store stage loop in explicit transactions, committing every N files (batch size).

### Implementation

**File: `crates/sift-store/src/metadata.rs`**

Add transaction management methods:

```rust
impl MetadataStore {
    /// Begin an explicit transaction.
    pub fn begin_transaction(&self) -> SiftResult<()> {
        let conn = self.conn.lock()
            .map_err(|e| sift_core::SiftError::Storage(format!("Lock error: {}", e)))?;
        conn.execute_batch("BEGIN IMMEDIATE")
            .map_err(|e| sift_core::SiftError::Storage(format!("Begin transaction: {}", e)))?;
        Ok(())
    }

    /// Commit the current transaction.
    pub fn commit_transaction(&self) -> SiftResult<()> {
        let conn = self.conn.lock()
            .map_err(|e| sift_core::SiftError::Storage(format!("Lock error: {}", e)))?;
        conn.execute_batch("COMMIT")
            .map_err(|e| sift_core::SiftError::Storage(format!("Commit transaction: {}", e)))?;
        Ok(())
    }
}
```

**File: `crates/sift-store/src/fts5.rs`** (and other FullTextStore impls)

Add similar `begin_transaction()` / `commit_transaction()` methods.

**File: `crates/sift-cli/src/pipeline.rs`**

Wrap the store loop in batched transactions:

```rust
// ---- Stage 4: Store (runs on the scoped main thread) ----
let mut store_error: Option<sift_core::SiftError> = None;
let batch_size = 100; // Commit every 100 files
let mut batch_count = 0u64;

// Begin initial transaction
if let Err(e) = metadata.begin_transaction() {
    store_error = Some(e);
}

for store_item in embed_rx {
    if store_error.is_some() { break; }

    pb.set_message(/* ... */);

    let chunk_count = store_item.embedded.len() as u32;

    // Delete old chunks for this URI if re-indexing
    let _ = engine.delete_by_uri(&store_item.item.uri);

    // Insert into stores
    if let Err(e) = engine.insert(&store_item.embedded) {
        store_error = Some(e);
        break;
    }

    // Update metadata
    if let Err(e) = metadata.upsert_source(
        &store_item.item.uri,
        &store_item.item.content_hash,
        store_item.item.size,
        &store_item.file_type,
        store_item.item.modified_at,
        chunk_count,
    ) {
        store_error = Some(e);
        break;
    }

    atomic_indexed.fetch_add(1, Ordering::Relaxed);
    atomic_chunks.fetch_add(chunk_count as u64, Ordering::Relaxed);
    *file_type_map.entry(store_item.file_type.clone()).or_insert(0) += 1;
    batch_count += 1;

    // Commit batch
    if batch_count % batch_size as u64 == 0 {
        if let Err(e) = metadata.commit_transaction() {
            store_error = Some(e);
            break;
        }
        if let Err(e) = metadata.begin_transaction() {
            store_error = Some(e);
            break;
        }
    }

    pb.inc(1);
}

// Final commit
if store_error.is_none() {
    if let Err(e) = metadata.commit_transaction() {
        store_error = Some(e);
    }
}

pb.finish_and_clear();
```

### Optimal Batch Size

Based on SQLite benchmarks:
- **1 (no batching)**: ~50 inserts/sec (fsync dominated)
- **100**: ~10,000 inserts/sec
- **1000**: ~50,000 inserts/sec
- **10000+**: diminishing returns, higher memory/rollback risk

A batch size of **100** provides the best balance of throughput and memory safety.

### SQLite PRAGMA Tuning

The metadata store already sets `journal_mode=WAL` and `synchronous=NORMAL`. Add one more pragma for bulk insert performance:

```rust
conn.execute_batch(
    "PRAGMA journal_mode=WAL;
     PRAGMA synchronous=NORMAL;
     PRAGMA foreign_keys=ON;
     PRAGMA cache_size=-8000;",  // 8MB cache (default is 2MB)
)
```

## Tests

```rust
#[test]
fn test_transaction_batching() {
    let store = MetadataStore::open_in_memory().unwrap();
    let hash = [0u8; 32];

    store.begin_transaction().unwrap();
    for i in 0..100 {
        store.upsert_source(
            &format!("file:///test_{}.txt", i),
            &hash, 10, "txt", None, 1,
        ).unwrap();
    }
    store.commit_transaction().unwrap();

    let stats = store.stats().unwrap();
    assert_eq!(stats.total_sources, 100);
}

#[test]
fn test_transaction_commit_persists() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("test.db");
    let hash = [0u8; 32];

    {
        let store = MetadataStore::open(&path).unwrap();
        store.begin_transaction().unwrap();
        store.upsert_source("file:///a.txt", &hash, 10, "txt", None, 1).unwrap();
        store.commit_transaction().unwrap();
    }

    // Reopen and verify
    let store = MetadataStore::open(&path).unwrap();
    assert_eq!(store.check_source("file:///a.txt", &hash).unwrap(), Some(true));
}

#[test]
fn test_multiple_transaction_batches() {
    let store = MetadataStore::open_in_memory().unwrap();
    let hash = [0u8; 32];

    // Batch 1
    store.begin_transaction().unwrap();
    for i in 0..50 {
        store.upsert_source(&format!("file:///{}.txt", i), &hash, 10, "txt", None, 1).unwrap();
    }
    store.commit_transaction().unwrap();

    // Batch 2
    store.begin_transaction().unwrap();
    for i in 50..100 {
        store.upsert_source(&format!("file:///{}.txt", i), &hash, 10, "txt", None, 1).unwrap();
    }
    store.commit_transaction().unwrap();

    let stats = store.stats().unwrap();
    assert_eq!(stats.total_sources, 100);
}
```

## Evaluation Metric

**Benchmark: Index 5,000 files measuring store phase time**

```bash
# With tracing enabled to see per-stage timing
RUST_LOG=info hyperfine --warmup 1 \
  'sift scan /path/to/5000-file-corpus'

# Before: expect ~50 files/sec store throughput
# After:  expect ~5000+ files/sec store throughput
```

Expected improvement: **10-50x faster** store phase for bulk indexing. The overall scan pipeline improvement depends on whether store is the bottleneck (it often is for keyword-only mode without embeddings).
