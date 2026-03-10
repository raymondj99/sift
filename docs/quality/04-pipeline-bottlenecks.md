# 04: Pipeline Structural Bottlenecks

**Category**: Performance
**Severity**: High
**Effort**: Medium
**Crates**: sift-cli, sift-store

## Problem

### 1. Embedding stage is single-threaded

```
[Discover+Filter] → chan(64) → [Parse+Chunk (rayon)] → chan(128) → [Embed (single thread)] → chan(128) → [Store]
```

Stage 3 (Embed) runs on one thread (`pipeline.rs:405-443`). It processes files sequentially, calling `embed_text_chunks_atomic` or `embed_image_chunks` one file at a time. The internal batching (chunks of 32) helps ONNX throughput, but there's no cross-file parallelism.

For CPU-based ONNX inference, this is the bottleneck. The parse stage can produce chunks faster than the embed stage can process them, leading to the parse→embed channel filling up and backpressuring.

**Fix**: Use a rayon thread pool or spawn multiple embed workers. The `OnnxEmbedder` is `Send + Sync` (wraps `Arc<Session>`), so it can be shared:

```rust
// Instead of a single thread draining parse_rx:
s.spawn(|| {
    parse_rx.into_iter().par_bridge().for_each(|parsed| {
        let embedded = embed_text_chunks_atomic(&parsed.chunks, embedder, &cache, &cache_hits);
        let _ = embed_tx.send(StoreItem { item: parsed.item, embedded, file_type: parsed.file_type });
    });
    drop(embed_tx);
});
```

### 2. BM25 store writes to disk on every insert

```rust
// bm25.rs:310-311
fn insert(&self, chunks: &[EmbeddedChunk]) -> SiftResult<()> {
    // ... add to in-memory index ...
    drop(inner);
    self.save()  // <-- serializes and writes ENTIRE index to disk
}
```

During a scan, `HybridSearchEngine::insert` calls `BM25Store::insert` for every file. Each call serializes the entire BM25 index (inverted index + document metadata) to JSON and writes it to disk. For 1,000 files with 10 chunks each, that's 1,000 full serializations of a growing index.

**Fix**: Only save on explicit flush. Add a `flush()` method and call it at the end of the scan pipeline, or after every N inserts:

```rust
impl FullTextStore for Bm25Store {
    fn insert(&self, chunks: &[EmbeddedChunk]) -> SiftResult<()> {
        let mut inner = self.inner.lock().map_err(...)?;
        for chunk in chunks { inner.add_doc(...); }
        Ok(())  // don't save
    }
}

impl Bm25Store {
    pub fn flush(&self) -> SiftResult<()> { self.save() }
}
```

### 3. Flat vector search uses full sort instead of partial sort

```rust
// flat.rs:363-372
let mut scored: Vec<(f32, &StoredEntry)> = entries
    .iter()
    .map(|entry| (cosine_similarity(query_vector, &entry.vector), entry))
    .collect();

scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
```

For 100,000 entries with top_k=10, this does O(n log n) sorting when O(n log k) with a min-heap would suffice. The `collect()` also allocates a full-size vec.

**Fix**: Use `select_nth_unstable_by` for partial sort, or use a `BinaryHeap` with bounded capacity:

```rust
use std::collections::BinaryHeap;
use std::cmp::Reverse;

let mut heap: BinaryHeap<Reverse<(OrderedFloat<f32>, usize)>> = BinaryHeap::with_capacity(top_k + 1);

for (i, entry) in entries.iter().enumerate() {
    let score = cosine_similarity(query_vector, &entry.vector);
    heap.push(Reverse((OrderedFloat(score), i)));
    if heap.len() > top_k {
        heap.pop();
    }
}
```

### 4. File content read twice: once in discover, once in parse

The pipeline reads each file twice:

1. **Discover** (`filesystem.rs:205`): `read_and_analyze` reads entire file for hash + MIME
2. **Parse** (`pipeline.rs:339`): `std::fs::read(&item.path)` reads entire file again for parsing

For a 10MB file, that's 20MB of I/O. For the common case where the file hasn't changed (skipped by hash check), the first read is all that's needed. But for changed/new files, both reads happen.

**Fix** (two options):

**Option A — Carry bytes through the pipeline**: `SourceItem` gains a `content: Vec<u8>` field. The discover stage passes the bytes it already read. The parse stage uses them directly. Tradeoff: increases channel message size and memory pressure.

**Option B — Deferred hash**: Don't hash during discovery. Instead:
1. Discover: only stat + MIME-from-extension (zero reads)
2. Filter by extension/size/modified_at (cheap)
3. Read + hash + parse in one pass in the rayon stage

Option B is better — it also avoids reading files that get filtered by extension or size.

### 5. `load_all_hashes` loads entire hash table into memory

```rust
// pipeline.rs:236
let known_hashes = metadata.load_all_hashes()?;
```

For 100,000 indexed files, this loads 100,000 `(String, Vec<u8>)` pairs (~4MB+). This is already much better than the original N-query approach (see your perf doc 04), but for very large indexes it could use a bloom filter for the common case (file unchanged → skip) with SQLite fallback for the rare case (file changed → re-index).

## Pipeline Architecture: Before vs After

### Current
```
[Discover: stat + read_all + hash + MIME] → [Filter: batch hash lookup] → [Parse+Chunk: read_all again] → [Embed: single-thread] → [Store]
```

### Proposed
```
[Discover: stat + MIME_from_ext] → [Filter: extension/size/mtime] → [Parse+Chunk+Hash: single read, rayon] → [Embed: rayon par_bridge] → [Store]
```

The proposed pipeline:
- Eliminates one full file read per changed file
- Eliminates all file reads for unchanged files (use mtime as first-pass filter, hash only if mtime changed)
- Parallelizes the embed stage
- Removes per-file BM25 disk writes

## Impact

On a 10,000 file codebase with 20% changed files:
- **Current**: 10,000 file reads (discover) + 2,000 file reads (parse) = 12,000 reads
- **Proposed**: 2,000 file reads (parse+hash) = 2,000 reads (6x reduction)
- **Embed parallelism**: Near-linear speedup with thread count for CPU inference
