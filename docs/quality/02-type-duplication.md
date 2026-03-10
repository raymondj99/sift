# 02: Type Duplication & Trait Placement

**Category**: Organization
**Severity**: Medium
**Effort**: Low
**Crates**: sift-core, sift-store

## Problem

### 1. The same 7 fields are duplicated across 5+ types

These fields — `uri`, `text`, `chunk_index`, `content_type`, `file_type`, `title`, `byte_range` — appear in:

| Type | Crate | Location |
|------|-------|----------|
| `StoredEntry` | sift-store | `flat.rs:19` |
| `EntryMeta` (flat) | sift-store | `flat.rs:32` |
| `EntryMeta` (hnsw) | sift-store | `hnsw.rs:24` |
| `SerEntry` | sift-store | `flat.rs:57` |
| `DocMeta` | sift-store | `bm25.rs:28` |
| `ExportEntry` | sift-store | `flat.rs:44` |
| `SearchResult` | sift-core | `types.rs:73` |

This means any schema change (e.g., adding a `language` field to search results) requires updating 5-7 structs. ripgrep avoids this by having a single canonical representation that flows through the pipeline.

### 2. `Embedder` trait defined in sift-core

```rust
// sift-core/src/types.rs:183
pub trait Embedder: Send + Sync {
    fn embed_batch(&self, texts: &[&str]) -> SiftResult<Vec<Vec<f32>>>;
    fn embed(&self, text: &str) -> SiftResult<Vec<f32>> { ... }
    fn dimensions(&self) -> usize;
    fn model_name(&self) -> &str;
}
```

This forces `sift-core` (which should be pure data types + config) to know about embedding concepts. The trait is implemented in `sift-embed` and consumed in `sift-cli`. It should live in `sift-embed` (with `sift-core` exporting only the types that the trait operates on).

The only reason it's in `sift-core` is so `pipeline.rs` can accept `Option<&dyn Embedder>` without depending on `sift-embed` at the type level. This can be solved with a local trait bound or by moving the pipeline to accept a generic.

### 3. `ScanStats` vs `IndexStats` overlap

```rust
// sift-core/types.rs
pub struct IndexStats {
    pub total_sources: u64,
    pub total_chunks: u64,
    pub index_size_bytes: u64,
    pub file_type_counts: HashMap<String, u64>,
}

// sift-cli/pipeline.rs
pub struct ScanStats {
    pub discovered: u64,
    pub skipped: u64,
    pub indexed: u64,
    pub chunks: u64,
    pub errors: u64,
    pub cache_hits: u64,
    pub file_types: HashMap<String, u64>,
}
```

`file_type_counts` and `file_types` are the same concept. `total_chunks` and `chunks` are the same concept. These should either be unified or `ScanStats` should embed `IndexStats`.

## Proposed Fix

### Step 1: Create a shared `ChunkMeta` struct in sift-core

```rust
// sift-core/src/types.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMeta {
    pub uri: String,
    pub text: String,
    pub chunk_index: u32,
    pub content_type: ContentType,
    pub file_type: String,
    pub title: Option<String>,
    pub byte_range: Option<(u64, u64)>,
}
```

Then `SearchResult` becomes:

```rust
pub struct SearchResult {
    #[serde(flatten)]
    pub meta: ChunkMeta,
    pub score: f32,
}
```

And `ExportEntry` becomes:

```rust
pub struct ExportEntry {
    pub meta: ChunkMeta,
    pub vector: Vec<f32>,
}
```

Store-internal types (`StoredEntry`, `DocMeta`) reuse `ChunkMeta` + whatever extra fields they need (e.g., `doc_len` for BM25).

### Step 2: Move `Embedder` trait to sift-embed

```rust
// sift-embed/src/traits.rs (already exists, currently re-exports from sift-core)
pub trait Embedder: Send + Sync {
    fn embed_batch(&self, texts: &[&str]) -> SiftResult<Vec<Vec<f32>>>;
    fn embed(&self, text: &str) -> SiftResult<Vec<f32>> { ... }
    fn dimensions(&self) -> usize;
    fn model_name(&self) -> &str;
}
```

The pipeline function signature changes from:
```rust
fn run_scan_pipeline(..., embedder: Option<&dyn Embedder>, ...)
```
to:
```rust
fn run_scan_pipeline<E: Embedder>(..., embedder: Option<&E>, ...)
```

Or keep `dyn Embedder` but define the trait in `sift-embed` and have `sift-cli` depend on it (which it already does when `embeddings` feature is enabled).

### Step 3: Unify stats types

```rust
pub struct ScanStats {
    pub discovered: u64,
    pub skipped: u64,
    pub indexed: u64,
    pub errors: u64,
    pub cache_hits: u64,
    pub index: IndexStats,  // embed the shared stats
}
```

## Impact

Reduces total struct field declarations by ~40 lines. More importantly, makes schema evolution safe — add a field in one place, not seven.
