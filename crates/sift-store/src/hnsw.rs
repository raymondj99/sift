use crate::flat::{ExportEntry, FlatVectorIndex};
use crate::traits::{VectorIndex, VectorStore};
use sift_core::{ContentType, EmbeddedChunk, SearchResult, SiftError, SiftResult};
use std::collections::HashMap;
use std::sync::Mutex;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

/// HNSW index file name within the index directory.
const HNSW_INDEX_FILE: &str = "vectors.usearch";
/// Sidecar JSON file mapping u64 labels to metadata.
const HNSW_META_FILE: &str = "vectors.usearch.meta.json";
/// Flat binary file for migration detection.
const FLAT_BIN_FILE: &str = "vectors.bin";

/// Default HNSW graph connectivity (number of bi-directional links per node).
const DEFAULT_M: usize = 16;
/// Default expansion factor during index construction.
const DEFAULT_EF_CONSTRUCTION: usize = 128;
/// Default expansion factor during search.
const DEFAULT_EF_SEARCH: usize = 64;

/// Metadata associated with each stored vector, keyed by u64 label.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct EntryMeta {
    uri: String,
    text: String,
    chunk_index: u32,
    content_type: ContentType,
    file_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    byte_range: Option<(u64, u64)>,
}

/// Inner mutable state protected by a mutex.
struct HnswInner {
    /// USearch HNSW index.
    index: Index,
    /// Monotonically increasing label counter.
    next_label: u64,
    /// Label -> metadata mapping.
    meta: HashMap<u64, EntryMeta>,
    /// Dimensionality (set on first insert, 0 until then).
    dimensions: usize,
}

/// Approximate nearest-neighbour vector index backed by USearch HNSW.
///
/// Provides O(log n) search instead of the O(n) brute-force scan of
/// [`FlatVectorIndex`]. Feature-gated behind `hnsw`.
pub struct HnswIndex {
    inner: Mutex<HnswInner>,
}

impl HnswIndex {
    /// Create a fresh, empty index. Dimensionality is determined on the first
    /// insert.
    pub fn new() -> Self {
        let options = Self::default_options(0);
        let index = Index::new(&options).expect("usearch: failed to create empty index");
        Self {
            inner: Mutex::new(HnswInner {
                index,
                next_label: 0,
                meta: HashMap::new(),
                dimensions: 0,
            }),
        }
    }

    /// Build default `IndexOptions` for the given dimensionality.
    fn default_options(dimensions: usize) -> IndexOptions {
        IndexOptions {
            dimensions,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            connectivity: DEFAULT_M,
            expansion_add: DEFAULT_EF_CONSTRUCTION,
            expansion_search: DEFAULT_EF_SEARCH,
            multi: false,
        }
    }

    /// Create a new index with known dimensionality.
    fn new_with_dimensions(dimensions: usize) -> SiftResult<Self> {
        let options = Self::default_options(dimensions);
        let index = Index::new(&options)
            .map_err(|e| SiftError::Storage(format!("usearch: create index: {e}")))?;
        Ok(Self {
            inner: Mutex::new(HnswInner {
                index,
                next_label: 0,
                meta: HashMap::new(),
                dimensions,
            }),
        })
    }

    /// Backwards-compatible load: treats `path` as a directory (or its parent
    /// if it points to a file) and delegates to [`load_or_create`].
    pub fn load(path: &std::path::Path) -> SiftResult<Self> {
        let dir = if path.is_dir() {
            path.to_path_buf()
        } else {
            path.parent().unwrap_or(path).to_path_buf()
        };
        Self::load_or_create(&dir)
    }

    /// Load an existing HNSW index from `index_dir`, migrate from flat if
    /// necessary, or create a new empty index.
    pub fn load_or_create(index_dir: &std::path::Path) -> SiftResult<Self> {
        let hnsw_path = index_dir.join(HNSW_INDEX_FILE);
        let meta_path = index_dir.join(HNSW_META_FILE);

        // 1. Try loading existing HNSW index.
        if hnsw_path.exists() && meta_path.exists() {
            return Self::load_from(index_dir);
        }

        // 2. Try migrating from FlatVectorIndex.
        let flat_path = index_dir.join(FLAT_BIN_FILE);
        if flat_path.exists() {
            tracing::info!("Migrating flat vector index -> HNSW");
            let flat = FlatVectorIndex::load(&flat_path)?;
            let hnsw = Self::migrate_from_flat(&flat)?;
            hnsw.save_to(index_dir)?;
            return Ok(hnsw);
        }

        // 3. Also try JSON migration path.
        let json_path = index_dir.join("vectors.json");
        if json_path.exists() {
            tracing::info!("Migrating vectors.json -> HNSW");
            let flat = FlatVectorIndex::load_json(&json_path)?;
            let hnsw = Self::migrate_from_flat(&flat)?;
            hnsw.save_to(index_dir)?;
            let bak_path = index_dir.join("vectors.json.bak");
            std::fs::rename(&json_path, &bak_path)?;
            return Ok(hnsw);
        }

        // 4. Empty index.
        Ok(Self::new())
    }

    /// Load HNSW index + sidecar metadata from `index_dir`.
    fn load_from(index_dir: &std::path::Path) -> SiftResult<Self> {
        let hnsw_path = index_dir.join(HNSW_INDEX_FILE);
        let meta_path = index_dir.join(HNSW_META_FILE);

        // Load sidecar metadata first to know dimensions.
        let meta_bytes = std::fs::read(&meta_path)?;
        let meta: HashMap<u64, EntryMeta> = serde_json::from_slice(&meta_bytes)
            .map_err(|e| SiftError::Storage(format!("usearch: read meta: {e}")))?;

        let next_label = meta.keys().copied().max().map_or(0, |k| k + 1);

        // Infer dimensions from any entry's vector (we will get it from the loaded index).
        // Create options with 0 dimensions; load will override.
        let options = Self::default_options(0);
        let index = Index::new(&options)
            .map_err(|e| SiftError::Storage(format!("usearch: create for load: {e}")))?;

        let hnsw_path_str = hnsw_path
            .to_str()
            .ok_or_else(|| SiftError::Storage("non-UTF-8 path".to_string()))?;
        index
            .load(hnsw_path_str)
            .map_err(|e| SiftError::Storage(format!("usearch: load: {e}")))?;

        let dimensions = index.dimensions();

        Ok(Self {
            inner: Mutex::new(HnswInner {
                index,
                next_label,
                meta,
                dimensions,
            }),
        })
    }

    /// Save HNSW index + sidecar metadata to `index_dir`.
    fn save_to(&self, index_dir: &std::path::Path) -> SiftResult<()> {
        let inner = self.inner.lock().map_err(lock_err)?;

        let hnsw_path = index_dir.join(HNSW_INDEX_FILE);
        let meta_path = index_dir.join(HNSW_META_FILE);

        let hnsw_path_str = hnsw_path
            .to_str()
            .ok_or_else(|| SiftError::Storage("non-UTF-8 path".to_string()))?;
        inner
            .index
            .save(hnsw_path_str)
            .map_err(|e| SiftError::Storage(format!("usearch: save: {e}")))?;

        let meta_bytes = serde_json::to_vec(&inner.meta)
            .map_err(|e| SiftError::Storage(format!("usearch: serialize meta: {e}")))?;
        sift_core::atomic_write(&meta_path, &meta_bytes)?;

        Ok(())
    }

    /// Build an `HnswIndex` from all entries in a `FlatVectorIndex`.
    pub fn migrate_from_flat(flat: &FlatVectorIndex) -> SiftResult<Self> {
        let entries = flat.export_all()?;
        if entries.is_empty() {
            return Ok(Self::new());
        }

        let dim = entries[0].vector.len();
        let hnsw = Self::new_with_dimensions(dim)?;

        // Convert entries to EmbeddedChunks for insert.
        let chunks: Vec<EmbeddedChunk> = entries
            .into_iter()
            .map(|e| EmbeddedChunk {
                chunk: sift_core::Chunk {
                    text: e.text,
                    source_uri: e.uri,
                    chunk_index: e.chunk_index,
                    content_type: e.content_type,
                    file_type: e.file_type,
                    title: e.title,
                    language: None,
                    byte_range: e.byte_range,
                },
                vector: e.vector,
            })
            .collect();

        hnsw.insert(&chunks)?;
        Ok(hnsw)
    }

    /// Ensure the inner index is initialized with the correct dimensions.
    /// Must be called with the lock held. Returns an error if dimensions
    /// mismatch.
    fn ensure_dimensions(inner: &mut HnswInner, dim: usize) -> SiftResult<()> {
        if inner.dimensions == 0 {
            // First insert - reinitialize with correct dimensions.
            let options = Self::default_options(dim);
            let index = Index::new(&options)
                .map_err(|e| SiftError::Storage(format!("usearch: create: {e}")))?;
            inner.index = index;
            inner.dimensions = dim;
        } else if inner.dimensions != dim {
            return Err(SiftError::Storage(format!(
                "dimension mismatch: index has {}, got {}",
                inner.dimensions, dim
            )));
        }
        Ok(())
    }
}

impl Default for HnswIndex {
    fn default() -> Self {
        Self::new()
    }
}

// -- Trait implementations --

impl VectorStore for HnswIndex {
    fn insert(&self, chunks: &[EmbeddedChunk]) -> SiftResult<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        let mut inner = self.inner.lock().map_err(lock_err)?;

        let dim = chunks[0].vector.len();
        Self::ensure_dimensions(&mut inner, dim)?;

        // Reserve capacity for new entries.
        let new_total = inner.index.size() + chunks.len();
        inner
            .index
            .reserve(new_total)
            .map_err(|e| SiftError::Storage(format!("usearch: reserve: {e}")))?;

        for ec in chunks {
            let label = inner.next_label;
            inner.next_label += 1;

            inner
                .index
                .add(label, &ec.vector)
                .map_err(|e| SiftError::Storage(format!("usearch: add: {e}")))?;

            inner.meta.insert(
                label,
                EntryMeta {
                    uri: ec.chunk.source_uri.clone(),
                    text: ec.chunk.text.clone(),
                    chunk_index: ec.chunk.chunk_index,
                    content_type: ec.chunk.content_type,
                    file_type: ec.chunk.file_type.clone(),
                    title: ec.chunk.title.clone(),
                    byte_range: ec.chunk.byte_range,
                },
            );
        }

        Ok(())
    }

    fn search(&self, query_vector: &[f32], top_k: usize) -> SiftResult<Vec<SearchResult>> {
        let inner = self.inner.lock().map_err(lock_err)?;

        if inner.meta.is_empty() || inner.dimensions == 0 {
            return Ok(Vec::new());
        }

        let results = inner
            .index
            .search(query_vector, top_k)
            .map_err(|e| SiftError::Storage(format!("usearch: search: {e}")))?;

        let mut out = Vec::with_capacity(results.keys.len());
        for (key, distance) in results.keys.iter().zip(results.distances.iter()) {
            if let Some(meta) = inner.meta.get(key) {
                // USearch Cosine distance = 1 - cosine_similarity.
                // Convert to similarity score for compatibility with FlatVectorIndex.
                let score = 1.0 - distance;
                out.push(SearchResult {
                    uri: meta.uri.clone(),
                    text: meta.text.clone(),
                    score,
                    chunk_index: meta.chunk_index,
                    content_type: meta.content_type,
                    file_type: meta.file_type.clone(),
                    title: meta.title.clone(),
                    byte_range: meta.byte_range,
                });
            }
        }

        Ok(out)
    }

    fn delete_by_uri(&self, uri: &str) -> SiftResult<u64> {
        let mut inner = self.inner.lock().map_err(lock_err)?;

        let labels_to_remove: Vec<u64> = inner
            .meta
            .iter()
            .filter(|(_, m)| m.uri == uri)
            .map(|(&label, _)| label)
            .collect();

        let count = labels_to_remove.len() as u64;

        for label in &labels_to_remove {
            let _ = inner.index.remove(*label);
            inner.meta.remove(label);
        }

        Ok(count)
    }

    fn count(&self) -> SiftResult<u64> {
        let inner = self.inner.lock().map_err(lock_err)?;
        Ok(inner.meta.len() as u64)
    }
}

impl VectorIndex for HnswIndex {
    fn save(&self, path: &std::path::Path) -> SiftResult<()> {
        // `path` points to a file like `vectors.bin`. We use its parent
        // directory as the index directory for our multi-file format.
        let index_dir = path
            .parent()
            .ok_or_else(|| SiftError::Storage("save path has no parent".to_string()))?;
        self.save_to(index_dir)
    }

    fn export_all(&self) -> SiftResult<Vec<ExportEntry>> {
        let inner = self.inner.lock().map_err(lock_err)?;

        let dim = inner.dimensions;
        let mut entries = Vec::with_capacity(inner.meta.len());

        for (&label, meta) in &inner.meta {
            // Extract the vector from the index.
            let mut vector = vec![0.0f32; dim];
            let _ = inner.index.get(label, &mut vector);

            entries.push(ExportEntry {
                uri: meta.uri.clone(),
                text: meta.text.clone(),
                chunk_index: meta.chunk_index,
                content_type: meta.content_type,
                file_type: meta.file_type.clone(),
                title: meta.title.clone(),
                byte_range: meta.byte_range,
                vector,
            });
        }

        Ok(entries)
    }
}

fn lock_err<T: std::fmt::Display>(e: T) -> SiftError {
    SiftError::Storage(format!("Lock error: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sift_core::Chunk;

    fn make_chunk(uri: &str, text: &str, idx: u32) -> EmbeddedChunk {
        EmbeddedChunk {
            chunk: Chunk {
                text: text.to_string(),
                source_uri: uri.to_string(),
                chunk_index: idx,
                content_type: ContentType::Text,
                file_type: "txt".to_string(),
                title: None,
                language: None,
                byte_range: None,
            },
            vector: vec![0.0; 3], // placeholder, will be overridden in tests
        }
    }

    fn make_chunk_with_vec(uri: &str, text: &str, idx: u32, vec: Vec<f32>) -> EmbeddedChunk {
        let mut c = make_chunk(uri, text, idx);
        c.vector = vec;
        c
    }

    #[test]
    fn test_insert_and_count() {
        let index = HnswIndex::new();
        let c = make_chunk_with_vec("file:///a.txt", "hello", 0, vec![1.0, 0.0, 0.0]);
        index.insert(&[c]).unwrap();
        assert_eq!(index.count().unwrap(), 1);
    }

    #[test]
    fn test_insert_multiple_and_search() {
        let index = HnswIndex::new();

        let c1 = make_chunk_with_vec("file:///a.txt", "hello", 0, vec![1.0, 0.0, 0.0]);
        let c2 = make_chunk_with_vec("file:///b.txt", "world", 0, vec![0.0, 1.0, 0.0]);
        index.insert(&[c1, c2]).unwrap();

        let results = index.search(&[1.0, 0.0, 0.0], 10).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].uri, "file:///a.txt");
        // Cosine similarity of [1,0,0] with itself should be ~1.0
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn test_delete_by_uri() {
        let index = HnswIndex::new();
        index
            .insert(&[
                make_chunk_with_vec("file:///a.txt", "hello", 0, vec![1.0, 0.0, 0.0]),
                make_chunk_with_vec("file:///a.txt", "world", 1, vec![0.0, 1.0, 0.0]),
                make_chunk_with_vec("file:///b.txt", "foo", 0, vec![0.0, 0.0, 1.0]),
            ])
            .unwrap();

        assert_eq!(index.count().unwrap(), 3);
        let deleted = index.delete_by_uri("file:///a.txt").unwrap();
        assert_eq!(deleted, 2);
        assert_eq!(index.count().unwrap(), 1);
    }

    #[test]
    fn test_empty_index_search() {
        let index = HnswIndex::new();
        let results = index.search(&[1.0, 0.0, 0.0], 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_persistence_roundtrip() {
        let dir = tempfile::TempDir::new().unwrap();

        let index = HnswIndex::new();
        index
            .insert(&[
                make_chunk_with_vec("file:///a.txt", "hello world", 0, vec![1.0, 2.0, 3.0]),
                make_chunk_with_vec("file:///b.txt", "foo bar", 0, vec![4.0, 5.0, 6.0]),
            ])
            .unwrap();

        // Save using VectorIndex::save (takes a file path, uses parent dir)
        let save_path = dir.path().join("vectors.bin");
        VectorIndex::save(&index, &save_path).unwrap();

        // Verify files exist
        assert!(dir.path().join(HNSW_INDEX_FILE).exists());
        assert!(dir.path().join(HNSW_META_FILE).exists());

        // Load
        let loaded = HnswIndex::load_or_create(dir.path()).unwrap();
        assert_eq!(loaded.count().unwrap(), 2);

        let results = loaded.search(&[1.0, 2.0, 3.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].uri, "file:///a.txt");
        assert_eq!(results[0].text, "hello world");
    }

    #[test]
    fn test_migration_from_flat() {
        let dir = tempfile::TempDir::new().unwrap();

        // Create a flat index and save it.
        let flat = FlatVectorIndex::new();
        flat.insert(&[
            make_chunk_with_vec("file:///a.txt", "migrated content", 0, vec![1.0, 2.0, 3.0]),
            make_chunk_with_vec("file:///b.txt", "also migrated", 0, vec![4.0, 5.0, 6.0]),
        ])
        .unwrap();
        let flat_path = dir.path().join(FLAT_BIN_FILE);
        flat.save(&flat_path).unwrap();

        // load_or_create should detect flat index and migrate.
        let hnsw = HnswIndex::load_or_create(dir.path()).unwrap();
        assert_eq!(hnsw.count().unwrap(), 2);

        let results = hnsw.search(&[1.0, 2.0, 3.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].uri, "file:///a.txt");
        assert_eq!(results[0].text, "migrated content");

        // HNSW files should now exist.
        assert!(dir.path().join(HNSW_INDEX_FILE).exists());
        assert!(dir.path().join(HNSW_META_FILE).exists());
    }

    #[test]
    fn test_export_all() {
        let index = HnswIndex::new();
        let chunk = EmbeddedChunk {
            chunk: Chunk {
                text: "test content".to_string(),
                source_uri: "file:///test.rs".to_string(),
                chunk_index: 5,
                content_type: ContentType::Code,
                file_type: "rs".to_string(),
                title: Some("My Module".to_string()),
                language: Some("rust".to_string()),
                byte_range: Some((100, 200)),
            },
            vector: vec![0.1, 0.2, 0.3],
        };
        index.insert(&[chunk]).unwrap();

        let entries = index.export_all().unwrap();
        assert_eq!(entries.len(), 1);

        let entry = &entries[0];
        assert_eq!(entry.uri, "file:///test.rs");
        assert_eq!(entry.text, "test content");
        assert_eq!(entry.chunk_index, 5);
        assert_eq!(entry.content_type, ContentType::Code);
        assert_eq!(entry.file_type, "rs");
        assert_eq!(entry.title.as_deref(), Some("My Module"));
        assert_eq!(entry.byte_range, Some((100, 200)));
    }

    #[test]
    fn test_load_or_create_empty_dir() {
        let dir = tempfile::TempDir::new().unwrap();
        let index = HnswIndex::load_or_create(dir.path()).unwrap();
        assert_eq!(index.count().unwrap(), 0);
    }

    #[test]
    fn test_batch_insert() {
        let index = HnswIndex::new();
        let chunks: Vec<EmbeddedChunk> = (0..50)
            .map(|i| {
                let mut v = vec![0.0f32; 3];
                v[i % 3] = 1.0;
                make_chunk_with_vec(
                    &format!("file:///doc{}.txt", i),
                    &format!("chunk {}", i),
                    0,
                    v,
                )
            })
            .collect();

        index.insert(&chunks).unwrap();
        assert_eq!(index.count().unwrap(), 50);

        let results = index.search(&[1.0, 0.0, 0.0], 5).unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_key_mapping_persistence() {
        let dir = tempfile::TempDir::new().unwrap();

        let index = HnswIndex::new();
        index
            .insert(&[
                make_chunk_with_vec("file:///a.txt", "hello", 0, vec![1.0, 0.0, 0.0]),
                make_chunk_with_vec("file:///b.txt", "world", 0, vec![0.0, 1.0, 0.0]),
            ])
            .unwrap();

        // Save to disk.
        let save_path = dir.path().join("vectors.bin");
        VectorIndex::save(&index, &save_path).unwrap();

        // Verify sidecar file exists and contains the right data.
        let meta_path = dir.path().join(HNSW_META_FILE);
        assert!(meta_path.exists());

        let data = std::fs::read(&meta_path).unwrap();
        let meta: HashMap<u64, EntryMeta> = serde_json::from_slice(&data).unwrap();
        assert_eq!(meta.len(), 2);

        // All URIs should be present in the metadata.
        let uris: Vec<&str> = meta.values().map(|m| m.uri.as_str()).collect();
        assert!(uris.contains(&"file:///a.txt"));
        assert!(uris.contains(&"file:///b.txt"));

        // Reload and verify the key mappings survived.
        let loaded = HnswIndex::load_or_create(dir.path()).unwrap();
        assert_eq!(loaded.count().unwrap(), 2);

        let results = loaded.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].uri, "file:///a.txt");
    }

    #[test]
    fn test_default_impl() {
        let index = HnswIndex::default();
        assert_eq!(index.count().unwrap(), 0);
    }

    #[test]
    fn test_insert_empty_chunks() {
        let index = HnswIndex::new();
        // Insert empty slice should be a no-op
        index.insert(&[]).unwrap();
        assert_eq!(index.count().unwrap(), 0);
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let index = HnswIndex::new();
        // First insert with 3 dimensions
        let c1 = make_chunk_with_vec("file:///a.txt", "hello", 0, vec![1.0, 0.0, 0.0]);
        index.insert(&[c1]).unwrap();

        // Second insert with 4 dimensions should fail
        let c2 = make_chunk_with_vec("file:///b.txt", "world", 0, vec![1.0, 0.0, 0.0, 0.0]);
        let err = index.insert(&[c2]);
        assert!(err.is_err());
        let msg = err.unwrap_err().to_string();
        assert!(msg.contains("dimension mismatch"), "got: {msg}");
    }

    #[test]
    fn test_export_all_empty_index() {
        let index = HnswIndex::new();
        let entries = index.export_all().unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_load_with_file_path() {
        // Test the `load` method that treats a file path by using its parent dir
        let dir = tempfile::TempDir::new().unwrap();

        let index = HnswIndex::new();
        index
            .insert(&[make_chunk_with_vec(
                "file:///a.txt",
                "hello",
                0,
                vec![1.0, 0.0, 0.0],
            )])
            .unwrap();
        index.save_to(dir.path()).unwrap();

        // Pass a file path (not directory) to load()
        let file_path = dir.path().join("some_file.bin");
        let loaded = HnswIndex::load(&file_path).unwrap();
        assert_eq!(loaded.count().unwrap(), 1);
    }

    #[test]
    fn test_load_with_directory_path() {
        // Test the `load` method with a directory path
        let dir = tempfile::TempDir::new().unwrap();

        let index = HnswIndex::new();
        index
            .insert(&[make_chunk_with_vec(
                "file:///a.txt",
                "hello",
                0,
                vec![1.0, 0.0, 0.0],
            )])
            .unwrap();
        index.save_to(dir.path()).unwrap();

        // Pass the directory itself to load()
        let loaded = HnswIndex::load(dir.path()).unwrap();
        assert_eq!(loaded.count().unwrap(), 1);
    }

    #[test]
    fn test_migration_from_json() {
        let dir = tempfile::TempDir::new().unwrap();

        // Create a legacy JSON flat file
        let flat = FlatVectorIndex::new();
        flat.insert(&[make_chunk_with_vec(
            "file:///a.txt",
            "json migrated",
            0,
            vec![1.0, 2.0, 3.0],
        )])
        .unwrap();
        let json_path = dir.path().join("vectors.json");
        flat.save_json(&json_path).unwrap();

        // load_or_create should detect JSON and migrate
        let hnsw = HnswIndex::load_or_create(dir.path()).unwrap();
        assert_eq!(hnsw.count().unwrap(), 1);

        // HNSW files should now exist
        assert!(dir.path().join(HNSW_INDEX_FILE).exists());
        assert!(dir.path().join(HNSW_META_FILE).exists());

        // JSON should be renamed to .bak
        assert!(!dir.path().join("vectors.json").exists());
        assert!(dir.path().join("vectors.json.bak").exists());
    }

    #[test]
    fn test_migrate_from_empty_flat() {
        let flat = FlatVectorIndex::new();
        let hnsw = HnswIndex::migrate_from_flat(&flat).unwrap();
        assert_eq!(hnsw.count().unwrap(), 0);
    }

    #[test]
    fn test_delete_nonexistent_uri() {
        let index = HnswIndex::new();
        index
            .insert(&[make_chunk_with_vec(
                "file:///a.txt",
                "hello",
                0,
                vec![1.0, 0.0, 0.0],
            )])
            .unwrap();

        let deleted = index.delete_by_uri("file:///nonexistent.txt").unwrap();
        assert_eq!(deleted, 0);
        assert_eq!(index.count().unwrap(), 1);
    }

    #[test]
    fn test_delete_then_search() {
        let index = HnswIndex::new();
        index
            .insert(&[
                make_chunk_with_vec("file:///a.txt", "hello", 0, vec![1.0, 0.0, 0.0]),
                make_chunk_with_vec("file:///b.txt", "world", 0, vec![0.0, 1.0, 0.0]),
            ])
            .unwrap();

        index.delete_by_uri("file:///a.txt").unwrap();
        assert_eq!(index.count().unwrap(), 1);

        let results = index.search(&[1.0, 0.0, 0.0], 10).unwrap();
        // Only b.txt should remain.
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].uri, "file:///b.txt");
    }
}
