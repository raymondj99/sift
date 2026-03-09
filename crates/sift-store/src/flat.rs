use crate::traits::{VectorIndex, VectorStore};
use sift_core::{ContentType, EmbeddedChunk, SearchResult, SiftResult};
use std::io::{Read, Write};
use std::sync::Mutex;

/// Magic bytes for the binary vector index format (version 1).
const MAGIC: &[u8; 4] = b"SFT1";

/// In-memory flat vector index using brute-force cosine similarity.
/// Good for indexes up to ~100K chunks.
///
/// Stores entries in a compact binary format on disk:
/// - Header: magic bytes "SFT1" (4), entry count (u64 LE), dimension (u32 LE)
/// - Per entry: key_len (u32 LE) + key bytes + vector (f32 x dim LE) + meta_len (u32 LE) + meta JSON bytes
pub struct FlatVectorIndex {
    entries: Mutex<Vec<StoredEntry>>,
}

struct StoredEntry {
    uri: String,
    text: String,
    vector: Vec<f32>,
    chunk_index: u32,
    content_type: ContentType,
    file_type: String,
    title: Option<String>,
    byte_range: Option<(u64, u64)>,
}

/// Serializable metadata for each entry (written as JSON within the binary format).
#[derive(serde::Serialize, serde::Deserialize)]
struct EntryMeta {
    chunk_index: u32,
    content_type: ContentType,
    file_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    byte_range: Option<(u64, u64)>,
    text: String,
}

/// A single entry for export.
pub struct ExportEntry {
    pub uri: String,
    pub text: String,
    pub chunk_index: u32,
    pub content_type: ContentType,
    pub file_type: String,
    pub title: Option<String>,
    pub byte_range: Option<(u64, u64)>,
    pub vector: Vec<f32>,
}

/// Legacy JSON serialization format (kept for migration from `vectors.json`).
#[derive(serde::Serialize, serde::Deserialize)]
struct SerEntry {
    uri: String,
    text: String,
    vector: Vec<f32>,
    chunk_index: u32,
    content_type: ContentType,
    file_type: String,
    title: Option<String>,
    byte_range: Option<(u64, u64)>,
}

impl FlatVectorIndex {
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
        }
    }

    /// Load from binary format (convenience alias for `load_bin`).
    pub fn load(path: &std::path::Path) -> SiftResult<Self> {
        Self::load_bin(path)
    }

    // ---- Binary format I/O ----

    /// Save the index to a binary file.
    pub fn save(&self, path: &std::path::Path) -> SiftResult<()> {
        let entries = self
            .entries
            .lock()
            .map_err(|e| sift_core::SiftError::Storage(format!("Lock error: {}", e)))?;

        let dim = entries.first().map(|e| e.vector.len() as u32).unwrap_or(0);

        let mut buf: Vec<u8> = Vec::new();

        // Header
        buf.write_all(MAGIC)
            .map_err(|e| sift_core::SiftError::Storage(format!("Write error: {}", e)))?;
        buf.write_all(&(entries.len() as u64).to_le_bytes())
            .map_err(|e| sift_core::SiftError::Storage(format!("Write error: {}", e)))?;
        buf.write_all(&dim.to_le_bytes())
            .map_err(|e| sift_core::SiftError::Storage(format!("Write error: {}", e)))?;

        // Entries
        for entry in entries.iter() {
            // Key (URI)
            let key_bytes = entry.uri.as_bytes();
            buf.write_all(&(key_bytes.len() as u32).to_le_bytes())
                .map_err(|e| sift_core::SiftError::Storage(format!("Write error: {}", e)))?;
            buf.write_all(key_bytes)
                .map_err(|e| sift_core::SiftError::Storage(format!("Write error: {}", e)))?;

            // Vector (raw f32 LE)
            for &val in &entry.vector {
                buf.write_all(&val.to_le_bytes())
                    .map_err(|e| sift_core::SiftError::Storage(format!("Write error: {}", e)))?;
            }

            // Metadata (JSON)
            let meta = EntryMeta {
                chunk_index: entry.chunk_index,
                content_type: entry.content_type,
                file_type: entry.file_type.clone(),
                title: entry.title.clone(),
                byte_range: entry.byte_range,
                text: entry.text.clone(),
            };
            let meta_bytes = serde_json::to_vec(&meta)
                .map_err(|e| sift_core::SiftError::Storage(format!("Serialize meta: {}", e)))?;
            buf.write_all(&(meta_bytes.len() as u32).to_le_bytes())
                .map_err(|e| sift_core::SiftError::Storage(format!("Write error: {}", e)))?;
            buf.write_all(&meta_bytes)
                .map_err(|e| sift_core::SiftError::Storage(format!("Write error: {}", e)))?;
        }

        std::fs::write(path, buf)?;
        Ok(())
    }

    /// Load from binary format.
    pub fn load_bin(path: &std::path::Path) -> SiftResult<Self> {
        let data = std::fs::read(path)?;
        let mut cursor = &data[..];

        // Header
        let mut magic = [0u8; 4];
        cursor
            .read_exact(&mut magic)
            .map_err(|e| sift_core::SiftError::Storage(format!("Read magic: {}", e)))?;
        if &magic != MAGIC {
            return Err(sift_core::SiftError::Storage(
                "Invalid binary vector file: bad magic".to_string(),
            ));
        }

        let mut count_buf = [0u8; 8];
        cursor
            .read_exact(&mut count_buf)
            .map_err(|e| sift_core::SiftError::Storage(format!("Read count: {}", e)))?;
        let count = u64::from_le_bytes(count_buf) as usize;

        let mut dim_buf = [0u8; 4];
        cursor
            .read_exact(&mut dim_buf)
            .map_err(|e| sift_core::SiftError::Storage(format!("Read dim: {}", e)))?;
        let dim = u32::from_le_bytes(dim_buf) as usize;

        let mut entries = Vec::with_capacity(count);

        for _ in 0..count {
            // Key
            let mut key_len_buf = [0u8; 4];
            cursor
                .read_exact(&mut key_len_buf)
                .map_err(|e| sift_core::SiftError::Storage(format!("Read key_len: {}", e)))?;
            let key_len = u32::from_le_bytes(key_len_buf) as usize;

            let mut key_bytes = vec![0u8; key_len];
            cursor
                .read_exact(&mut key_bytes)
                .map_err(|e| sift_core::SiftError::Storage(format!("Read key: {}", e)))?;
            let uri = String::from_utf8(key_bytes)
                .map_err(|e| sift_core::SiftError::Storage(format!("Invalid UTF-8 key: {}", e)))?;

            // Vector
            let mut vector = Vec::with_capacity(dim);
            for _ in 0..dim {
                let mut f_buf = [0u8; 4];
                cursor
                    .read_exact(&mut f_buf)
                    .map_err(|e| sift_core::SiftError::Storage(format!("Read vector: {}", e)))?;
                vector.push(f32::from_le_bytes(f_buf));
            }

            // Metadata
            let mut meta_len_buf = [0u8; 4];
            cursor
                .read_exact(&mut meta_len_buf)
                .map_err(|e| sift_core::SiftError::Storage(format!("Read meta_len: {}", e)))?;
            let meta_len = u32::from_le_bytes(meta_len_buf) as usize;

            let mut meta_bytes = vec![0u8; meta_len];
            cursor
                .read_exact(&mut meta_bytes)
                .map_err(|e| sift_core::SiftError::Storage(format!("Read meta: {}", e)))?;
            let meta: EntryMeta = serde_json::from_slice(&meta_bytes)
                .map_err(|e| sift_core::SiftError::Storage(format!("Deserialize meta: {}", e)))?;

            entries.push(StoredEntry {
                uri,
                text: meta.text,
                vector,
                chunk_index: meta.chunk_index,
                content_type: meta.content_type,
                file_type: meta.file_type,
                title: meta.title,
                byte_range: meta.byte_range,
            });
        }

        Ok(Self {
            entries: Mutex::new(entries),
        })
    }

    // ---- Legacy JSON I/O (for migration) ----

    /// Load from legacy JSON format.
    pub fn load_json(path: &std::path::Path) -> SiftResult<Self> {
        let data = std::fs::read(path)?;
        let serializable: Vec<SerEntry> = serde_json::from_slice(&data)
            .map_err(|e| sift_core::SiftError::Storage(format!("Deserialize error: {}", e)))?;

        let entries: Vec<StoredEntry> = serializable
            .into_iter()
            .map(|e| StoredEntry {
                uri: e.uri,
                text: e.text,
                vector: e.vector,
                chunk_index: e.chunk_index,
                content_type: e.content_type,
                file_type: e.file_type,
                title: e.title,
                byte_range: e.byte_range,
            })
            .collect();

        Ok(Self {
            entries: Mutex::new(entries),
        })
    }

    /// Save as legacy JSON format (kept for compatibility / debugging).
    pub fn save_json(&self, path: &std::path::Path) -> SiftResult<()> {
        let entries = self
            .entries
            .lock()
            .map_err(|e| sift_core::SiftError::Storage(format!("Lock error: {}", e)))?;

        let serializable: Vec<SerEntry> = entries
            .iter()
            .map(|e| SerEntry {
                uri: e.uri.clone(),
                text: e.text.clone(),
                vector: e.vector.clone(),
                chunk_index: e.chunk_index,
                content_type: e.content_type,
                file_type: e.file_type.clone(),
                title: e.title.clone(),
                byte_range: e.byte_range,
            })
            .collect();

        let data = serde_json::to_vec(&serializable)
            .map_err(|e| sift_core::SiftError::Storage(format!("Serialize error: {}", e)))?;

        std::fs::write(path, data)?;
        Ok(())
    }

    // ---- Migration ----

    /// Load from binary if available, otherwise migrate from JSON, or create empty.
    ///
    /// Migration: reads `vectors.json`, saves as `vectors.bin`, renames JSON to `.json.bak`.
    pub fn load_or_migrate(index_dir: &std::path::Path) -> SiftResult<Self> {
        let bin_path = index_dir.join("vectors.bin");
        let json_path = index_dir.join("vectors.json");

        if bin_path.exists() {
            return Self::load_bin(&bin_path);
        }

        if json_path.exists() {
            tracing::info!("Migrating vectors.json -> vectors.bin");
            let store = Self::load_json(&json_path)?;
            store.save(&bin_path)?;
            let bak_path = index_dir.join("vectors.json.bak");
            std::fs::rename(&json_path, &bak_path)?;
            return Ok(store);
        }

        Ok(Self::new())
    }

    /// Export all entries for JSONL output.
    pub fn export_all(&self) -> SiftResult<Vec<ExportEntry>> {
        let entries = self
            .entries
            .lock()
            .map_err(|e| sift_core::SiftError::Storage(format!("Lock error: {}", e)))?;

        Ok(entries
            .iter()
            .map(|e| ExportEntry {
                uri: e.uri.clone(),
                text: e.text.clone(),
                chunk_index: e.chunk_index,
                content_type: e.content_type,
                file_type: e.file_type.clone(),
                title: e.title.clone(),
                byte_range: e.byte_range,
                vector: e.vector.clone(),
            })
            .collect())
    }
}

impl Default for FlatVectorIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ---- Trait implementations ----

impl VectorStore for FlatVectorIndex {
    fn insert(&self, chunks: &[EmbeddedChunk]) -> SiftResult<()> {
        let mut entries = self
            .entries
            .lock()
            .map_err(|e| sift_core::SiftError::Storage(format!("Lock error: {}", e)))?;

        for ec in chunks {
            entries.push(StoredEntry {
                uri: ec.chunk.source_uri.clone(),
                text: ec.chunk.text.clone(),
                vector: ec.vector.clone(),
                chunk_index: ec.chunk.chunk_index,
                content_type: ec.chunk.content_type,
                file_type: ec.chunk.file_type.clone(),
                title: ec.chunk.title.clone(),
                byte_range: ec.chunk.byte_range,
            });
        }

        Ok(())
    }

    fn search(&self, query_vector: &[f32], top_k: usize) -> SiftResult<Vec<SearchResult>> {
        let entries = self
            .entries
            .lock()
            .map_err(|e| sift_core::SiftError::Storage(format!("Lock error: {}", e)))?;

        let mut scored: Vec<(f32, &StoredEntry)> = entries
            .iter()
            .map(|entry| {
                let score = cosine_similarity(query_vector, &entry.vector);
                (score, entry)
            })
            .collect();

        // Sort descending by score
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let results = scored
            .into_iter()
            .take(top_k)
            .map(|(score, entry)| SearchResult {
                uri: entry.uri.clone(),
                text: entry.text.clone(),
                score,
                chunk_index: entry.chunk_index,
                content_type: entry.content_type,
                file_type: entry.file_type.clone(),
                title: entry.title.clone(),
                byte_range: entry.byte_range,
            })
            .collect();

        Ok(results)
    }

    fn delete_by_uri(&self, uri: &str) -> SiftResult<u64> {
        let mut entries = self
            .entries
            .lock()
            .map_err(|e| sift_core::SiftError::Storage(format!("Lock error: {}", e)))?;

        let before = entries.len();
        entries.retain(|e| e.uri != uri);
        Ok((before - entries.len()) as u64)
    }

    fn count(&self) -> SiftResult<u64> {
        let entries = self
            .entries
            .lock()
            .map_err(|e| sift_core::SiftError::Storage(format!("Lock error: {}", e)))?;
        Ok(entries.len() as u64)
    }
}

impl VectorIndex for FlatVectorIndex {
    fn save(&self, path: &std::path::Path) -> SiftResult<()> {
        FlatVectorIndex::save(self, path)
    }

    fn export_all(&self) -> SiftResult<Vec<ExportEntry>> {
        FlatVectorIndex::export_all(self)
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let (dot, norm_a, norm_b) = a
        .iter()
        .zip(b.iter())
        .fold((0.0f32, 0.0f32, 0.0f32), |(dot, na, nb), (&ai, &bi)| {
            (dot + ai * bi, na + ai * ai, nb + bi * bi)
        });

    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
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
            vector: vec![0.0; 3], // placeholder
        }
    }

    #[test]
    fn test_insert_and_count() {
        let store = FlatVectorIndex::new();
        store
            .insert(&[make_chunk("file:///a.txt", "hello", 0)])
            .unwrap();
        assert_eq!(store.count().unwrap(), 1);
    }

    #[test]
    fn test_search_returns_similar() {
        let store = FlatVectorIndex::new();

        let mut c1 = make_chunk("file:///a.txt", "hello", 0);
        c1.vector = vec![1.0, 0.0, 0.0];

        let mut c2 = make_chunk("file:///b.txt", "world", 0);
        c2.vector = vec![0.0, 1.0, 0.0];

        store.insert(&[c1, c2]).unwrap();

        let results = store.search(&[1.0, 0.0, 0.0], 10).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].uri, "file:///a.txt");
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn test_delete_by_uri() {
        let store = FlatVectorIndex::new();
        store
            .insert(&[
                make_chunk("file:///a.txt", "hello", 0),
                make_chunk("file:///a.txt", "world", 1),
                make_chunk("file:///b.txt", "foo", 0),
            ])
            .unwrap();

        assert_eq!(store.count().unwrap(), 3);
        let deleted = store.delete_by_uri("file:///a.txt").unwrap();
        assert_eq!(deleted, 2);
        assert_eq!(store.count().unwrap(), 1);
    }

    #[test]
    fn test_cosine_similarity() {
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!((cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]) - 0.0).abs() < 1e-6);
        assert!((cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]) - -1.0).abs() < 1e-6);
    }

    #[test]
    fn test_binary_save_load_roundtrip() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("vectors.bin");

        let store = FlatVectorIndex::new();
        let mut c1 = make_chunk("file:///a.txt", "hello world", 0);
        c1.vector = vec![1.0, 2.0, 3.0];
        store.insert(&[c1]).unwrap();
        store.save(&path).unwrap();

        let loaded = FlatVectorIndex::load_bin(&path).unwrap();
        assert_eq!(loaded.count().unwrap(), 1);
        let results = loaded.search(&[1.0, 2.0, 3.0], 1).unwrap();
        assert_eq!(results[0].text, "hello world");
    }

    #[test]
    fn test_json_save_load_roundtrip() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("vectors.json");

        let store = FlatVectorIndex::new();
        let mut c1 = make_chunk("file:///a.txt", "hello world", 0);
        c1.vector = vec![1.0, 2.0, 3.0];
        store.insert(&[c1]).unwrap();
        store.save_json(&path).unwrap();

        let loaded = FlatVectorIndex::load_json(&path).unwrap();
        assert_eq!(loaded.count().unwrap(), 1);
        let results = loaded.search(&[1.0, 2.0, 3.0], 1).unwrap();
        assert_eq!(results[0].text, "hello world");
    }

    #[test]
    fn test_load_or_migrate_from_json() {
        let dir = tempfile::TempDir::new().unwrap();
        let json_path = dir.path().join("vectors.json");

        // Create a legacy JSON file
        let store = FlatVectorIndex::new();
        let mut c1 = make_chunk("file:///a.txt", "migrated content", 0);
        c1.vector = vec![1.0, 2.0, 3.0];
        store.insert(&[c1]).unwrap();
        store.save_json(&json_path).unwrap();

        // load_or_migrate should read JSON, write binary, rename JSON
        let loaded = FlatVectorIndex::load_or_migrate(dir.path()).unwrap();
        assert_eq!(loaded.count().unwrap(), 1);

        // Binary file should now exist
        assert!(dir.path().join("vectors.bin").exists());
        // JSON should be renamed to .bak
        assert!(!dir.path().join("vectors.json").exists());
        assert!(dir.path().join("vectors.json.bak").exists());

        // A second call should load from binary
        let loaded2 = FlatVectorIndex::load_or_migrate(dir.path()).unwrap();
        assert_eq!(loaded2.count().unwrap(), 1);
        let results = loaded2.search(&[1.0, 2.0, 3.0], 1).unwrap();
        assert_eq!(results[0].text, "migrated content");
    }

    #[test]
    fn test_load_or_migrate_empty() {
        let dir = tempfile::TempDir::new().unwrap();
        let loaded = FlatVectorIndex::load_or_migrate(dir.path()).unwrap();
        assert_eq!(loaded.count().unwrap(), 0);
    }

    #[test]
    fn test_load_or_migrate_from_binary() {
        let dir = tempfile::TempDir::new().unwrap();
        let bin_path = dir.path().join("vectors.bin");

        let store = FlatVectorIndex::new();
        let mut c1 = make_chunk("file:///a.txt", "binary content", 0);
        c1.vector = vec![4.0, 5.0, 6.0];
        store.insert(&[c1]).unwrap();
        store.save(&bin_path).unwrap();

        let loaded = FlatVectorIndex::load_or_migrate(dir.path()).unwrap();
        assert_eq!(loaded.count().unwrap(), 1);
        let results = loaded.search(&[4.0, 5.0, 6.0], 1).unwrap();
        assert_eq!(results[0].text, "binary content");
    }

    #[test]
    fn test_export_all_empty_store() {
        let store = FlatVectorIndex::new();
        let entries = store.export_all().unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_export_all_returns_all_entries() {
        let store = FlatVectorIndex::new();
        let mut c1 = make_chunk("file:///a.txt", "hello", 0);
        c1.vector = vec![1.0, 2.0, 3.0];

        let mut c2 = make_chunk("file:///b.txt", "world", 0);
        c2.vector = vec![4.0, 5.0, 6.0];

        let mut c3 = make_chunk("file:///a.txt", "again", 1);
        c3.vector = vec![7.0, 8.0, 9.0];

        store.insert(&[c1, c2, c3]).unwrap();

        let entries = store.export_all().unwrap();
        assert_eq!(entries.len(), 3);
    }

    #[test]
    fn test_export_all_preserves_fields() {
        let store = FlatVectorIndex::new();

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
        store.insert(&[chunk]).unwrap();

        let entries = store.export_all().unwrap();
        assert_eq!(entries.len(), 1);

        let entry = &entries[0];
        assert_eq!(entry.uri, "file:///test.rs");
        assert_eq!(entry.text, "test content");
        assert_eq!(entry.chunk_index, 5);
        assert_eq!(entry.content_type, ContentType::Code);
        assert_eq!(entry.file_type, "rs");
        assert_eq!(entry.title.as_deref(), Some("My Module"));
        assert_eq!(entry.byte_range, Some((100, 200)));
        assert_eq!(entry.vector, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn test_export_all_preserves_none_fields() {
        let store = FlatVectorIndex::new();
        store
            .insert(&[make_chunk("file:///a.txt", "hello", 0)])
            .unwrap();

        let entries = store.export_all().unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].title.is_none());
        assert!(entries[0].byte_range.is_none());
    }

    #[test]
    fn test_export_all_after_delete() {
        let store = FlatVectorIndex::new();
        store
            .insert(&[
                make_chunk("file:///a.txt", "hello", 0),
                make_chunk("file:///b.txt", "world", 0),
            ])
            .unwrap();

        store.delete_by_uri("file:///a.txt").unwrap();

        let entries = store.export_all().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].uri, "file:///b.txt");
    }

    #[test]
    fn test_binary_roundtrip_preserves_all_fields() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("vectors.bin");

        let store = FlatVectorIndex::new();
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
        store.insert(&[chunk]).unwrap();
        store.save(&path).unwrap();

        let loaded = FlatVectorIndex::load_bin(&path).unwrap();
        let entries = loaded.export_all().unwrap();
        assert_eq!(entries.len(), 1);

        let entry = &entries[0];
        assert_eq!(entry.uri, "file:///test.rs");
        assert_eq!(entry.text, "test content");
        assert_eq!(entry.chunk_index, 5);
        assert_eq!(entry.content_type, ContentType::Code);
        assert_eq!(entry.file_type, "rs");
        assert_eq!(entry.title.as_deref(), Some("My Module"));
        assert_eq!(entry.byte_range, Some((100, 200)));
        assert_eq!(entry.vector, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn test_empty_store_binary_roundtrip() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("vectors.bin");

        let store = FlatVectorIndex::new();
        store.save(&path).unwrap();

        let loaded = FlatVectorIndex::load_bin(&path).unwrap();
        assert_eq!(loaded.count().unwrap(), 0);
    }

    #[test]
    fn test_cosine_768_dimensions() {
        let a: Vec<f32> = (0..768).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..768).map(|i| (i as f32).cos()).collect();
        let score = cosine_similarity(&a, &b);
        assert!(score > -1.0 && score < 1.0);
    }

    #[test]
    fn test_cosine_zero_vectors() {
        let z = vec![0.0f32; 768];
        assert_eq!(cosine_similarity(&z, &z), 0.0);
    }
}
