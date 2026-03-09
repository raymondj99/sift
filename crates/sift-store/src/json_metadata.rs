use serde::{Deserialize, Serialize};
use sift_core::{IndexStats, SiftError, SiftResult};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[derive(Serialize, Deserialize, Clone)]
struct SourceRecord {
    content_hash: [u8; 32],
    file_size: u64,
    file_type: String,
    modified_at: Option<i64>,
    indexed_at: i64,
    chunk_count: u32,
}

#[derive(Serialize, Deserialize, Default)]
struct StoreData {
    sources: HashMap<String, SourceRecord>,
    meta: HashMap<String, String>,
}

struct Inner {
    data: StoreData,
    path: Option<PathBuf>,
}

impl Inner {
    fn flush(&self) -> SiftResult<()> {
        if let Some(ref path) = self.path {
            let json = serde_json::to_string(&self.data)
                .map_err(|e| SiftError::Storage(format!("JSON serialize error: {}", e)))?;
            std::fs::write(path, json)
                .map_err(|e| SiftError::Storage(format!("File write error: {}", e)))?;
        }
        Ok(())
    }
}

/// JSON-file-backed metadata store for tracking indexed sources.
/// Used as a lightweight fallback when the `sqlite` feature is disabled.
pub struct MetadataStore {
    inner: Mutex<Inner>,
}

impl MetadataStore {
    pub fn open(path: &Path) -> SiftResult<Self> {
        let data = if path.exists() {
            let json = std::fs::read_to_string(path)
                .map_err(|e| SiftError::Storage(format!("File read error: {}", e)))?;
            serde_json::from_str(&json)
                .map_err(|e| SiftError::Storage(format!("JSON parse error: {}", e)))?
        } else {
            StoreData::default()
        };

        Ok(Self {
            inner: Mutex::new(Inner {
                data,
                path: Some(path.to_path_buf()),
            }),
        })
    }

    pub fn open_in_memory() -> SiftResult<Self> {
        Ok(Self {
            inner: Mutex::new(Inner {
                data: StoreData::default(),
                path: None,
            }),
        })
    }

    pub fn check_source(&self, uri: &str, content_hash: &[u8; 32]) -> SiftResult<Option<bool>> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| SiftError::Storage(format!("Lock error: {}", e)))?;

        match inner.data.sources.get(uri) {
            Some(record) => Ok(Some(record.content_hash == *content_hash)),
            None => Ok(None),
        }
    }

    pub fn upsert_source(
        &self,
        uri: &str,
        content_hash: &[u8; 32],
        file_size: u64,
        file_type: &str,
        modified_at: Option<i64>,
        chunk_count: u32,
    ) -> SiftResult<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| SiftError::Storage(format!("Lock error: {}", e)))?;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| SiftError::Storage(format!("System clock error: {}", e)))?
            .as_secs() as i64;
        inner.data.sources.insert(
            uri.to_string(),
            SourceRecord {
                content_hash: *content_hash,
                file_size,
                file_type: file_type.to_string(),
                modified_at,
                indexed_at: now,
                chunk_count,
            },
        );

        inner.flush()
    }

    pub fn remove_source(&self, uri: &str) -> SiftResult<bool> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| SiftError::Storage(format!("Lock error: {}", e)))?;

        let existed = inner.data.sources.remove(uri).is_some();
        if existed {
            inner.flush()?;
        }
        Ok(existed)
    }

    pub fn stats(&self) -> SiftResult<IndexStats> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| SiftError::Storage(format!("Lock error: {}", e)))?;

        let total_sources = inner.data.sources.len() as u64;
        let total_chunks: u64 = inner
            .data
            .sources
            .values()
            .map(|r| r.chunk_count as u64)
            .sum();

        let mut file_type_counts: HashMap<String, u64> = HashMap::new();
        for record in inner.data.sources.values() {
            *file_type_counts
                .entry(record.file_type.clone())
                .or_insert(0) += 1;
        }

        Ok(IndexStats {
            total_sources,
            total_chunks,
            index_size_bytes: 0,
            file_type_counts,
        })
    }

    pub fn list_sources(&self) -> SiftResult<Vec<(String, String, u32)>> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| SiftError::Storage(format!("Lock error: {}", e)))?;

        let mut sources: Vec<(String, String, u32)> = inner
            .data
            .sources
            .iter()
            .map(|(uri, record)| (uri.clone(), record.file_type.clone(), record.chunk_count))
            .collect();
        sources.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(sources)
    }

    pub fn find_stale_sources(&self) -> SiftResult<Vec<String>> {
        let sources = self.list_sources()?;
        let stale: Vec<String> = sources
            .into_iter()
            .filter(|(uri, _, _)| {
                if let Some(path) = uri.strip_prefix("file://") {
                    !std::path::Path::new(path).exists()
                } else {
                    false
                }
            })
            .map(|(uri, _, _)| uri)
            .collect();
        Ok(stale)
    }

    pub fn set_meta(&self, key: &str, value: &str) -> SiftResult<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| SiftError::Storage(format!("Lock error: {}", e)))?;

        inner.data.meta.insert(key.to_string(), value.to_string());
        inner.flush()
    }

    pub fn get_meta(&self, key: &str) -> SiftResult<Option<String>> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| SiftError::Storage(format!("Lock error: {}", e)))?;

        Ok(inner.data.meta.get(key).cloned())
    }

    pub fn uris_modified_after(
        &self,
        after_ts: i64,
    ) -> SiftResult<std::collections::HashSet<String>> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| SiftError::Storage(format!("Lock error: {}", e)))?;

        let uris = inner
            .data
            .sources
            .iter()
            .filter(|(_, record)| record.modified_at.is_some_and(|ts| ts >= after_ts))
            .map(|(uri, _)| uri.clone())
            .collect();
        Ok(uris)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_store_lifecycle() {
        let store = MetadataStore::open_in_memory().unwrap();
        let hash = [0u8; 32];

        assert!(store
            .check_source("file:///test.txt", &hash)
            .unwrap()
            .is_none());

        store
            .upsert_source("file:///test.txt", &hash, 100, "txt", Some(1000), 5)
            .unwrap();

        assert_eq!(
            store.check_source("file:///test.txt", &hash).unwrap(),
            Some(true)
        );

        let new_hash = [1u8; 32];
        assert_eq!(
            store.check_source("file:///test.txt", &new_hash).unwrap(),
            Some(false)
        );

        let stats = store.stats().unwrap();
        assert_eq!(stats.total_sources, 1);
        assert_eq!(stats.total_chunks, 5);

        assert!(store.remove_source("file:///test.txt").unwrap());
        assert!(!store.remove_source("file:///test.txt").unwrap());
    }

    #[test]
    fn test_list_sources() {
        let store = MetadataStore::open_in_memory().unwrap();
        let hash = [0u8; 32];

        store
            .upsert_source("file:///a.txt", &hash, 10, "txt", None, 1)
            .unwrap();
        store
            .upsert_source("file:///b.rs", &hash, 20, "rs", None, 3)
            .unwrap();

        let sources = store.list_sources().unwrap();
        assert_eq!(sources.len(), 2);
    }

    #[test]
    fn test_metadata_kv() {
        let store = MetadataStore::open_in_memory().unwrap();

        assert!(store.get_meta("model").unwrap().is_none());
        store.set_meta("model", "nomic-embed-text-v2").unwrap();
        assert_eq!(
            store.get_meta("model").unwrap().as_deref(),
            Some("nomic-embed-text-v2")
        );

        store.set_meta("model", "bge-m3").unwrap();
        assert_eq!(store.get_meta("model").unwrap().as_deref(), Some("bge-m3"));
    }

    #[test]
    fn test_uris_modified_after() {
        let store = MetadataStore::open_in_memory().unwrap();
        let hash = [0u8; 32];

        store
            .upsert_source("file:///old.txt", &hash, 10, "txt", Some(1000), 1)
            .unwrap();
        store
            .upsert_source("file:///mid.txt", &hash, 10, "txt", Some(2000), 1)
            .unwrap();
        store
            .upsert_source("file:///new.txt", &hash, 10, "txt", Some(3000), 1)
            .unwrap();

        let result = store.uris_modified_after(2000).unwrap();
        assert_eq!(result.len(), 2);
        assert!(result.contains("file:///mid.txt"));
        assert!(result.contains("file:///new.txt"));
    }

    #[test]
    fn test_persistence() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("metadata.json");
        let hash = [42u8; 32];

        {
            let store = MetadataStore::open(&path).unwrap();
            store
                .upsert_source("file:///x.txt", &hash, 50, "txt", Some(100), 2)
                .unwrap();
            store.set_meta("version", "1").unwrap();
        }

        {
            let store = MetadataStore::open(&path).unwrap();
            assert_eq!(
                store.check_source("file:///x.txt", &hash).unwrap(),
                Some(true)
            );
            assert_eq!(store.get_meta("version").unwrap().as_deref(), Some("1"));
            let stats = store.stats().unwrap();
            assert_eq!(stats.total_sources, 1);
            assert_eq!(stats.total_chunks, 2);
        }
    }
}
