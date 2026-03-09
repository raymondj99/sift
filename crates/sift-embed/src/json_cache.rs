use serde::{Deserialize, Serialize};
use sift_core::SiftResult;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[derive(Serialize, Deserialize, Default)]
struct CacheData {
    entries: HashMap<String, Vec<f32>>,
}

struct Inner {
    data: CacheData,
    path: Option<PathBuf>,
}

impl Inner {
    fn flush(&self) {
        if let Some(ref path) = self.path {
            if let Ok(json) = serde_json::to_string(&self.data) {
                let _ = std::fs::write(path, json);
            }
        }
    }
}

/// Content-addressed embedding cache backed by a JSON file.
/// Maps BLAKE3 hash of text to embedding vector. Used as a lightweight
/// fallback when the `sqlite` feature is disabled.
pub struct EmbeddingCache {
    inner: Mutex<Inner>,
    hits: Mutex<u64>,
    misses: Mutex<u64>,
}

impl EmbeddingCache {
    pub fn open(path: &Path) -> SiftResult<Self> {
        let data = if path.exists() {
            let json = std::fs::read_to_string(path).map_err(|e| {
                sift_core::SiftError::Other(anyhow::anyhow!("Failed to read cache file: {}", e))
            })?;
            serde_json::from_str(&json).map_err(|e| {
                sift_core::SiftError::Other(anyhow::anyhow!("Failed to parse cache JSON: {}", e))
            })?
        } else {
            CacheData::default()
        };

        Ok(Self {
            inner: Mutex::new(Inner {
                data,
                path: Some(path.to_path_buf()),
            }),
            hits: Mutex::new(0),
            misses: Mutex::new(0),
        })
    }

    pub fn in_memory() -> SiftResult<Self> {
        Ok(Self {
            inner: Mutex::new(Inner {
                data: CacheData::default(),
                path: None,
            }),
            hits: Mutex::new(0),
            misses: Mutex::new(0),
        })
    }

    pub fn get(&self, text: &str) -> Option<Vec<f32>> {
        let key = hex_hash(text);
        let inner = self.inner.lock().ok()?;
        match inner.data.entries.get(&key) {
            Some(vec) => {
                if let Ok(mut hits) = self.hits.lock() {
                    *hits += 1;
                }
                Some(vec.clone())
            }
            None => {
                if let Ok(mut misses) = self.misses.lock() {
                    *misses += 1;
                }
                None
            }
        }
    }

    pub fn put(&self, text: &str, vector: &[f32]) {
        let key = hex_hash(text);
        if let Ok(mut inner) = self.inner.lock() {
            inner.data.entries.insert(key, vector.to_vec());
            inner.flush();
        }
    }

    pub fn put_batch(&self, entries: &[(&str, &[f32])]) {
        if entries.is_empty() {
            return;
        }
        if let Ok(mut inner) = self.inner.lock() {
            for (text, vector) in entries {
                let key = hex_hash(text);
                inner.data.entries.insert(key, vector.to_vec());
            }
            inner.flush();
        }
    }

    pub fn stats(&self) -> (u64, u64) {
        let hits = self.hits.lock().map(|h| *h).unwrap_or(0);
        let misses = self.misses.lock().map(|m| *m).unwrap_or(0);
        (hits, misses)
    }

    pub fn len(&self) -> usize {
        self.inner
            .lock()
            .map(|inner| inner.data.entries.len())
            .unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn hex_hash(text: &str) -> String {
    let hash = blake3::hash(text.as_bytes());
    hash.to_hex().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_miss_returns_none() {
        let cache = EmbeddingCache::in_memory().unwrap();
        assert!(cache.get("nonexistent").is_none());
        assert_eq!(cache.stats(), (0, 1));
    }

    #[test]
    fn test_put_then_get() {
        let cache = EmbeddingCache::in_memory().unwrap();
        cache.put("hello", &[1.0, 2.0, 3.0]);
        assert_eq!(cache.get("hello").unwrap(), vec![1.0, 2.0, 3.0]);
        assert_eq!(cache.stats(), (1, 0));
    }

    #[test]
    fn test_batch_put() {
        let cache = EmbeddingCache::in_memory().unwrap();
        cache.put_batch(&[("a", &[1.0]), ("b", &[2.0])]);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get("a").unwrap(), vec![1.0]);
        assert_eq!(cache.get("b").unwrap(), vec![2.0]);
    }

    #[test]
    fn test_overwrite() {
        let cache = EmbeddingCache::in_memory().unwrap();
        cache.put("key", &[1.0]);
        cache.put("key", &[2.0]);
        assert_eq!(cache.get("key").unwrap(), vec![2.0]);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_persistence() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("cache.json");

        {
            let cache = EmbeddingCache::open(&path).unwrap();
            cache.put("persistent", &[3.125, 2.875]);
            assert_eq!(cache.len(), 1);
        }

        {
            let cache = EmbeddingCache::open(&path).unwrap();
            assert_eq!(cache.len(), 1);
            assert_eq!(cache.get("persistent").unwrap(), vec![3.125, 2.875]);
        }
    }

    #[test]
    fn test_high_dim_vector_roundtrips() {
        let cache = EmbeddingCache::in_memory().unwrap();
        let vec768: Vec<f32> = (0..768).map(|i| (i as f32) * 0.001 - 0.384).collect();
        cache.put("doc", &vec768);
        let retrieved = cache.get("doc").unwrap();
        assert_eq!(retrieved, vec768);
    }
}
