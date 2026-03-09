use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;
use std::sync::Mutex;
use sift_core::SiftResult;

/// Content-addressed embedding cache backed by SQLite.
/// Maps BLAKE3 hash of text → embedding vector to avoid re-computation.
pub struct EmbeddingCache {
    conn: Mutex<Connection>,
    hits: Mutex<u64>,
    misses: Mutex<u64>,
}

impl EmbeddingCache {
    /// Open a persistent cache at the given path. Creates the DB if it doesn't exist.
    pub fn open(path: &Path) -> SiftResult<Self> {
        let conn = Connection::open(path).map_err(|e| {
            sift_core::SiftError::Other(anyhow::anyhow!("Failed to open embedding cache: {}", e))
        })?;

        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             CREATE TABLE IF NOT EXISTS cache (
                 hash BLOB PRIMARY KEY,
                 vector BLOB NOT NULL
             );",
        )
        .map_err(|e| {
            sift_core::SiftError::Other(anyhow::anyhow!("Failed to init cache schema: {}", e))
        })?;

        Ok(Self {
            conn: Mutex::new(conn),
            hits: Mutex::new(0),
            misses: Mutex::new(0),
        })
    }

    /// Create an in-memory cache (for tests).
    pub fn in_memory() -> SiftResult<Self> {
        let conn = Connection::open_in_memory().map_err(|e| {
            sift_core::SiftError::Other(anyhow::anyhow!("Failed to open in-memory cache: {}", e))
        })?;

        conn.execute_batch(
            "CREATE TABLE cache (
                 hash BLOB PRIMARY KEY,
                 vector BLOB NOT NULL
             );",
        )
        .map_err(|e| {
            sift_core::SiftError::Other(anyhow::anyhow!("Failed to init cache schema: {}", e))
        })?;

        Ok(Self {
            conn: Mutex::new(conn),
            hits: Mutex::new(0),
            misses: Mutex::new(0),
        })
    }

    /// Look up a cached embedding by text content.
    pub fn get(&self, text: &str) -> Option<Vec<f32>> {
        let hash = blake3_hash(text);
        let conn = self.conn.lock().ok()?;
        let result: Option<Vec<u8>> = conn
            .query_row(
                "SELECT vector FROM cache WHERE hash = ?1",
                params![hash.as_slice()],
                |row| row.get(0),
            )
            .optional()
            .ok()?;

        match result {
            Some(blob) => {
                if let Ok(mut hits) = self.hits.lock() {
                    *hits += 1;
                }
                Some(bytes_to_f32(&blob))
            }
            None => {
                if let Ok(mut misses) = self.misses.lock() {
                    *misses += 1;
                }
                None
            }
        }
    }

    /// Store an embedding in the cache.
    pub fn put(&self, text: &str, vector: &[f32]) {
        let hash = blake3_hash(text);
        let blob = f32_to_bytes(vector);
        if let Ok(conn) = self.conn.lock() {
            let _ = conn.execute(
                "INSERT OR REPLACE INTO cache (hash, vector) VALUES (?1, ?2)",
                params![hash.as_slice(), blob],
            );
        }
    }

    /// Store a batch of embeddings. Uses a transaction for efficiency.
    pub fn put_batch(&self, entries: &[(&str, &[f32])]) {
        if entries.is_empty() {
            return;
        }
        if let Ok(conn) = self.conn.lock() {
            let _ = conn.execute_batch("BEGIN");
            for (text, vector) in entries {
                let hash = blake3_hash(text);
                let blob = f32_to_bytes(vector);
                let _ = conn.execute(
                    "INSERT OR REPLACE INTO cache (hash, vector) VALUES (?1, ?2)",
                    params![hash.as_slice(), blob],
                );
            }
            let _ = conn.execute_batch("COMMIT");
        }
    }

    /// Return (hits, misses) for this session.
    pub fn stats(&self) -> (u64, u64) {
        let hits = self.hits.lock().map(|h| *h).unwrap_or(0);
        let misses = self.misses.lock().map(|m| *m).unwrap_or(0);
        (hits, misses)
    }

    /// Number of entries in the cache.
    pub fn len(&self) -> usize {
        self.conn
            .lock()
            .ok()
            .and_then(|conn| {
                conn.query_row("SELECT COUNT(*) FROM cache", [], |row| row.get::<_, i64>(0))
                    .ok()
            })
            .unwrap_or(0) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn blake3_hash(text: &str) -> [u8; 32] {
    *blake3::hash(text.as_bytes()).as_bytes()
}

fn f32_to_bytes(vec: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vec.len() * 4);
    for &v in vec {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::TempDir;

    // --- Basic contract ---

    #[test]
    fn test_miss_returns_none_and_increments_miss_counter() {
        let cache = EmbeddingCache::in_memory().unwrap();
        assert!(cache.get("nonexistent").is_none());
        assert_eq!(cache.stats(), (0, 1));
    }

    #[test]
    fn test_put_then_get_returns_exact_vector() {
        let cache = EmbeddingCache::in_memory().unwrap();
        cache.put("hello", &[1.0, 2.0, 3.0]);
        assert_eq!(cache.get("hello").unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_hit_increments_hit_counter() {
        let cache = EmbeddingCache::in_memory().unwrap();
        cache.put("x", &[1.0]);
        cache.get("x");
        cache.get("x");
        assert_eq!(cache.stats(), (2, 0));
    }

    #[test]
    fn test_distinct_keys_are_independent() {
        let cache = EmbeddingCache::in_memory().unwrap();
        cache.put("alpha", &[1.0]);
        cache.put("beta", &[2.0]);
        assert_eq!(cache.get("alpha").unwrap(), vec![1.0]);
        assert_eq!(cache.get("beta").unwrap(), vec![2.0]);
        assert_eq!(cache.len(), 2);
    }

    // --- Overwrite semantics ---

    #[test]
    fn test_put_same_key_twice_keeps_latest_value() {
        let cache = EmbeddingCache::in_memory().unwrap();
        cache.put("key", &[1.0, 0.0]);
        cache.put("key", &[0.0, 1.0]);
        assert_eq!(cache.get("key").unwrap(), vec![0.0, 1.0]);
        assert_eq!(cache.len(), 1); // no duplicates
    }

    // --- Batch operations ---

    #[test]
    fn test_batch_put_inserts_all_entries() {
        let cache = EmbeddingCache::in_memory().unwrap();
        let vecs: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32]).collect();
        let entries: Vec<(&str, &[f32])> = (0..100)
            .map(|i| {
                let s: &str = Box::leak(format!("text_{}", i).into_boxed_str());
                (s, vecs[i].as_slice())
            })
            .collect();
        cache.put_batch(&entries);
        assert_eq!(cache.len(), 100);
        assert_eq!(cache.get("text_42").unwrap(), vec![42.0]);
    }

    #[test]
    fn test_batch_put_empty_is_noop() {
        let cache = EmbeddingCache::in_memory().unwrap();
        cache.put_batch(&[]);
        assert_eq!(cache.len(), 0);
    }

    // --- Edge cases ---

    #[test]
    fn test_empty_string_is_valid_cache_key() {
        let cache = EmbeddingCache::in_memory().unwrap();
        cache.put("", &[9.0]);
        assert_eq!(cache.get("").unwrap(), vec![9.0]);
    }

    #[test]
    fn test_high_dimensional_vector_roundtrips_exactly() {
        let cache = EmbeddingCache::in_memory().unwrap();
        // Realistic 768-dim embedding
        let vec768: Vec<f32> = (0..768).map(|i| (i as f32) * 0.001 - 0.384).collect();
        cache.put("doc", &vec768);
        let retrieved = cache.get("doc").unwrap();
        assert_eq!(retrieved.len(), 768);
        assert_eq!(retrieved, vec768); // exact f32 equality — no lossy encoding
    }

    #[test]
    fn test_negative_and_special_float_values_preserved() {
        let cache = EmbeddingCache::in_memory().unwrap();
        let special = vec![0.0, -0.0, f32::MIN, f32::MAX, f32::EPSILON, 1e-38];
        cache.put("special", &special);
        let result = cache.get("special").unwrap();
        assert_eq!(result.len(), special.len());
        for (a, b) in result.iter().zip(special.iter()) {
            assert_eq!(a.to_bits(), b.to_bits()); // bit-exact, catches -0.0
        }
    }

    // --- Disk persistence ---

    #[test]
    fn test_cache_survives_close_and_reopen() {
        let dir = TempDir::new().unwrap();
        let path: PathBuf = dir.path().join("cache.db");

        // Session 1: write
        {
            let cache = EmbeddingCache::open(&path).unwrap();
            cache.put("persistent", &[3.125, 2.875]);
            assert_eq!(cache.len(), 1);
        }
        // cache dropped, connection closed

        // Session 2: read back
        {
            let cache = EmbeddingCache::open(&path).unwrap();
            assert_eq!(cache.len(), 1);
            assert_eq!(cache.get("persistent").unwrap(), vec![3.125, 2.875]);
        }
    }

    #[test]
    fn test_stats_reset_on_new_session_but_data_persists() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("cache.db");

        {
            let cache = EmbeddingCache::open(&path).unwrap();
            cache.put("x", &[1.0]);
            cache.get("x"); // hit
            assert_eq!(cache.stats(), (1, 0));
        }

        {
            let cache = EmbeddingCache::open(&path).unwrap();
            assert_eq!(cache.stats(), (0, 0)); // stats are per-session
            assert_eq!(cache.len(), 1); // data survived
        }
    }

    // --- Serialization helpers ---

    #[test]
    fn test_f32_bytes_roundtrip() {
        let original = vec![1.0f32, -2.5, 0.0, f32::MAX, f32::MIN_POSITIVE];
        let bytes = f32_to_bytes(&original);
        assert_eq!(bytes.len(), original.len() * 4);
        let decoded = bytes_to_f32(&bytes);
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_empty_vector_roundtrip() {
        let bytes = f32_to_bytes(&[]);
        assert!(bytes.is_empty());
        let decoded = bytes_to_f32(&bytes);
        assert!(decoded.is_empty());
    }
}
