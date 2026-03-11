use rusqlite::{params, Connection};
use sift_core::{IndexStats, SiftResult};
use std::path::Path;
use std::sync::Mutex;

/// SQLite-backed metadata store for tracking indexed sources.
pub struct MetadataStore {
    conn: Mutex<Connection>,
}

impl MetadataStore {
    pub fn open(path: &Path) -> SiftResult<Self> {
        let conn = Connection::open(path)
            .map_err(|e| sift_core::SiftError::Storage(format!("SQLite open error: {e}")))?;

        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             PRAGMA foreign_keys=ON;
             PRAGMA busy_timeout=5000;
             PRAGMA cache_size=-8000;",
        )
        .map_err(|e| sift_core::SiftError::Storage(format!("SQLite pragma error: {e}")))?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS sources (
                uri TEXT PRIMARY KEY,
                content_hash BLOB NOT NULL,
                file_size INTEGER,
                file_type TEXT,
                modified_at INTEGER,
                indexed_at INTEGER NOT NULL,
                chunk_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'indexed'
            );

            CREATE TABLE IF NOT EXISTS index_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );",
        )
        .map_err(|e| sift_core::SiftError::Storage(format!("SQLite schema error: {e}")))?;

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    pub fn open_in_memory() -> SiftResult<Self> {
        let conn = Connection::open_in_memory()
            .map_err(|e| sift_core::SiftError::Storage(format!("SQLite error: {e}")))?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS sources (
                uri TEXT PRIMARY KEY,
                content_hash BLOB NOT NULL,
                file_size INTEGER,
                file_type TEXT,
                modified_at INTEGER,
                indexed_at INTEGER NOT NULL,
                chunk_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'indexed'
            );

            CREATE TABLE IF NOT EXISTS index_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );",
        )
        .map_err(|e| sift_core::SiftError::Storage(format!("SQLite schema error: {e}")))?;

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Check if a source exists and its content hash matches.
    /// Returns Some(true) if hash matches (skip), Some(false) if hash differs (re-index), None if new.
    pub fn check_source(&self, uri: &str, content_hash: &[u8; 32]) -> SiftResult<Option<bool>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| sift_core::SiftError::Storage(format!("Lock error: {e}")))?;

        let mut stmt = conn
            .prepare("SELECT content_hash FROM sources WHERE uri = ?1")
            .map_err(|e| sift_core::SiftError::Storage(format!("Query error: {e}")))?;

        let result: Option<Vec<u8>> = stmt.query_row(params![uri], |row| row.get(0)).ok();

        match result {
            Some(stored_hash) => Ok(Some(stored_hash.as_slice() == content_hash.as_slice())),
            None => Ok(None),
        }
    }

    /// Upsert a source record.
    pub fn upsert_source(
        &self,
        uri: &str,
        content_hash: &[u8; 32],
        file_size: u64,
        file_type: &str,
        modified_at: Option<i64>,
        chunk_count: u32,
    ) -> SiftResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| sift_core::SiftError::Storage(format!("Lock error: {e}")))?;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| sift_core::SiftError::Storage(format!("System clock error: {e}")))?
            .as_secs() as i64;

        conn.execute(
            "INSERT INTO sources (uri, content_hash, file_size, file_type, modified_at, indexed_at, chunk_count, status)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 'indexed')
             ON CONFLICT(uri) DO UPDATE SET
                content_hash = excluded.content_hash,
                file_size = excluded.file_size,
                file_type = excluded.file_type,
                modified_at = excluded.modified_at,
                indexed_at = excluded.indexed_at,
                chunk_count = excluded.chunk_count,
                status = 'indexed'",
            params![
                uri,
                content_hash.as_slice(),
                file_size as i64,
                file_type,
                modified_at,
                now,
                i64::from(chunk_count),
            ],
        )
        .map_err(|e| sift_core::SiftError::Storage(format!("Upsert error: {e}")))?;

        Ok(())
    }

    /// Batch upsert multiple source records in a single transaction.
    #[allow(clippy::type_complexity)]
    pub fn upsert_batch(
        &self,
        items: &[(&str, &[u8; 32], u64, &str, Option<i64>, u32)],
    ) -> SiftResult<()> {
        if items.is_empty() {
            return Ok(());
        }

        let conn = self
            .conn
            .lock()
            .map_err(|e| sift_core::SiftError::Storage(format!("Lock error: {e}")))?;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| sift_core::SiftError::Storage(format!("System clock error: {e}")))?
            .as_secs() as i64;

        let tx = conn
            .unchecked_transaction()
            .map_err(|e| sift_core::SiftError::Storage(format!("Transaction error: {e}")))?;

        {
            let mut stmt = tx
                .prepare_cached(
                    "INSERT INTO sources (uri, content_hash, file_size, file_type, modified_at, indexed_at, chunk_count, status)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 'indexed')
                     ON CONFLICT(uri) DO UPDATE SET
                        content_hash = excluded.content_hash,
                        file_size = excluded.file_size,
                        file_type = excluded.file_type,
                        modified_at = excluded.modified_at,
                        indexed_at = excluded.indexed_at,
                        chunk_count = excluded.chunk_count,
                        status = 'indexed'",
                )
                .map_err(|e| sift_core::SiftError::Storage(format!("Prepare error: {e}")))?;

            for &(uri, hash, size, file_type, modified_at, chunk_count) in items {
                stmt.execute(params![
                    uri,
                    hash.as_slice(),
                    size as i64,
                    file_type,
                    modified_at,
                    now,
                    i64::from(chunk_count),
                ])
                .map_err(|e| sift_core::SiftError::Storage(format!("Upsert error: {e}")))?;
            }
        }

        tx.commit()
            .map_err(|e| sift_core::SiftError::Storage(format!("Commit error: {e}")))?;

        Ok(())
    }

    /// Remove a source and return true if it existed.
    pub fn remove_source(&self, uri: &str) -> SiftResult<bool> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| sift_core::SiftError::Storage(format!("Lock error: {e}")))?;

        let rows = conn
            .execute("DELETE FROM sources WHERE uri = ?1", params![uri])
            .map_err(|e| sift_core::SiftError::Storage(format!("Delete error: {e}")))?;

        Ok(rows > 0)
    }

    /// Get index statistics.
    pub fn stats(&self) -> SiftResult<IndexStats> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| sift_core::SiftError::Storage(format!("Lock error: {e}")))?;

        let total_sources: u64 = conn
            .query_row("SELECT COUNT(*) FROM sources", [], |row| row.get(0))
            .map_err(|e| sift_core::SiftError::Storage(format!("Query error: {e}")))?;

        let total_chunks: u64 = conn
            .query_row(
                "SELECT COALESCE(SUM(chunk_count), 0) FROM sources",
                [],
                |row| row.get(0),
            )
            .map_err(|e| sift_core::SiftError::Storage(format!("Query error: {e}")))?;

        let mut stmt = conn
            .prepare("SELECT file_type, COUNT(*) FROM sources GROUP BY file_type")
            .map_err(|e| sift_core::SiftError::Storage(format!("Query error: {e}")))?;

        let file_type_counts = stmt
            .query_map([], |row| {
                let ft: String = row.get(0)?;
                let count: u64 = row.get(1)?;
                Ok((ft, count))
            })
            .map_err(|e| sift_core::SiftError::Storage(format!("Query error: {e}")))?
            .filter_map(std::result::Result::ok)
            .collect();

        Ok(IndexStats {
            total_sources,
            total_chunks,
            index_size_bytes: 0, // computed externally
            file_type_counts,
        })
    }

    /// List all indexed source URIs.
    pub fn list_sources(&self) -> SiftResult<Vec<(String, String, u32)>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| sift_core::SiftError::Storage(format!("Lock error: {e}")))?;

        let mut stmt = conn
            .prepare("SELECT uri, file_type, chunk_count FROM sources ORDER BY uri")
            .map_err(|e| sift_core::SiftError::Storage(format!("Query error: {e}")))?;

        let results = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, u32>(2)?,
                ))
            })
            .map_err(|e| sift_core::SiftError::Storage(format!("Query error: {e}")))?
            .filter_map(std::result::Result::ok)
            .collect();

        Ok(results)
    }

    /// Find sources that no longer exist on disk (for cleanup).
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

    /// Set a metadata key-value pair.
    pub fn set_meta(&self, key: &str, value: &str) -> SiftResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| sift_core::SiftError::Storage(format!("Lock error: {e}")))?;

        conn.execute(
            "INSERT INTO index_meta (key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )
        .map_err(|e| sift_core::SiftError::Storage(format!("Meta set error: {e}")))?;

        Ok(())
    }

    /// Return the set of URIs whose `modified_at` is >= `after_ts`.
    pub fn uris_modified_after(
        &self,
        after_ts: i64,
    ) -> SiftResult<std::collections::HashSet<String>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| sift_core::SiftError::Storage(format!("Lock error: {e}")))?;

        let mut stmt = conn
            .prepare("SELECT uri FROM sources WHERE modified_at >= ?1")
            .map_err(|e| sift_core::SiftError::Storage(format!("Query error: {e}")))?;

        let uris = stmt
            .query_map(params![after_ts], |row| row.get::<_, String>(0))
            .map_err(|e| sift_core::SiftError::Storage(format!("Query error: {e}")))?
            .filter_map(std::result::Result::ok)
            .collect();

        Ok(uris)
    }

    /// Get a metadata value by key.
    pub fn get_meta(&self, key: &str) -> SiftResult<Option<String>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| sift_core::SiftError::Storage(format!("Lock error: {e}")))?;

        let result = conn
            .query_row(
                "SELECT value FROM index_meta WHERE key = ?1",
                params![key],
                |row| row.get(0),
            )
            .ok();

        Ok(result)
    }

    /// Load all known (uri, `content_hash`) pairs into memory for fast batch lookups.
    pub fn load_all_hashes(&self) -> SiftResult<std::collections::HashMap<String, Vec<u8>>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| sift_core::SiftError::Storage(format!("Lock error: {e}")))?;

        let mut stmt = conn
            .prepare("SELECT uri, content_hash FROM sources")
            .map_err(|e| sift_core::SiftError::Storage(format!("Query error: {e}")))?;

        let map = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
            })
            .map_err(|e| sift_core::SiftError::Storage(format!("Query error: {e}")))?
            .filter_map(std::result::Result::ok)
            .collect();

        Ok(map)
    }

    /// Begin an explicit transaction for batch operations.
    pub fn begin_transaction(&self) -> SiftResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| sift_core::SiftError::Storage(format!("Lock error: {e}")))?;
        conn.execute_batch("BEGIN IMMEDIATE")
            .map_err(|e| sift_core::SiftError::Storage(format!("Begin transaction: {e}")))?;
        Ok(())
    }

    /// Commit the current transaction.
    pub fn commit_transaction(&self) -> SiftResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| sift_core::SiftError::Storage(format!("Lock error: {e}")))?;
        conn.execute_batch("COMMIT")
            .map_err(|e| sift_core::SiftError::Storage(format!("Commit transaction: {e}")))?;
        Ok(())
    }

    /// Rollback the current transaction.
    pub fn rollback_transaction(&self) -> SiftResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| sift_core::SiftError::Storage(format!("Lock error: {e}")))?;
        conn.execute_batch("ROLLBACK")
            .map_err(|e| sift_core::SiftError::Storage(format!("Rollback transaction: {e}")))?;
        Ok(())
    }
}

/// RAII guard that auto-rolls back an uncommitted transaction on drop.
pub struct TransactionGuard<'a> {
    metadata: &'a MetadataStore,
    committed: bool,
}

impl<'a> TransactionGuard<'a> {
    /// Begin a new transaction and return a guard.
    pub fn begin(metadata: &'a MetadataStore) -> SiftResult<Self> {
        metadata.begin_transaction()?;
        Ok(Self {
            metadata,
            committed: false,
        })
    }

    /// Commit and consume the guard. Starts a new transaction for the next batch.
    pub fn commit_and_reopen(&mut self) -> SiftResult<()> {
        self.metadata.commit_transaction()?;
        self.metadata.begin_transaction()?;
        // Still active — a new transaction is open
        Ok(())
    }

    /// Final commit — marks the guard as committed so drop is a no-op.
    pub fn commit(mut self) -> SiftResult<()> {
        self.metadata.commit_transaction()?;
        self.committed = true;
        Ok(())
    }
}

impl Drop for TransactionGuard<'_> {
    fn drop(&mut self) {
        if !self.committed {
            let _ = self.metadata.rollback_transaction();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_store_lifecycle() {
        let store = MetadataStore::open_in_memory().unwrap();

        // New source should return None
        let hash = [0u8; 32];
        assert!(store
            .check_source("file:///test.txt", &hash)
            .unwrap()
            .is_none());

        // Insert source
        store
            .upsert_source("file:///test.txt", &hash, 100, "txt", Some(1000), 5)
            .unwrap();

        // Same hash should return Some(true)
        assert_eq!(
            store.check_source("file:///test.txt", &hash).unwrap(),
            Some(true)
        );

        // Different hash should return Some(false)
        let new_hash = [1u8; 32];
        assert_eq!(
            store.check_source("file:///test.txt", &new_hash).unwrap(),
            Some(false)
        );

        // Stats should reflect one source
        let stats = store.stats().unwrap();
        assert_eq!(stats.total_sources, 1);
        assert_eq!(stats.total_chunks, 5);

        // Remove source
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

        // Overwrite
        store.set_meta("model", "bge-m3").unwrap();
        assert_eq!(store.get_meta("model").unwrap().as_deref(), Some("bge-m3"));
    }

    #[test]
    fn test_uris_modified_after_filters_by_timestamp() {
        let store = MetadataStore::open_in_memory().unwrap();
        let hash = [0u8; 32];

        // Insert sources with different modified_at timestamps
        store
            .upsert_source("file:///old.txt", &hash, 10, "txt", Some(1000), 1)
            .unwrap();
        store
            .upsert_source("file:///mid.txt", &hash, 10, "txt", Some(2000), 1)
            .unwrap();
        store
            .upsert_source("file:///new.txt", &hash, 10, "txt", Some(3000), 1)
            .unwrap();

        // After 2000: should include mid.txt and new.txt
        let result = store.uris_modified_after(2000).unwrap();
        assert_eq!(result.len(), 2);
        assert!(result.contains("file:///mid.txt"));
        assert!(result.contains("file:///new.txt"));
        assert!(!result.contains("file:///old.txt"));
    }

    #[test]
    fn test_uris_modified_after_high_threshold_returns_empty() {
        let store = MetadataStore::open_in_memory().unwrap();
        let hash = [0u8; 32];

        store
            .upsert_source("file:///old.txt", &hash, 10, "txt", Some(1000), 1)
            .unwrap();

        // After 9999: nothing qualifies
        let result = store.uris_modified_after(9999).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_uris_modified_after_zero_threshold_returns_all() {
        let store = MetadataStore::open_in_memory().unwrap();
        let hash = [0u8; 32];

        store
            .upsert_source("file:///a.txt", &hash, 10, "txt", Some(100), 1)
            .unwrap();
        store
            .upsert_source("file:///b.txt", &hash, 10, "txt", Some(200), 1)
            .unwrap();

        let result = store.uris_modified_after(0).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_uris_modified_after_excludes_null_modified_at() {
        let store = MetadataStore::open_in_memory().unwrap();
        let hash = [0u8; 32];

        // Source with no modified_at (None)
        store
            .upsert_source("file:///no_ts.txt", &hash, 10, "txt", None, 1)
            .unwrap();
        // Source with a timestamp
        store
            .upsert_source("file:///has_ts.txt", &hash, 10, "txt", Some(5000), 1)
            .unwrap();

        let result = store.uris_modified_after(1000).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result.contains("file:///has_ts.txt"));
        assert!(!result.contains("file:///no_ts.txt"));
    }

    #[test]
    fn test_uris_modified_after_on_empty_store() {
        let store = MetadataStore::open_in_memory().unwrap();
        let result = store.uris_modified_after(0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_uris_modified_after_exact_boundary() {
        let store = MetadataStore::open_in_memory().unwrap();
        let hash = [0u8; 32];

        store
            .upsert_source("file:///exact.txt", &hash, 10, "txt", Some(5000), 1)
            .unwrap();

        // Exact match: >= should include it
        let result = store.uris_modified_after(5000).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result.contains("file:///exact.txt"));

        // One past: should exclude it
        let result = store.uris_modified_after(5001).unwrap();
        assert!(result.is_empty());
    }

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

        store
            .upsert_source("file:///a.txt", &hash_a, 10, "txt", None, 1)
            .unwrap();
        store
            .upsert_source("file:///b.rs", &hash_b, 20, "rs", None, 3)
            .unwrap();

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

        store
            .upsert_source("file:///a.txt", &hash_v1, 10, "txt", None, 1)
            .unwrap();
        store
            .upsert_source("file:///a.txt", &hash_v2, 10, "txt", None, 1)
            .unwrap();

        let hashes = store.load_all_hashes().unwrap();
        assert_eq!(hashes.len(), 1);
        assert_eq!(hashes["file:///a.txt"].as_slice(), hash_v2.as_slice());
    }

    #[test]
    fn test_transaction_batching() {
        let store = MetadataStore::open_in_memory().unwrap();
        let hash = [0u8; 32];

        store.begin_transaction().unwrap();
        for i in 0..100 {
            store
                .upsert_source(&format!("file:///test_{i}.txt"), &hash, 10, "txt", None, 1)
                .unwrap();
        }
        store.commit_transaction().unwrap();

        let stats = store.stats().unwrap();
        assert_eq!(stats.total_sources, 100);
    }

    #[test]
    fn test_multiple_transaction_batches() {
        let store = MetadataStore::open_in_memory().unwrap();
        let hash = [0u8; 32];

        // Batch 1
        store.begin_transaction().unwrap();
        for i in 0..50 {
            store
                .upsert_source(&format!("file:///{i}.txt"), &hash, 10, "txt", None, 1)
                .unwrap();
        }
        store.commit_transaction().unwrap();

        // Batch 2
        store.begin_transaction().unwrap();
        for i in 50..100 {
            store
                .upsert_source(&format!("file:///{i}.txt"), &hash, 10, "txt", None, 1)
                .unwrap();
        }
        store.commit_transaction().unwrap();

        let stats = store.stats().unwrap();
        assert_eq!(stats.total_sources, 100);
    }

    #[test]
    fn test_upsert_batch_inserts_multiple_records() {
        let store = MetadataStore::open_in_memory().unwrap();
        let hash_a = [1u8; 32];
        let hash_b = [2u8; 32];

        store
            .upsert_batch(&[
                ("file:///a.txt", &hash_a, 100, "txt", Some(1000), 3),
                ("file:///b.rs", &hash_b, 200, "rs", Some(2000), 5),
            ])
            .unwrap();

        let stats = store.stats().unwrap();
        assert_eq!(stats.total_sources, 2);
        assert_eq!(stats.total_chunks, 8);

        // Verify individual records
        assert_eq!(
            store.check_source("file:///a.txt", &hash_a).unwrap(),
            Some(true)
        );
        assert_eq!(
            store.check_source("file:///b.rs", &hash_b).unwrap(),
            Some(true)
        );
    }

    #[test]
    fn test_upsert_batch_empty_input_does_nothing() {
        let store = MetadataStore::open_in_memory().unwrap();

        store.upsert_batch(&[]).unwrap();

        let stats = store.stats().unwrap();
        assert_eq!(stats.total_sources, 0);
        assert_eq!(stats.total_chunks, 0);
    }

    #[test]
    fn test_find_stale_sources_returns_empty_for_in_memory() {
        let store = MetadataStore::open_in_memory().unwrap();
        let hash = [0u8; 32];

        // Insert sources with URIs that don't exist on disk
        store
            .upsert_source(
                "file:///nonexistent/path/abc123.txt",
                &hash,
                10,
                "txt",
                None,
                1,
            )
            .unwrap();
        store
            .upsert_source(
                "file:///nonexistent/path/def456.rs",
                &hash,
                20,
                "rs",
                None,
                2,
            )
            .unwrap();

        // All file:// URIs point to non-existent paths, so all should be stale
        let stale = store.find_stale_sources().unwrap();
        assert_eq!(stale.len(), 2);
        assert!(stale.contains(&"file:///nonexistent/path/abc123.txt".to_string()));
        assert!(stale.contains(&"file:///nonexistent/path/def456.rs".to_string()));
    }

    #[test]
    fn test_find_stale_sources_non_file_uris_not_stale() {
        let store = MetadataStore::open_in_memory().unwrap();
        let hash = [0u8; 32];

        // Non-file URIs should never be considered stale
        store
            .upsert_source("https://example.com/doc", &hash, 10, "html", None, 1)
            .unwrap();
        store
            .upsert_source("s3://bucket/object", &hash, 10, "txt", None, 1)
            .unwrap();

        let stale = store.find_stale_sources().unwrap();
        assert!(stale.is_empty());
    }

    #[test]
    fn test_transaction_guard_begin_and_commit() {
        let store = MetadataStore::open_in_memory().unwrap();
        let hash = [0u8; 32];

        {
            let guard = TransactionGuard::begin(&store).unwrap();
            store
                .upsert_source("file:///guarded.txt", &hash, 10, "txt", None, 1)
                .unwrap();
            guard.commit().unwrap();
        }

        // Data should be persisted after commit
        assert_eq!(
            store.check_source("file:///guarded.txt", &hash).unwrap(),
            Some(true)
        );
    }

    #[test]
    fn test_transaction_guard_auto_rollback_on_drop() {
        let store = MetadataStore::open_in_memory().unwrap();
        let hash = [0u8; 32];

        {
            let _guard = TransactionGuard::begin(&store).unwrap();
            store
                .upsert_source("file:///dropped.txt", &hash, 10, "txt", None, 1)
                .unwrap();
            // guard drops here without commit -> rollback
        }

        // Data should NOT be persisted because guard was dropped
        assert!(store
            .check_source("file:///dropped.txt", &hash)
            .unwrap()
            .is_none());
    }

    #[test]
    fn test_transaction_guard_commit_and_reopen() {
        let store = MetadataStore::open_in_memory().unwrap();
        let hash = [0u8; 32];

        {
            let mut guard = TransactionGuard::begin(&store).unwrap();

            // Batch 1
            store
                .upsert_source("file:///batch1.txt", &hash, 10, "txt", None, 1)
                .unwrap();
            guard.commit_and_reopen().unwrap();

            // Batch 2 (new transaction after reopen)
            store
                .upsert_source("file:///batch2.txt", &hash, 20, "txt", None, 2)
                .unwrap();
            guard.commit().unwrap();
        }

        // Both batches should be persisted
        assert_eq!(
            store.check_source("file:///batch1.txt", &hash).unwrap(),
            Some(true)
        );
        assert_eq!(
            store.check_source("file:///batch2.txt", &hash).unwrap(),
            Some(true)
        );
        let stats = store.stats().unwrap();
        assert_eq!(stats.total_sources, 2);
        assert_eq!(stats.total_chunks, 3);
    }

    #[test]
    fn test_stats_returns_correct_file_type_counts() {
        let store = MetadataStore::open_in_memory().unwrap();
        let hash = [0u8; 32];

        store
            .upsert_source("file:///a.txt", &hash, 10, "txt", None, 1)
            .unwrap();
        store
            .upsert_source("file:///b.txt", &hash, 10, "txt", None, 2)
            .unwrap();
        store
            .upsert_source("file:///c.rs", &hash, 10, "rs", None, 3)
            .unwrap();
        store
            .upsert_source("file:///d.py", &hash, 10, "py", None, 4)
            .unwrap();

        let stats = store.stats().unwrap();
        assert_eq!(stats.total_sources, 4);
        assert_eq!(stats.total_chunks, 10);
        assert_eq!(stats.file_type_counts.len(), 3);
        assert_eq!(stats.file_type_counts["txt"], 2);
        assert_eq!(stats.file_type_counts["rs"], 1);
        assert_eq!(stats.file_type_counts["py"], 1);
        // index_size_bytes is computed externally, always 0 from MetadataStore
        assert_eq!(stats.index_size_bytes, 0);
    }

    #[test]
    fn test_batch_matches_individual_checks() {
        let store = MetadataStore::open_in_memory().unwrap();
        let hash_a = [1u8; 32];
        let hash_b = [2u8; 32];

        store
            .upsert_source("file:///a.txt", &hash_a, 10, "txt", None, 1)
            .unwrap();
        store
            .upsert_source("file:///b.txt", &hash_b, 20, "txt", None, 2)
            .unwrap();

        // Individual checks
        assert_eq!(
            store.check_source("file:///a.txt", &hash_a).unwrap(),
            Some(true)
        );
        assert_eq!(
            store.check_source("file:///a.txt", &hash_b).unwrap(),
            Some(false)
        );
        assert!(store
            .check_source("file:///c.txt", &hash_a)
            .unwrap()
            .is_none());

        // Batch check should give same results
        let hashes = store.load_all_hashes().unwrap();
        assert!(hashes
            .get("file:///a.txt")
            .is_some_and(|h| h.as_slice() == hash_a.as_slice()));
        assert!(hashes
            .get("file:///a.txt")
            .is_none_or(|h| h.as_slice() != hash_b.as_slice()));
        assert!(!hashes.contains_key("file:///c.txt"));
    }
}
